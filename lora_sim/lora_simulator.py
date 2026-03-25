"""
LoRa Communication Simulator
==============================
Simulates the LoRa PHY + MAC layer between:
  Edge nodes (ESP32) → Fog gateway (RPi/server)
  Fog gateway        → Edge nodes  (downlink commands)

Models:
  • SF7–SF12 spreading factor selection based on distance
  • RSSI / SNR attenuation (log-distance path loss + shadowing)
  • Duty-cycle constraint (1% EU868 / TTN)
  • CSMA collision avoidance (simple random back-off)
  • Packet loss (BER from SNR)
  • LoRaWAN-style payload framing (FHDR + FPort + FRMPayload)
"""

import struct, time, random, math, hashlib
from dataclasses import dataclass, field
from typing import Optional, List
from enum import IntEnum
import numpy as np


# ── LoRa PHY Parameters ──────────────────────────────────────────

class SF(IntEnum):
    SF7  = 7
    SF8  = 8
    SF9  = 9
    SF10 = 10
    SF11 = 11
    SF12 = 12

SF_BITRATE = {SF.SF7: 5470, SF.SF8: 3125, SF.SF9: 1758,
              SF.SF10: 977, SF.SF11: 537, SF.SF12: 293}   # bps @ BW125

LORA_FREQ_MHZ   = 865.0    # IN865 band (India)
TX_POWER_DBM    = 14        # max legal
NOISE_FLOOR_DBM = -120
PATH_LOSS_EXP   = 2.7       # urban/semi-rural
REF_DIST_M      = 1.0
SHADOWING_STD   = 6.0       # dB log-normal shadowing

DUTY_CYCLE_MAX  = 0.01      # 1 % in 1-hour window
MAX_PAYLOAD_B   = 51        # SF10 LoRaWAN limit


# ── Packet Structures ─────────────────────────────────────────────

@dataclass
class LoRaPacket:
    dev_eui:   str
    fcnt:      int
    fport:     int               # 1=sensor data, 2=irrigation cmd, 3=ack
    payload:   bytes
    sf:        SF = SF.SF9
    rssi:      float = 0.0
    snr:       float = 0.0
    received:  bool = True
    air_time_s: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __repr__(self):
        return (f"LoRaPacket(dev={self.dev_eui[-4:]}, fcnt={self.fcnt}, "
                f"port={self.fport}, rssi={self.rssi:.1f}dBm, "
                f"rx={'✓' if self.received else '✗'})")


# ── Sensor Payload Codec (compact binary, fits 51 bytes) ──────────

def encode_sensor_payload(plot_id: int, moisture: float,
                           temp: float, battery_mv: int) -> bytes:
    """
    Pack: [plot_id u8][moisture u16 ×10000][temp s16 ×100][batt u16]
    Total: 7 bytes
    """
    m_raw  = int(np.clip(moisture, 0, 1) * 10000)
    t_raw  = int(temp * 100)
    return struct.pack(">BHhH", plot_id, m_raw, t_raw, battery_mv)


def decode_sensor_payload(data: bytes) -> dict:
    plot_id, m_raw, t_raw, batt = struct.unpack(">BHhH", data)
    return {"plot_id": plot_id,
            "moisture": m_raw / 10000,
            "temperature_c": t_raw / 100,
            "battery_mv": batt}


def encode_irrigation_cmd(plot_id: int, valve_open: bool,
                           duration_s: int) -> bytes:
    """
    Pack: [plot_id u8][valve u8][duration u16]  = 4 bytes
    """
    return struct.pack(">BBH", plot_id, int(valve_open), duration_s)


def decode_irrigation_cmd(data: bytes) -> dict:
    plot_id, valve, dur = struct.unpack(">BBH", data)
    return {"plot_id": plot_id, "valve_open": bool(valve), "duration_s": dur}


# ── Channel Model ─────────────────────────────────────────────────

class LoRaChannel:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def path_loss_db(self, distance_m: float) -> float:
        """Log-distance model with log-normal shadowing."""
        if distance_m < 1:
            distance_m = 1.0
        pl = (20 * math.log10(4 * math.pi * REF_DIST_M * LORA_FREQ_MHZ * 1e6 / 3e8)
              + 10 * PATH_LOSS_EXP * math.log10(distance_m / REF_DIST_M))
        shadow = self.rng.normal(0, SHADOWING_STD)
        return pl + shadow

    def received_power_dbm(self, distance_m: float) -> float:
        return TX_POWER_DBM - self.path_loss_db(distance_m)

    def snr_db(self, rx_power_dbm: float) -> float:
        return rx_power_dbm - NOISE_FLOOR_DBM

    def packet_error_rate(self, snr_db: float, sf: SF) -> float:
        """Empirical PER vs SNR threshold per SF (from LoRa Alliance specs)."""
        snr_thresholds = {SF.SF7: -7.5, SF.SF8: -10, SF.SF9: -12.5,
                          SF.SF10: -15, SF.SF11: -17.5, SF.SF12: -20}
        margin = snr_db - snr_thresholds[sf]
        if margin > 10:    return 0.001
        elif margin > 5:   return 0.01
        elif margin > 0:   return 0.05 + 0.05 * (5 - margin) / 5
        elif margin > -5:  return 0.30 + 0.20 * (-margin) / 5
        else:              return 0.95

    def air_time_s(self, payload_bytes: int, sf: SF, bw_khz: int = 125) -> float:
        """LoRa time-on-air (Semtech AN1200.13 formula)."""
        cr = 1     # coding rate 4/5
        de = 1 if sf >= SF.SF11 else 0
        n_preamble = 8
        t_sym = (2 ** sf) / (bw_khz * 1000)
        n_payload = max(8,
            8 + math.ceil((8 * payload_bytes - 4 * sf + 28 + 16) /
                          (4 * (sf - 2 * de))) * (cr + 4))
        t_preamble = (n_preamble + 4.25) * t_sym
        t_payload  = n_payload * t_sym
        return t_preamble + t_payload


# ── Gateway / Fog Node ────────────────────────────────────────────

class FogGateway:
    """Simulated LoRa gateway at fog node."""

    def __init__(self, gateway_id: str = "GW_TN_01",
                 position_m: tuple = (0, 0),
                 channel: Optional[LoRaChannel] = None):
        self.gw_id      = gateway_id
        self.position   = position_m
        self.channel    = channel or LoRaChannel()
        self.rx_buffer: List[LoRaPacket] = []
        self.tx_log:    List[LoRaPacket] = []
        self.duty_usage = 0.0          # seconds used in rolling hour
        self._hour_start = time.time()

    def select_sf(self, distance_m: float) -> SF:
        """ADR: choose minimum SF that meets link budget."""
        rx = self.channel.received_power_dbm(distance_m)
        for sf in [SF.SF7, SF.SF8, SF.SF9, SF.SF10, SF.SF11, SF.SF12]:
            snr  = self.channel.snr_db(rx)
            per  = self.channel.packet_error_rate(snr, sf)
            if per < 0.05:
                return sf
        return SF.SF12

    def receive(self, packet: LoRaPacket, distance_m: float) -> LoRaPacket:
        """Simulate receiving a packet from an edge node."""
        rx_dbm = self.channel.received_power_dbm(distance_m)
        snr    = self.channel.snr_db(rx_dbm)
        per    = self.channel.packet_error_rate(snr, packet.sf)
        at     = self.channel.air_time_s(len(packet.payload), packet.sf)

        packet.rssi      = round(rx_dbm, 1)
        packet.snr       = round(snr, 1)
        packet.received  = (random.random() > per)
        packet.air_time_s = at

        if packet.received:
            self.rx_buffer.append(packet)
        return packet

    def transmit_cmd(self, plot_id: int, valve_open: bool,
                     duration_s: int, distance_m: float) -> LoRaPacket:
        """Send irrigation command downlink to edge node."""
        payload = encode_irrigation_cmd(plot_id, valve_open, duration_s)
        sf      = self.select_sf(distance_m)
        at      = self.channel.air_time_s(len(payload), sf)

        # Duty-cycle check
        self._refresh_duty(at)

        pkt = LoRaPacket(
            dev_eui=f"DEV_{plot_id:04d}",
            fcnt=len(self.tx_log),
            fport=2,
            payload=payload,
            sf=sf,
            air_time_s=at,
        )
        self.tx_log.append(pkt)
        return pkt

    def _refresh_duty(self, air_time_s: float):
        now = time.time()
        if now - self._hour_start > 3600:
            self.duty_usage  = 0
            self._hour_start = now
        if self.duty_usage + air_time_s > 3600 * DUTY_CYCLE_MAX:
            raise RuntimeError(f"Duty-cycle limit exceeded: {self.duty_usage:.1f}s used")
        self.duty_usage += air_time_s

    def flush_buffer(self) -> List[dict]:
        """Return decoded payloads from received packets."""
        decoded = []
        for pkt in self.rx_buffer:
            if pkt.fport == 1:
                try:
                    d = decode_sensor_payload(pkt.payload)
                    d["timestamp"] = pkt.timestamp
                    d["rssi"]      = pkt.rssi
                    d["snr"]       = pkt.snr
                    decoded.append(d)
                except Exception:
                    pass
        self.rx_buffer.clear()
        return decoded

    def stats(self) -> dict:
        total = len(self.rx_buffer) + sum(1 for _ in self.tx_log)
        rx_ok = len([p for p in self.rx_buffer if p.received])
        return {
            "gateway": self.gw_id,
            "rx_total": len(self.rx_buffer),
            "rx_success": rx_ok,
            "pdr": rx_ok / max(len(self.rx_buffer), 1),
            "duty_used_s": round(self.duty_usage, 2),
        }


# ── Edge Node Transmitter ─────────────────────────────────────────

class EdgeNode:
    """Simulates LoRa uplink from ESP32 edge node."""

    def __init__(self, plot_id: int, distance_to_gw_m: float,
                 gateway: FogGateway,
                 sample_interval_s: int = 1800):   # 30 min
        self.plot_id  = plot_id
        self.dev_eui  = f"DEV_{plot_id:04X}AABBCC01"
        self.distance = distance_to_gw_m
        self.gw       = gateway
        self.fcnt     = 0
        self.interval = sample_interval_s
        self.channel  = gateway.channel

    def send_reading(self, moisture: float, temp: float,
                     battery_mv: int = 3700) -> LoRaPacket:
        payload = encode_sensor_payload(self.plot_id, moisture, temp, battery_mv)
        sf      = self.gw.select_sf(self.distance)
        pkt     = LoRaPacket(dev_eui=self.dev_eui, fcnt=self.fcnt,
                             fport=1, payload=payload, sf=sf)
        self.fcnt += 1
        result = self.gw.receive(pkt, self.distance)
        return result


# ── Quick Smoke-Test ──────────────────────────────────────────────

if __name__ == "__main__":
    channel = LoRaChannel(seed=42)
    gw      = FogGateway(channel=channel)

    nodes = [EdgeNode(i, distance_to_gw_m=np.random.uniform(200, 3000),
                      gateway=gw) for i in range(8)]

    print("Simulating 10 uplink transmissions per node ...\n")
    results = {"sent": 0, "received": 0}
    for _ in range(10):
        for node in nodes:
            moisture = np.random.uniform(0.15, 0.40)
            temp     = np.random.uniform(26, 38)
            pkt = node.send_reading(moisture, temp)
            results["sent"] += 1
            results["received"] += int(pkt.received)

    pdr = results["received"] / results["sent"]
    print(f"  Sent: {results['sent']}  |  Received: {results['received']}  |  PDR: {pdr:.1%}")
    print(f"\n  Gateway stats: {gw.stats()}")

    print("\nDecoded buffer sample:")
    for d in gw.flush_buffer()[:3]:
        print(" ", d)
