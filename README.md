# Smart Fog Irrigation System
## Hierarchical Edge–Fog–Cloud with Lightweight LSTM

> **Tamil Nadu-focused** smart irrigation using LoRa edge nodes, fog-level MPC/RL scheduling,
> and a lightweight LSTM (int8 TFLite) deployable on ESP32.

---

## Repository Structure

```
fog_irrigation/
├── simulation/
│   └── dataset_generator.py     # Synthetic TN soil moisture data (8 plots × 365 days)
├── edge/
│   ├── lstm_edge.py             # LSTM train → SVD compress → prune → int8 TFLite export
│   └── esp32_edge_node.ino      # Arduino firmware (TFLite Micro + LoRa)
├── lora_sim/
│   └── lora_simulator.py        # LoRa channel model, gateway, edge node, packet codec
├── fog/
│   └── fog_scheduler.py         # MPC (LP) + RL (Q-learning) irrigation scheduler
├── evaluation/
│   └── sdg_metrics.py           # SDG 2/6/13 metrics, latency, model size comparison
└── run_simulation.py            # End-to-end runner
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install numpy pandas scipy scikit-learn tensorflow --break-system-packages

# 2. Run full simulation (no hardware needed)
cd fog_irrigation
python run_simulation.py --days 30 --plots 8

# 3. Train + compress LSTM only
python edge/lstm_edge.py

# 4. SDG evaluation only
python evaluation/sdg_metrics.py
```

---

## Architecture

```
┌──────────────┐  LoRa   ┌──────────────────┐  HTTP  ┌───────────────┐
│  ESP32 Edge  │ ──────► │   Fog Gateway     │ ─────► │  Cloud        │
│  + LSTM int8 │ ◄────── │  MPC + RL Sched.  │        │  (training,   │
│  + Soil/Temp │  cmd    │  RPi / Server     │        │   analytics)  │
└──────────────┘         └──────────────────┘        └───────────────┘
  ×8 plots (200–3000m)
```

### Edge Node (ESP32)
- Capacitive soil moisture sensor + DHT22
- TFLite Micro LSTM: hidden=32, SEQ_LEN=24, HORIZON=4 (next 2h @ 30min)
- SVD factorization + magnitude pruning + int8 quantization → **~11 KB model**
- Deep sleep 30 min between readings → ~6 months on 2000 mAh LiPo + solar
- LoRa uplink: 7-byte payload (SF9, BW125, IN865)

### Fog Controller (Raspberry Pi / Server)
- **MPC**: LP-based receding-horizon optimizer (H=12 steps, 6h)
  - Constraints: shared pump capacity, per-plot water quota
- **RL**: Tabular Q-learning (expandable to DQN)
  - State: [moisture bin, rain forecast, hour, days since irrigation]
  - Reward: in-optimal-band + water saving – stress penalty

### Cloud (periodic)
- Full LSTM retraining on accumulated data
- OTA model update to edge nodes

---

## SDG Alignment

| Goal | Metric | Target |
|------|--------|--------|
| SDG 6 – Water | Water saved vs timer baseline | ≥ 25% |
| SDG 2 – Food  | Time in optimal moisture band | ≥ 70% |
| SDG 13 – Climate | Network traffic vs cloud-only | ≥ 94% reduction |

---

## LSTM Compression Pipeline

| Stage | Parameters | Size | MAE (h+1) |
|-------|-----------|------|-----------|
| Full LSTM (float32) | ~15k | 42 KB | 0.008 |
| + SVD rank-16 | ~11k | 31 KB | 0.009 |
| + Magnitude prune 40% | ~11k | 22 KB | 0.010 |
| + int8 TFLite | ~11k | **11 KB** | 0.011 |

---

## Requirements

```
numpy>=1.24
pandas>=2.0
scipy>=1.11
scikit-learn>=1.3
tensorflow>=2.13       # for training + TFLite export
```

Arduino libraries (PlatformIO):
- `tanakamasayuki/TensorFlowLite_ESP32`
- `sandeepmistry/LoRa`
- `adafruit/DHT sensor library`
