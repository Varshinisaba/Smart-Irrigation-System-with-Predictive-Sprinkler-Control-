"""
Tamil Nadu Soil Moisture Simulation Dataset Generator
Generates realistic multi-plot, multi-sensor data for 4 crops common in TN:
  paddy, sugarcane, groundnut, cotton
Incorporates: seasonal rainfall, temperature cycles, irrigation events,
              sensor noise, and LoRa packet-loss simulation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ── Reproducibility ──────────────────────────────────────────────
np.random.seed(42)

# ── Tamil Nadu Climate Constants ─────────────────────────────────
TN_RAIN_MONTHS   = {6,7,8,9,10,11}          # SW + NE monsoon
TN_HOT_MONTHS    = {3,4,5}                   # Pre-monsoon heat
TN_BASE_TEMP     = 28.0                      # °C annual mean
TN_TEMP_AMP      = 6.0                       # seasonal amplitude

CROP_PARAMS = {
    "paddy":     {"fc": 0.40, "pwp": 0.20, "kc": 1.2, "depletion_rate": 0.025},
    "sugarcane": {"fc": 0.38, "pwp": 0.18, "kc": 1.1, "depletion_rate": 0.020},
    "groundnut": {"fc": 0.30, "pwp": 0.14, "kc": 0.8, "depletion_rate": 0.018},
    "cotton":    {"fc": 0.32, "pwp": 0.15, "kc": 0.9, "depletion_rate": 0.022},
}

PLOT_CONFIGS = [
    {"plot_id": f"plot_{i:02d}",
     "crop":    list(CROP_PARAMS.keys())[i % 4],
     "area_m2": np.random.randint(500, 2000),
     "lat":     10.5 + np.random.uniform(-0.5, 0.5),
     "lon":     77.5 + np.random.uniform(-0.5, 0.5)}
    for i in range(8)
]


def tamil_nadu_temp(doy: int) -> float:
    """Day-of-year → mean temperature (°C) for Tamil Nadu."""
    return TN_BASE_TEMP + TN_TEMP_AMP * np.sin(2 * np.pi * (doy - 60) / 365)


def rainfall_mm(month: int, rng: np.random.Generator) -> float:
    """Stochastic daily rainfall (mm)."""
    if month in TN_RAIN_MONTHS:
        return rng.exponential(8.0) if rng.random() < 0.35 else 0.0
    return rng.exponential(1.0) if rng.random() < 0.05 else 0.0


def et0_penman_simple(temp_c: float, doy: int) -> float:
    """Simplified Penman-Monteith ET₀ (mm/day)."""
    ra = 15 + 5 * np.sin(2 * np.pi * (doy - 80) / 365)   # extraterrestrial radiation proxy
    return max(0.1, 0.0023 * (temp_c + 17.8) * (ra ** 0.5) * 0.408)


def generate_plot_timeseries(
    plot_cfg: dict,
    start: datetime,
    days: int = 365,
    dt_minutes: int = 30,
    lora_loss_prob: float = 0.08,
) -> pd.DataFrame:
    """
    Simulate soil moisture + irrigation events for one plot.
    Returns a DataFrame at `dt_minutes` resolution.
    """
    rng = np.random.default_rng(seed=abs(hash(plot_cfg["plot_id"])) % 2**31)
    crop = CROP_PARAMS[plot_cfg["crop"]]

    fc, pwp, kc, drain = crop["fc"], crop["pwp"], crop["kc"], crop["depletion_rate"]
    theta = fc * 0.85          # initial moisture (fraction of saturation)

    steps_per_day   = 24 * 60 // dt_minutes
    total_steps     = days * steps_per_day

    timestamps, moistures, temps, rainfalls = [], [], [], []
    irrigations, et_vals, lora_received    = [], [], []

    for step in range(total_steps):
        ts   = start + timedelta(minutes=step * dt_minutes)
        doy  = ts.timetuple().tm_yday
        hour = ts.hour

        # ── Daily values (update at midnight) ────────────────────
        if hour == 0 and ts.minute == 0:
            temp_day  = tamil_nadu_temp(doy) + rng.normal(0, 1.5)
            rain_day  = rainfall_mm(ts.month, rng)
            et0_day   = et0_penman_simple(temp_day, doy)
            etc_day   = et0_day * kc / steps_per_day  # per-step
            rain_step = rain_day / steps_per_day

        # ── Soil moisture dynamics ────────────────────────────────
        theta -= etc_day + drain * rng.uniform(0.5, 1.5) / steps_per_day
        theta += rain_step * 0.6            # infiltration efficiency

        # Simple rule-based irrigation trigger (will be replaced by LSTM+MPC)
        irr_mm = 0.0
        if theta < pwp + 0.03:             # critical threshold
            irr_mm = (fc - theta) * 1000   # mm equivalent
            theta  = fc * rng.uniform(0.90, 0.98)

        theta = float(np.clip(theta, pwp * 0.95, fc))

        # ── Sensor noise ─────────────────────────────────────────
        noise = rng.normal(0, 0.004)
        temp_noise = temp_day + 2 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.3)

        # ── LoRa packet loss simulation ───────────────────────────
        received = 0 if rng.random() < lora_loss_prob else 1

        timestamps.append(ts)
        moistures.append(round(theta + noise, 4))
        temps.append(round(temp_noise, 2))
        rainfalls.append(round(rain_step, 3))
        irrigations.append(round(irr_mm, 2))
        et_vals.append(round(etc_day * steps_per_day, 3))
        lora_received.append(received)

    df = pd.DataFrame({
        "timestamp":       timestamps,
        "plot_id":         plot_cfg["plot_id"],
        "crop":            plot_cfg["crop"],
        "soil_moisture":   moistures,   # vol. water content (m³/m³)
        "temperature_c":   temps,
        "rainfall_mm":     rainfalls,
        "irrigation_mm":   irrigations,
        "et_mm_day":       et_vals,
        "lora_received":   lora_received,
        "lat":             plot_cfg["lat"],
        "lon":             plot_cfg["lon"],
    })
    return df


def generate_full_dataset(output_dir: str = "data", days: int = 365):
    os.makedirs(output_dir, exist_ok=True)
    start = datetime(2023, 6, 1)
    all_dfs = []

    for cfg in PLOT_CONFIGS:
        print(f"  Generating {cfg['plot_id']} ({cfg['crop']}) ...")
        df = generate_plot_timeseries(cfg, start, days=days)
        df.to_csv(f"{output_dir}/{cfg['plot_id']}.csv", index=False)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(f"{output_dir}/all_plots.csv", index=False)
    print(f"\n✅  Dataset saved → {output_dir}/  ({len(combined):,} rows across {len(PLOT_CONFIGS)} plots)")
    return combined


if __name__ == "__main__":
    generate_full_dataset(output_dir="data")
