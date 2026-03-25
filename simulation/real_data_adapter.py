"""
Real Data Adapter
==================
Cleans and integrates the two uploaded datasets into the fog irrigation pipeline:

  1. SmartIrrigationDataDerive.csv
       - Raw sensor readings: temperature, pressure, soil moisture (ADC), class labels
       - Soil moisture is raw capacitive ADC (0–480): higher = drier
       - Calibration: Very Wet ≈ 170 ADC, Very Dry ≈ 360 ADC
       - Maps to VWC (m³/m³) using linear calibration

  2. cropdata_updated.csv
       - Crop-specific MOI (Moisture Optimum Index 1–100) thresholds
       - 5 crops × 7 soil types × 8 growth stages
       - result: 0=no irrigation, 1=irrigate, 2=urgent irrigate
       - Used to set per-crop FC/PWP bounds and irrigation triggers
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


# ── Calibration Constants (derived from class label analysis) ────
ADC_VERY_WET  = 170    # mean ADC for "Very Wet"  class
ADC_VERY_DRY  = 359    # mean ADC for "Very Dry"  class
VWC_VERY_WET  = 0.38   # field capacity proxy
VWC_VERY_DRY  = 0.18   # permanent wilting point proxy

# Crop FC/PWP from agronomic literature + MOI data calibration
CROP_BOUNDS = {
    "Wheat":   {"fc": 0.34, "pwp": 0.14, "kc": 1.0},
    "Potato":  {"fc": 0.38, "pwp": 0.16, "kc": 1.1},
    "Carrot":  {"fc": 0.36, "pwp": 0.15, "kc": 0.9},
    "Tomato":  {"fc": 0.35, "pwp": 0.15, "kc": 1.05},
    "Chilli":  {"fc": 0.33, "pwp": 0.14, "kc": 0.85},
    # TN crops added for compatibility
    "paddy":     {"fc": 0.40, "pwp": 0.20, "kc": 1.2},
    "sugarcane": {"fc": 0.38, "pwp": 0.18, "kc": 1.1},
    "groundnut": {"fc": 0.30, "pwp": 0.14, "kc": 0.8},
    "cotton":    {"fc": 0.32, "pwp": 0.15, "kc": 0.9},
}


# ══════════════════════════════════════════════════════════════════
# 1 ▸ CLEAN SmartIrrigationDataDerive
# ══════════════════════════════════════════════════════════════════

def adc_to_vwc(adc: float) -> float:
    """
    Convert raw capacitive ADC reading to volumetric water content.
    Capacitive sensors: low ADC = wet, high ADC = dry (inverted).
    Linear interpolation between calibration points.
    """
    vwc = VWC_VERY_WET + (VWC_VERY_DRY - VWC_VERY_WET) * \
          (adc - ADC_VERY_WET) / (ADC_VERY_DRY - ADC_VERY_WET)
    return float(np.clip(vwc, 0.10, 0.45))


def load_sensor_data(path: str) -> pd.DataFrame:
    """
    Load, clean, and calibrate the SmartIrrigation sensor dataset.
    Returns a DataFrame with proper timestamps and VWC values.
    """
    df = pd.read_csv(path)

    # ── Fix altitude column (has trailing '-') ───────────────────
    df["altitude"] = df["altitude"].astype(str).str.replace("-", "", regex=False)
    df["altitude"] = pd.to_numeric(df["altitude"], errors="coerce").fillna(12.0)

    # ── Fix temperature outliers (178.7°C = sensor fault) ────────
    temp_median = df.loc[df["temperature"] < 60, "temperature"].median()
    df.loc[df["temperature"] > 60, "temperature"] = temp_median

    # ── Fix negative soil moisture (sensor glitch) ────────────────
    df.loc[df["soilmiosture"] < 0, "soilmiosture"] = df["soilmiosture"].median()

    # ── Convert ADC → VWC ────────────────────────────────────────
    df["vwc"] = df["soilmiosture"].apply(adc_to_vwc)

    # ── Build proper timestamp (all same date → synthesize time) ──
    # Original data is all 2022-10-08, so we create a synthetic
    # 30-minute time series from the sequential readings
    base_time = datetime(2022, 10, 8, 0, 0, 0)
    df = df.sort_values("id").reset_index(drop=True)
    df["timestamp"] = [base_time + timedelta(minutes=30 * i) for i in range(len(df))]

    # ── Rename for consistency ────────────────────────────────────
    df = df.rename(columns={
        "temperature":  "temperature_c",
        "soilmiosture": "soil_adc",
        "pressure":     "pressure_hpa",
    })

    # ── Keep useful columns ───────────────────────────────────────
    df = df[["timestamp", "id", "temperature_c", "pressure_hpa",
             "altitude", "soil_adc", "vwc", "class", "status"]]

    print(f"  Sensor data: {len(df)} rows, VWC range: "
          f"{df['vwc'].min():.3f}–{df['vwc'].max():.3f}")
    return df


# ══════════════════════════════════════════════════════════════════
# 2 ▸ CLEAN cropdata
# ══════════════════════════════════════════════════════════════════

def load_crop_data(path: str) -> pd.DataFrame:
    """
    Load and clean the crop threshold dataset.
    Returns crop × stage × soil_type irrigation thresholds.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "crop ID":        "crop",
        "soil_type":      "soil_type",
        "Seedling Stage": "growth_stage",
        "MOI":            "moi",
        "temp":           "optimal_temp",
        "humidity":       "optimal_humidity",
        "result":         "irrigation_label",
    })

    # result: 0=no irrigation needed, 1=irrigate, 2=urgent
    df["irrigate"]        = df["irrigation_label"] >= 1
    df["urgent_irrigate"] = df["irrigation_label"] == 2

    # Normalize MOI (1–100) to VWC range per crop
    for crop in df["crop"].unique():
        bounds = CROP_BOUNDS.get(crop, {"fc": 0.35, "pwp": 0.15})
        mask   = df["crop"] == crop
        df.loc[mask, "moi_vwc"] = (
            bounds["pwp"] +
            (df.loc[mask, "moi"] / 100) * (bounds["fc"] - bounds["pwp"])
        )

    print(f"  Crop data: {len(df)} rows, "
          f"{df['crop'].nunique()} crops, "
          f"{df['growth_stage'].nunique()} stages")
    return df


# ══════════════════════════════════════════════════════════════════
# 3 ▸ BUILD MULTI-PLOT DATASET FOR SIMULATION
# ══════════════════════════════════════════════════════════════════

def build_plot_dataset(sensor_df: pd.DataFrame,
                        crop_df:   pd.DataFrame,
                        n_plots:   int = 8,
                        output_dir: str = "data") -> pd.DataFrame:
    """
    Distribute the sensor readings across n_plots, each assigned a
    different crop and soil type from the crop dataset.
    Augments data to 365 days using seasonal patterns.
    """
    os.makedirs(output_dir, exist_ok=True)

    crops      = crop_df["crop"].unique()
    soil_types = crop_df["soil_type"].unique()

    all_dfs = []
    rows_per_plot = len(sensor_df)

    for i in range(n_plots):
        crop      = crops[i % len(crops)]
        soil      = soil_types[i % len(soil_types)]
        bounds    = CROP_BOUNDS.get(crop, {"fc": 0.35, "pwp": 0.15, "kc": 1.0})
        plot_id   = f"plot_{i:02d}"

        # ── Use real sensor readings as base ──────────────────────
        plot_df = sensor_df.copy()
        plot_df["plot_id"]  = plot_id
        plot_df["crop"]     = crop
        plot_df["soil_type"] = soil

        # ── Add small per-plot VWC variation (soil heterogeneity) ─
        rng   = np.random.default_rng(seed=i * 7)
        noise = rng.normal(0, 0.008, len(plot_df))
        plot_df["soil_moisture"] = np.clip(plot_df["vwc"] + noise,
                                            bounds["pwp"] * 0.9,
                                            bounds["fc"])

        # ── Augment to ~365 days by tiling + seasonal scaling ─────
        n_tiles = max(1, int(365 * 48 / rows_per_plot))
        tiled   = pd.concat([plot_df] * n_tiles, ignore_index=True)
        tiled   = tiled.iloc[:365 * 48].copy()   # exactly 365 days × 48 steps/day

        # Rebuild timestamps
        base = datetime(2023, 1, 1)
        tiled["timestamp"] = [base + timedelta(minutes=30 * j) for j in range(len(tiled))]

        # Seasonal moisture modulation (monsoon wetter, summer drier)
        doys = np.array([t.timetuple().tm_yday for t in tiled["timestamp"]])
        seasonal = 0.03 * np.sin(2 * np.pi * (doys - 150) / 365)  # peak in June
        tiled["soil_moisture"] = np.clip(
            tiled["soil_moisture"] + seasonal, bounds["pwp"], bounds["fc"])

        # ── Rainfall proxy from pressure drop ────────────────────
        tiled["rainfall_mm"] = np.where(
            tiled["pressure_hpa"] < tiled["pressure_hpa"].quantile(0.15), 5.0, 0.0)

        # ── ET proxy from temperature ──────────────────────────────
        tiled["et_mm_day"] = (0.0023 * (tiled["temperature_c"] + 17.8)
                               * np.sqrt(15) * 0.408).clip(lower=0.5)

        # ── Irrigation events (from crop thresholds) ───────────────
        crop_thresh = crop_df[crop_df["crop"] == crop]["moi_vwc"].quantile(0.3)
        tiled["irrigation_mm"] = np.where(
            tiled["soil_moisture"] < crop_thresh, 15.0, 0.0)

        # ── LoRa packet loss simulation ────────────────────────────
        tiled["lora_received"] = rng.random(len(tiled)) > 0.08

        keep = ["timestamp", "plot_id", "crop", "soil_type",
                "soil_moisture", "temperature_c", "pressure_hpa",
                "rainfall_mm", "et_mm_day", "irrigation_mm",
                "lora_received", "class", "status"]
        tiled = tiled[keep]

        tiled.to_csv(f"{output_dir}/{plot_id}.csv", index=False)
        all_dfs.append(tiled)
        print(f"  {plot_id} ({crop}, {soil}): {len(tiled)} rows, "
              f"VWC={tiled['soil_moisture'].mean():.3f}±{tiled['soil_moisture'].std():.3f}")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(f"{output_dir}/all_plots.csv", index=False)
    print(f"\n  ✅ Combined dataset: {len(combined):,} rows → {output_dir}/all_plots.csv")
    return combined


# ══════════════════════════════════════════════════════════════════
# 4 ▸ CROP THRESHOLD LOOKUP (used by fog scheduler)
# ══════════════════════════════════════════════════════════════════

def get_irrigation_threshold(crop_df: pd.DataFrame,
                               crop: str,
                               growth_stage: str = None,
                               soil_type: str = None) -> dict:
    """
    Look up optimal moisture range for a crop/stage/soil combination.
    Returns dict with pwp, fc, irrigate_below, urgent_below thresholds.
    """
    mask = crop_df["crop"] == crop
    if growth_stage:
        mask &= crop_df["growth_stage"] == growth_stage
    if soil_type:
        mask &= crop_df["soil_type"] == soil_type

    subset = crop_df[mask]
    if subset.empty:
        subset = crop_df[crop_df["crop"] == crop]

    bounds = CROP_BOUNDS.get(crop, {"fc": 0.35, "pwp": 0.15})

    # MOI thresholds per irrigation label
    irrigate_moi = subset[subset["irrigation_label"] >= 1]["moi"].min() if len(subset) else 40
    urgent_moi   = subset[subset["irrigation_label"] == 2]["moi"].min() if len(subset) else 20

    return {
        "crop":             crop,
        "fc":               bounds["fc"],
        "pwp":              bounds["pwp"],
        "irrigate_below":   bounds["pwp"] + (irrigate_moi / 100) * (bounds["fc"] - bounds["pwp"]),
        "urgent_below":     bounds["pwp"] + (urgent_moi  / 100) * (bounds["fc"] - bounds["pwp"]),
        "kc":               bounds.get("kc", 1.0),
    }


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def prepare_real_data(sensor_csv: str, crop_csv: str,
                       n_plots: int = 8,
                       output_dir: str = "data") -> pd.DataFrame:
    print("\n  Loading & cleaning sensor data ...")
    sensor_df = load_sensor_data(sensor_csv)

    print("  Loading & cleaning crop threshold data ...")
    crop_df = load_crop_data(crop_csv)

    print(f"\n  Building {n_plots}-plot simulation dataset ...")
    combined = build_plot_dataset(sensor_df, crop_df, n_plots, output_dir)

    return combined


if __name__ == "__main__":
    prepare_real_data(
        sensor_csv="SmartIrrigationDataDerive__1_.csv",
        crop_csv="cropdata_updated__1_.csv",
        n_plots=8,
        output_dir="data"
    )
