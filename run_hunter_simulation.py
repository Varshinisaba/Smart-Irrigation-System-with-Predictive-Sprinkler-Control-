"""
run_hunter_simulation.py
=========================
Run the complete novel Hunter MP Rotator simulation.

Place in your files (2) folder and run:
    python run_hunter_simulation.py

Flow (matches your diagram):
  crop+soil CSV + crop CSV
        ↓
  LSTM predicts moisture
        ↓
  ET deficit check
        ↓
  Moisture deficit check
        ↓
  Regression → sprinkler geometry
        ↓
  Fog scheduler
        ↓
  Execute Hunter MP Rotator
        ↓
  Store in water ledger → Test metrics
"""

import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from sprinkler.sprinkler_hunter import (
    HunterSprinklerController, CROP_PARAMS, PLOT_ASSIGNMENTS
)

os.makedirs("results", exist_ok=True)
os.makedirs("charts",  exist_ok=True)

print("=" * 65)
print("  HUNTER MP ROTATOR — Fog Irrigation Novel Simulation")
print("=" * 65)

# ── Setup controller ─────────────────────────────────────────────
controller = HunterSprinklerController()
controller.setup(PLOT_ASSIGNMENTS)

# ── Simulate 30 days ─────────────────────────────────────────────
print("\n[Running 30-day simulation...]")
print(f"\n  {'Day':<5} {'Plot':<10} {'Crop':<10} {'Deficit':>8} "
      f"{'Duration':>9} {'Vol(L)':>8} {'Saved(L)':>9} {'Model':<22}")
print("  " + "-" * 90)

np.random.seed(42)
N_DAYS   = 30
all_results = []

for day in range(1, N_DAYS + 1):
    for pa in PLOT_ASSIGNMENTS:
        pid   = pa["plot"]
        crop  = pa["crop"]
        stage = pa["stage"]
        cp    = CROP_PARAMS.get(crop, CROP_PARAMS["Wheat"])

        # Simulate LSTM moisture prediction
        current_m = np.random.uniform(cp["pwp"] + 0.01, cp["fc"] - 0.01)
        et_rate   = np.random.uniform(3.0, 7.0)

        # LSTM predicts drop
        predicted_m = current_m - np.random.uniform(0.02, 0.08)

        # Only irrigate if predicted moisture < optimal_low
        if predicted_m >= cp["optimal_low"]:
            continue

        result = controller.execute_irrigation(
            day             = day,
            plot_id         = pid,
            crop            = crop,
            stage           = stage,
            current_moisture= current_m,
            et_rate_mm_day  = et_rate,
            trigger         = "LSTM_FOG",
        )

        if result.get("irrigate"):
            all_results.append({
                "day":   day, "plot_id": pid, "crop": crop,
                **result
            })
            print(f"  {day:<5} {pid:<10} {crop:<10} "
                  f"{result['moisture_deficit']:>8.4f} "
                  f"{result['duration_min']:>8.1f}m "
                  f"{result['volume_L']:>8.1f} "
                  f"{result['saved_L']:>9.1f} "
                  f"{result['regression_model']:<22}")

# ── Water ledger summary ─────────────────────────────────────────
print("\n" + "=" * 65)
ledger_summary = controller.ledger.summary()
df_ledger = controller.ledger.to_dataframe()

print("  WATER LEDGER SUMMARY")
print("=" * 65)
print(f"  Total irrigation events : {ledger_summary.get('total_events', 0)}")
print(f"  Smart water used        : {ledger_summary.get('total_smart_L', 0):,.1f} L")
print(f"  Timer baseline          : {ledger_summary.get('total_timer_L', 0):,.1f} L")
print(f"  Water SAVED             : {ledger_summary.get('total_saved_L', 0):,.1f} L "
      f"({ledger_summary.get('saved_pct', 0):.1f}%)")
print(f"  Avg irrigation duration : {ledger_summary.get('avg_duration_min', 0):.1f} min")
print(f"  Avg sprinkler radius    : {ledger_summary.get('avg_radius_m', 0):.2f} m")

# Per crop breakdown
print("\n  Per-crop water usage:")
print(f"  {'Crop':<12} {'Events':>7} {'Smart_L':>10} "
      f"{'Timer_L':>10} {'Saved_L':>10} {'Saved%':>8}")
print("  " + "-" * 62)
for crop in df_ledger["crop"].unique():
    sub = df_ledger[df_ledger["crop"] == crop]
    smart = sub["smart_L"].sum()
    timer = sub["timer_L"].sum()
    saved = sub["saved_L"].sum()
    savep = saved / max(timer, 1e-6) * 100
    print(f"  {crop:<12} {len(sub):>7} {smart:>10.1f} "
          f"{timer:>10.1f} {saved:>10.1f} {savep:>7.1f}%")

# Regression model metrics
print("\n  Regression Model Comparison:")
print(f"  {'Model':<22} {'MAE_dur':>10} {'MAE_rad':>10} "
      f"{'R2_dur':>8} {'R2_rad':>8}")
print("  " + "-" * 62)
for name, m in controller.regression.metrics.items():
    best = " ← BEST" if name == controller.regression.best_model_name else ""
    print(f"  {name:<22} {m['mae_duration']:>10.3f} {m['mae_radius']:>10.3f} "
          f"{m['r2_duration']:>8.4f} {m['r2_radius']:>8.4f}{best}")

# Hunter model usage
print("\n  Hunter MP Model Selection:")
for pa in PLOT_ASSIGNMENTS:
    pid  = pa["plot"]
    crop = pa["crop"]
    s    = controller.sprinklers.get(pid)
    if s:
        print(f"  {pid} ({crop:8s}): {s.model_name}  "
              f"radius={s.radius_m:.1f}m  "
              f"flow={s.flow_L_min:.2f}L/min  "
              f"arc={s.arc_deg}°  "
              f"obstacles={len(s.obstacles)}")

# Save results
df_results = pd.DataFrame(all_results)
df_results.to_csv("results/hunter_events.csv", index=False)
df_ledger.to_csv("results/water_ledger.csv", index=False)

summary = {
    "hunter_models_used": list(set(
        controller.sprinklers[pa["plot"]].model_name
        for pa in PLOT_ASSIGNMENTS
    )),
    "best_regression_model": controller.regression.best_model_name,
    "regression_metrics":    controller.regression.metrics,
    "water_ledger":          ledger_summary,
    "per_crop": {
        crop: {
            "events":   int(len(df_ledger[df_ledger["crop"]==crop])),
            "smart_L":  float(df_ledger[df_ledger["crop"]==crop]["smart_L"].sum()),
            "saved_L":  float(df_ledger[df_ledger["crop"]==crop]["saved_L"].sum()),
        }
        for crop in df_ledger["crop"].unique()
    } if not df_ledger.empty else {},
    "novel_contributions": [
        "Hunter MP Rotator real specs (MP1000/MP2000/MP3000)",
        "Regression-based geometry: Linear + Ridge + Random Forest compared",
        "ET deficit + moisture deficit dual check before irrigation",
        "Obstacle-aware arc selection (coconut trees)",
        "Water ledger with per-crop tracking",
        "Fog scheduler integration end-to-end",
    ],
}

with open("results/hunter_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 65)
print("  Files saved:")
print("  results/hunter_events.csv")
print("  results/water_ledger.csv")
print("  results/hunter_summary.json")
print("\n  Now run: python generate_hunter_charts.py")
print("=" * 65)
