"""
End-to-End Simulation Runner
"""
import argparse, os, sys, json
import numpy as np
import pandas as pd
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
from sprinkler.sprinkler_hunter import HunterSprinklerController, CROP_PARAMS as HUNTER_CP, PLOT_ASSIGNMENTS
from simulation.dataset_generator import generate_full_dataset, PLOT_CONFIGS
# Combined CROP_PARAMS — includes both TN crops and your real CSV crops
CROP_PARAMS = {
    "Wheat":     {"fc": 0.34, "pwp": 0.14, "kc": 1.0,  "depletion_rate": 0.018},
    "Potato":    {"fc": 0.38, "pwp": 0.16, "kc": 1.1,  "depletion_rate": 0.020},
    "Carrot":    {"fc": 0.36, "pwp": 0.15, "kc": 0.9,  "depletion_rate": 0.018},
    "Tomato":    {"fc": 0.35, "pwp": 0.15, "kc": 1.05, "depletion_rate": 0.019},
    "Chilli":    {"fc": 0.33, "pwp": 0.14, "kc": 0.85, "depletion_rate": 0.017},
    "Coconut":   {"fc": 0.45, "pwp": 0.18, "kc": 1.1,  "depletion_rate": 0.015},
    "paddy":     {"fc": 0.40, "pwp": 0.20, "kc": 1.2,  "depletion_rate": 0.025},
    "sugarcane": {"fc": 0.38, "pwp": 0.18, "kc": 1.1,  "depletion_rate": 0.020},
    "groundnut": {"fc": 0.30, "pwp": 0.14, "kc": 0.8,  "depletion_rate": 0.018},
    "cotton":    {"fc": 0.32, "pwp": 0.15, "kc": 0.9,  "depletion_rate": 0.022},
}
from simulation.real_data_adapter import prepare_real_data
from lora_sim.lora_simulator      import FogGateway, EdgeNode, LoRaChannel
from fog.fog_scheduler            import FogIrrigationController, PlotState
from evaluation.sdg_metrics       import run_full_evaluation
 
 
def parse_args():
    p = argparse.ArgumentParser(description="Fog Irrigation Simulation")
    p.add_argument("--days",        type=int, default=30)
    p.add_argument("--plots",       type=int, default=8)
    p.add_argument("--skip-train",  action="store_true")
    p.add_argument("--output",      default="results")
    p.add_argument("--sensor-csv",  default=None)
    p.add_argument("--crop-csv",    default=None)
    return p.parse_args()
 
 
def simulate_communication_layer(df_plot, plot_id_int, gateway, distance_m):
    node = EdgeNode(plot_id=plot_id_int, distance_to_gw_m=distance_m, gateway=gateway)
    sent = received = 0
    for _, row in df_plot.iterrows():
        pkt = node.send_reading(row["soil_moisture"], row["temperature_c"])
        sent += 1
        received += int(pkt.received)
    return {"sent": sent, "received": received, "pdr": received / max(sent, 1)}
 
 
def simulate_fog_scheduling(df_all, controller, n_steps=48, timestamps=None):
    records = []
    timestamps = timestamps if timestamps is not None else sorted(df_all["timestamp"].unique())[:n_steps]
    last_irrigated = {}   # plot_id -> last irrigation timestamp index
    MIN_GAP_STEPS  = 8    # minimum 8 steps (4 hours) between irrigations per plot
    for ts in timestamps:
        snapshot = df_all[df_all["timestamp"] == ts]
        if snapshot.empty:
            continue
        plots = []
        for _, row in snapshot.iterrows():
            crop_name = row.get("crop", "paddy")
            crop_p = CROP_PARAMS.get(crop_name, {"fc": 0.35, "pwp": 0.15})
            ps = PlotState(
                plot_id       = row["plot_id"],
                moisture      = float(row["soil_moisture"]),
                moisture_pred = [float(row["soil_moisture"])] * 12,
                crop          = crop_name,
                fc            = crop_p["fc"],
                pwp           = crop_p["pwp"],
                area_m2       = 1000.0,
            )
            plots.append(ps)
        if not plots:
            continue
        rain_fcast = [float(row.get("rainfall_mm", 0)) * 2] * 12
        hour = pd.Timestamp(ts).hour
        try:
            decisions = controller.run_cycle(plots, rain_fcast, hour)
        except Exception as e:
            print(f"  [Scheduler] Cycle failed at {ts}: {e}")
            continue
        for idx, d in enumerate(decisions):
            pid = d.plot_id
            ts_idx = list(timestamps).index(ts)
            # Enforce minimum gap between irrigations
            if d.irrigate:
                last = last_irrigated.get(pid, -MIN_GAP_STEPS - 1)
                if ts_idx - last < MIN_GAP_STEPS:
                    d = type(d)(plot_id=pid, irrigate=False,
                                duration_s=0, volume_m3=0.0,
                                trigger="SKIP_GAP", timestamp=d.timestamp)
                else:
                    last_irrigated[pid] = ts_idx
            records.append({
                "timestamp":  ts,
                "plot_id":    d.plot_id,
                "irrigate":   d.irrigate,
                "duration_s": d.duration_s,
                "volume_m3":  d.volume_m3,
                "trigger":    d.trigger,
            })
    return pd.DataFrame(records)
 
 
def print_summary(lora_stats, schedule_df, eval_results):
    print("\n" + "="*55)
    print("  SIMULATION SUMMARY REPORT")
    print("="*55)
    avg_pdr = np.mean([v["pdr"] for v in lora_stats.values()])
    print(f"  LoRa PDR:               {avg_pdr:.1%}")
    if not schedule_df.empty:
        irr = schedule_df[schedule_df["irrigate"]].shape[0]
        vol = schedule_df["volume_m3"].sum()
        print(f"  Irrigation events:      {irr}")
        print(f"  Total water used:       {vol:.2f} m3")
    if eval_results:
        s6  = eval_results.get("sdg6_vs_timer", {})
        s2  = eval_results.get("sdg2_yield", {})
        lat = eval_results.get("latency", {})
        sz  = eval_results.get("model_size", {})
        print(f"  SDG6 water saved:       {s6.get('water_saved_pct',0):.1f}%")
        print(f"  SDG2 optimal band:      {s2.get('time_in_optimal_band_pct',0):.1f}%")
        print(f"  Model size (int8):      {sz.get('int8_tflite_kb',0)} KB")
        print(f"  Network reduction:      {eval_results.get('network_traffic',{}).get('traffic_reduction_pct',0):.1f}%")
    print("="*55)
 
 
def main():
    args   = parse_args()
    outdir = args.output
    os.makedirs(outdir, exist_ok=True)
    os.makedirs("data",   exist_ok=True)
    os.makedirs("models", exist_ok=True)
 
    print("\n" + "="*55)
    print("  Smart Irrigation Fog System - Full Simulation")
    print(f"  Days={args.days}  Plots={args.plots}")
    print("="*55)
 
    if args.sensor_csv and args.crop_csv:
        print("\n[1/6] Loading YOUR real datasets ...")
        df_all = prepare_real_data(
            sensor_csv = args.sensor_csv,
            crop_csv   = args.crop_csv,
            n_plots    = args.plots,
            output_dir = "data",
        )
    else:
        print("\n[1/6] Generating synthetic dataset ...")
        df_all = generate_full_dataset(output_dir="data", days=args.days)
 
    if not args.skip_train:
        print("\n[2/6] Training + compressing edge LSTM ...")
        try:
            from edge.lstm_edge import train_and_compress
            train_and_compress("data/plot_00.csv", output_dir="models")
        except ImportError:
            print("  TensorFlow not available - skipping.")
    else:
        print("\n[2/6] Skipping LSTM training (--skip-train)")
 
    print("\n[3/6] Simulating LoRa communication ...")
    channel    = LoRaChannel(seed=0)
    gateway    = FogGateway(channel=channel)
    lora_stats = {}
    plot_ids   = df_all["plot_id"].unique()[:args.plots]
    for i, pid in enumerate(plot_ids):
        sub  = df_all[df_all["plot_id"] == pid].head(96)
        dist = np.random.uniform(300, 2500)
        stats = simulate_communication_layer(sub, i, gateway, dist)
        lora_stats[pid] = stats
        print(f"  {pid}: PDR={stats['pdr']:.1%}  ({dist:.0f}m from GW)")
 
    print("\n[4/6] Running fog MPC+RL scheduler ...")
    controller  = FogIrrigationController(n_plots=args.plots)
    df_sample   = df_all[df_all["plot_id"].isin(plot_ids)].copy()
    # Use every 8th timestamp = every 4 hours (realistic fog check interval)
    all_ts     = sorted(df_sample["timestamp"].unique())
    sampled_ts = all_ts[::8]
    schedule_df = simulate_fog_scheduling(df_sample, controller, n_steps=len(sampled_ts), timestamps=sampled_ts)
    schedule_path = os.path.join(outdir, "schedule_decisions.csv")
    schedule_df.to_csv(schedule_path, index=False)
    print(f"  Schedule saved -> {schedule_path}  ({len(schedule_df)} decisions)")
 
    print("\n[5/6] Running SDG evaluation ...")
    smart_water_m3 = float(schedule_df["volume_m3"].sum()) if not schedule_df.empty else 0.0
    n_plots_actual = len(df_all["plot_id"].unique())
    timer_water_m3 = n_plots_actual * 30 * 3.0
    eval_results = run_full_evaluation(
        data_csv        = os.path.join("data", "all_plots.csv"),
        output_dir      = outdir,
        smart_water_m3  = smart_water_m3,
        timer_water_m3  = timer_water_m3,
    )
 
    print_summary(lora_stats, schedule_df, eval_results)
 
    # ── Step 6: Hunter MP Rotator simulation ─────────────────────
    # Filter PLOT_ASSIGNMENTS to only plots in your CSV
    filtered_assignments = [pa for pa in PLOT_ASSIGNMENTS if pa["plot"] in plot_ids]
    hunter = HunterSprinklerController()
    hunter.setup(filtered_assignments)
 
    plot_crop_map = {pa["plot"]: (pa["crop"], pa["stage"]) for pa in PLOT_ASSIGNMENTS}
 
    for pid in plot_ids:
        crop, stage = plot_crop_map.get(pid, ("Wheat", "tillering"))
        cp  = HUNTER_CP.get(crop, HUNTER_CP["Wheat"])
        sub = df_all[df_all["plot_id"] == pid].head(48)
        for _, row in sub.iterrows():
            current_m = float(row["soil_moisture"])
            et_rate   = float(row.get("et_mm_day", 4.5))
            hunter.execute_irrigation(
                day              = 1,
                plot_id          = pid,
                crop             = crop,
                stage            = stage,
                current_moisture = current_m,
                et_rate_mm_day   = et_rate,
                trigger          = "LSTM_FOG",
            )
 
    hunter_summary = hunter.ledger.summary()
    print(f"  Hunter events:     {hunter_summary.get('total_events', 0)}")
    print(f"  Smart water used:  {hunter_summary.get('total_smart_L', 0):.1f} L")
    print(f"  Timer baseline:    {hunter_summary.get('total_timer_L', 0):.1f} L")
    print(f"  Water saved:       {hunter_summary.get('total_saved_L', 0):.1f} L "
          f"({hunter_summary.get('saved_pct', 0):.1f}%)")
    hunter.ledger.to_dataframe().to_csv(
        os.path.join(outdir, "water_ledger.csv"), index=False)
    print(f"  Water ledger -> {outdir}/water_ledger.csv")
 
    summary = {
        "config":        {"days": args.days, "plots": args.plots},
        "lora_stats":    lora_stats,
        "schedule_rows": len(schedule_df),
        "evaluation":    eval_results,
        "hunter_mp":     hunter_summary,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    # ── Combined savings calculation ──────────────────────────────
    sdg6_saved_pct  = eval_results.get("sdg6_vs_timer", {}).get("water_saved_pct", 0)
    hunter_saved_pct = hunter_summary.get("saved_pct", 0)

    # Combined: what % is saved overall vs a pure timer baseline
    combined_saved_pct = round(
        (1 - (1 - sdg6_saved_pct / 100) * (1 - hunter_saved_pct / 100)) * 100, 1
    )

    print("\n" + "="*55)
    print("  COMBINED WATER SAVINGS")
    print("="*55)
    print(f"  Fog Scheduler saving:   {sdg6_saved_pct:.1f}%  (MPC+RL vs timer)")
    print(f"  Hunter MP saving:       {hunter_saved_pct:.1f}%  (smart duration vs fixed timer)")
    print(f"  Combined total saving:  {combined_saved_pct:.1f}%  (vs pure timer baseline)")
    print("="*55)

    summary["combined_water_saved_pct"] = combined_saved_pct
    summary["sdg6_saved_pct"]           = sdg6_saved_pct
    summary["hunter_saved_pct"]         = hunter_saved_pct

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Full summary -> {outdir}/summary.json")

 
if __name__ == "__main__":
    main()