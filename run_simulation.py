"""
End-to-End Simulation Runner
"""
import argparse, os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.dataset_generator import generate_full_dataset, PLOT_CONFIGS, CROP_PARAMS
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


def simulate_fog_scheduling(df_all, controller, n_steps=48):
    records = []
    timestamps = sorted(df_all["timestamp"].unique())[:n_steps]
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
        for d in decisions:
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
        print("\n[1/5] Loading YOUR real datasets ...")
        df_all = prepare_real_data(
            sensor_csv = args.sensor_csv,
            crop_csv   = args.crop_csv,
            n_plots    = args.plots,
            output_dir = "data",
        )
    else:
        print("\n[1/5] Generating synthetic dataset ...")
        df_all = generate_full_dataset(output_dir="data", days=args.days)

    if not args.skip_train:
        print("\n[2/5] Training + compressing edge LSTM ...")
        try:
            from edge.lstm_edge import train_and_compress
            train_and_compress("data/plot_00.csv", output_dir="models")
        except ImportError:
            print("  TensorFlow not available - skipping.")
    else:
        print("\n[2/5] Skipping LSTM training (--skip-train)")

    print("\n[3/5] Simulating LoRa communication ...")
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

    print("\n[4/5] Running fog MPC+RL scheduler ...")
    controller  = FogIrrigationController(n_plots=args.plots)
    df_sample   = df_all[df_all["plot_id"].isin(plot_ids)].copy()
    schedule_df = simulate_fog_scheduling(df_sample, controller, n_steps=48)
    schedule_path = os.path.join(outdir, "schedule_decisions.csv")
    schedule_df.to_csv(schedule_path, index=False)
    print(f"  Schedule saved -> {schedule_path}  ({len(schedule_df)} decisions)")

    print("\n[5/5] Running SDG evaluation ...")
    eval_results = run_full_evaluation(
        data_csv   = os.path.join("data", "all_plots.csv"),
        output_dir = outdir,
    )

    print_summary(lora_stats, schedule_df, eval_results)

    summary = {
        "config":        {"days": args.days, "plots": args.plots},
        "lora_stats":    lora_stats,
        "schedule_rows": len(schedule_df),
        "evaluation":    eval_results,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Full summary -> {outdir}/summary.json")


if __name__ == "__main__":
    main()
