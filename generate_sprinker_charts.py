"""
generate_sprinkler_charts.py
=============================
Generates all charts for the novel sprinkler simulation.
Run AFTER run_sprinkler_simulation.py

Charts:
  1. Tilt angle vs range + coverage per crop
  2. Water distribution field map (2D heatmap per plot)
  3. Water saved: smart vs fixed baseline
  4. Obstacle impact analysis
  5. Christiansen Uniformity Coefficient
  6. Coconut special chart
  7. End-to-end flow diagram
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, FancyArrowPatch
import json

sys.path.insert(0, os.path.dirname(__file__))
from sprinkler.crop_params     import CROP_PARAMS, PLOT_ASSIGNMENTS
from sprinkler.sprinkler_model import build_sprinkler_for_plot

os.makedirs("charts", exist_ok=True)

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       10,
    "axes.titlesize":  12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":      150,
})

COLORS = {
    "Coconut": "#8B4513", "Wheat": "#DAA520", "Potato": "#DEB887",
    "Tomato": "#FF6347",  "Chilli": "#FF4500",
    "smart":  "#2E86AB",  "fixed": "#E84855",
    "green":  "#3BB273",  "orange": "#F4A261",
}


# ══════════════════════════════════════════════════════════════════
# Load results
# ══════════════════════════════════════════════════════════════════
df_ev = pd.read_csv("results/sprinkler_events.csv")
df_sm = pd.read_csv("results/sprinkler_summary.csv")
with open("results/sprinkler_novel_summary.json") as f:
    summary = json.load(f)


# ══════════════════════════════════════════════════════════════════
# CHART 1 — Tilt Angle vs Range + Coverage (per crop)
# ══════════════════════════════════════════════════════════════════
def chart_tilt_analysis():
    print("  Generating tilt angle analysis chart...")

    # Rebuild sprinklers to get angle curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Tilt Angle Analysis — Range & Coverage per Crop\n"
                 "(Novel: Field-optimal angle differs from crop-standard angle)",
                 fontweight="bold")

    crops_to_show = ["Coconut", "Wheat", "Potato", "Tomato", "Chilli"]
    plot_ids      = ["plot_00", "plot_01", "plot_02", "plot_03", "plot_04"]

    for idx, (pid, crop) in enumerate(zip(plot_ids, crops_to_show)):
        ax  = axes[idx // 3][idx % 3]
        s   = build_sprinkler_for_plot(pid, crop, CROP_PARAMS[crop].get("soil","Red Soil") if idx>0 else "Red Soil")
        cp  = CROP_PARAMS[crop]

        angles   = list(range(10, 76, 2))
        ranges   = [s.water_range(a)           for a in angles]
        coverages= [s.coverage_efficiency(a)*100 for a in angles]

        crop_tilt  = cp["optimal_tilt_deg"]
        field_tilt = summary["tilt_summary"][pid]["field_optimal_deg"]

        color = COLORS.get(crop, "#333")
        ax2   = ax.twinx()

        ax.plot(angles, ranges,    color=color,       lw=2.5, label="Range (m)")
        ax2.plot(angles, coverages, color=COLORS["smart"], lw=2, ls="--", label="Coverage %")

        ax.axvline(crop_tilt,  color="orange", ls=":",  lw=2, label=f"Crop std: {crop_tilt}°")
        ax.axvline(field_tilt, color=COLORS["green"], ls="-", lw=2.5, label=f"Field opt: {field_tilt}°")

        ax.set_xlabel("Tilt Angle (degrees)")
        ax.set_ylabel("Water Range (m)", color=color)
        ax2.set_ylabel("Coverage (%)", color=COLORS["smart"])
        ax.set_title(f"{crop}", fontweight="bold")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    # Hide last subplot
    axes[1][2].axis("off")
    axes[1][2].text(0.5, 0.6,
        "Key Finding:\nField-optimal angle\ndiffers from crop-standard\ndue to obstacles +\nfield geometry",
        ha="center", va="center", transform=axes[1][2].transAxes,
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#D5F5E3", edgecolor=COLORS["green"], lw=2))

    plt.tight_layout()
    plt.savefig("charts/sprinkler_tilt_analysis.png", bbox_inches="tight")
    plt.close()
    print("  ✓ charts/sprinkler_tilt_analysis.png")


# ══════════════════════════════════════════════════════════════════
# CHART 2 — Field Map with Water Distribution + Obstacles
# ══════════════════════════════════════════════════════════════════
def chart_field_maps():
    print("  Generating field maps...")

    crops    = ["Coconut", "Wheat", "Tomato"]
    plot_ids = ["plot_00", "plot_01", "plot_03"]
    soils    = ["Red Soil", "Black Soil", "Red Soil"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Water Distribution Field Maps — Obstacle-Aware Sprinkler\n"
                 "(Dark blue = high water, white = dry/blocked zones)",
                 fontweight="bold")

    for ax, pid, crop, soil in zip(axes, plot_ids, crops, soils):
        s  = build_sprinkler_for_plot(pid, crop, soil)
        tr = summary["tilt_summary"][pid]
        tilt = tr["field_optimal_deg"]
        f  = s.field

        # Create grid
        res = 0.5
        xs  = np.arange(0, f.width  + res, res)
        ys  = np.arange(0, f.length + res, res)
        XX, YY = np.meshgrid(xs, ys)
        water_map = np.zeros_like(XX)

        R  = s.water_range(tilt)
        sx = f.sprinkler_x
        sy = f.sprinkler_y
        sigma = R * 0.15

        for rot in np.linspace(0, 360, 180, endpoint=False):
            blocked = any(obs.blocks_ray(sx, sy, rot, R) for obs in f.obstacles)
            if blocked:
                continue
            rad = np.radians(rot)
            lx  = sx + R * np.cos(rad)
            ly  = sy + R * np.sin(rad)
            dist2 = (XX - lx)**2 + (YY - ly)**2
            water_map += np.exp(-dist2 / (2 * sigma**2))

        if water_map.max() > 0:
            water_map /= water_map.max()

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "water", ["white", "#B3E5FC", "#0288D1", "#01579B"])
        im = ax.imshow(water_map, cmap=cmap, origin="lower",
                       extent=[0, f.width, 0, f.length],
                       vmin=0, vmax=1, aspect="equal")

        # Draw obstacles
        for obs in f.obstacles:
            color = "#8B4513" if obs.obstacle_type == "tree" else "#888888"
            circ  = Circle((obs.x, obs.y), obs.radius,
                            color=color, alpha=0.6, zorder=5)
            ax.add_patch(circ)
            if obs.obstacle_type == "tree":
                ax.text(obs.x, obs.y, "🌴" if crop == "Coconut" else "🌿",
                        ha="center", va="center", fontsize=7, zorder=6)

        # Sprinkler position
        ax.plot(sx, sy, "r^", ms=12, zorder=10, label="Sprinkler")
        ax.add_patch(Circle((sx, sy), R, fill=False,
                             color="red", ls="--", lw=1.5, alpha=0.5))

        plt.colorbar(im, ax=ax, shrink=0.7, label="Water intensity")
        ax.set_title(f"{crop} ({soil})\nTilt={tilt}°  Range={R:.1f}m  "
                     f"Coverage={tr['coverage_pct']}%  CU={tr['uniformity_cu']}%",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Width (m)")
        ax.set_ylabel("Length (m)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("charts/sprinkler_field_maps.png", bbox_inches="tight")
    plt.close()
    print("  ✓ charts/sprinkler_field_maps.png")


# ══════════════════════════════════════════════════════════════════
# CHART 3 — Water Saved: Smart vs Fixed
# ══════════════════════════════════════════════════════════════════
def chart_water_savings():
    print("  Generating water savings chart...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Water Savings — Adaptive Tilt vs Fixed 45° Sprinkler",
                 fontweight="bold")

    crops  = df_sm["crop"].tolist()
    smart  = df_sm["smart_volume_L"].tolist()
    fixed  = df_sm["fixed_volume_L"].tolist()
    saved  = df_sm["saved_L"].tolist()
    savedp = df_sm["saved_pct"].tolist()
    colors = [COLORS.get(c, "#666") for c in crops]

    x = np.arange(len(crops))
    w = 0.38

    # Smart vs Fixed volume
    axes[0].bar(x - w/2, smart, w, label="Smart (adaptive)", color=COLORS["smart"], alpha=0.85)
    axes[0].bar(x + w/2, fixed, w, label="Fixed 45°",        color=COLORS["fixed"], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{c}\n{pa['soil'][:8]}" for c, pa in
                               zip(crops, PLOT_ASSIGNMENTS)], fontsize=7)
    axes[0].set_ylabel("Total Water Used (L) over 30 days")
    axes[0].set_title("Smart vs Fixed Water Usage")
    axes[0].legend(fontsize=9)

    # Water saved per plot
    bar_colors = [COLORS["green"] if s > 0 else "#ccc" for s in saved]
    bars = axes[1].bar(x, saved, color=bar_colors, edgecolor="white")
    for bar, val, pct in zip(bars, saved, savedp):
        if val > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 5,
                         f"{pct:.1f}%", ha="center", fontsize=8, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(crops, fontsize=8)
    axes[1].set_ylabel("Water Saved (L)")
    axes[1].set_title("Water Saved per Plot (30 days)")

    # Pie chart total savings
    total_smart = df_sm["smart_volume_L"].sum()
    total_fixed = df_sm["fixed_volume_L"].sum()
    total_saved = max(0, total_fixed - total_smart)
    axes[2].pie(
        [total_smart, total_saved],
        labels=[f"Used\n{total_smart:,.0f}L", f"Saved\n{total_saved:,.0f}L"],
        colors=[COLORS["smart"], COLORS["green"]],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11},
    )
    axes[2].set_title(f"Total Water Budget\n(vs Fixed 45° baseline)", fontweight="bold")

    plt.tight_layout()
    plt.savefig("charts/sprinkler_water_savings.png", bbox_inches="tight")
    plt.close()
    print("  ✓ charts/sprinkler_water_savings.png")


# ══════════════════════════════════════════════════════════════════
# CHART 4 — Uniformity Coefficient + Coverage
# ══════════════════════════════════════════════════════════════════
def chart_uniformity():
    print("  Generating uniformity chart...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Irrigation Quality Metrics per Plot", fontweight="bold")

    crops    = df_sm["crop"].tolist()
    cu       = df_sm["uniformity_cu"].tolist()
    coverage = df_sm["coverage_pct"].tolist()
    colors   = [COLORS.get(c, "#666") for c in crops]
    x        = np.arange(len(crops))

    # CU chart
    bars = axes[0].bar(x, cu, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, cu):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axes[0].axhline(80, color="orange", ls="--", lw=2, label="Acceptable (80%)")
    axes[0].axhline(90, color="green",  ls="--", lw=2, label="Excellent (90%)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(crops, fontsize=9)
    axes[0].set_ylabel("Christiansen Uniformity Coefficient (%)")
    axes[0].set_title("Water Distribution Uniformity (CU)")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 110)

    # Coverage chart
    bars = axes[1].bar(x, coverage, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, coverage):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels(crops, fontsize=9)
    axes[1].set_ylabel("Field Coverage (%)")
    axes[1].set_title("Effective Field Coverage per Plot")
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig("charts/sprinkler_uniformity.png", bbox_inches="tight")
    plt.close()
    print("  ✓ charts/sprinkler_uniformity.png")


# ══════════════════════════════════════════════════════════════════
# CHART 5 — Coconut Special Chart
# ══════════════════════════════════════════════════════════════════
def chart_coconut():
    print("  Generating coconut chart...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Coconut — Novel Hyperparameters (Not in Existing Papers)",
                 fontweight="bold", color=COLORS["Coconut"])

    # Water comparison across all crops
    crop_names = list(CROP_PARAMS.keys())
    water_day  = [CROP_PARAMS[c]["water_per_day_L"] for c in crop_names]
    root_depth = [CROP_PARAMS[c]["root_depth_m"]    for c in crop_names]
    tilt_angle = [CROP_PARAMS[c]["optimal_tilt_deg"] for c in crop_names]
    bar_colors = [COLORS.get(c, "#666") for c in crop_names]

    bars = axes[0].bar(crop_names, water_day, color=bar_colors, edgecolor="white")
    for bar, val in zip(bars, water_day):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 1,
                     f"{val}L", ha="center", fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Water Needed (L/day/plant)")
    axes[0].set_title("Daily Water Requirement\n(Coconut = 29x more than Chilli)")
    axes[0].tick_params(axis="x", rotation=20)

    # Root depth comparison
    bars = axes[1].barh(crop_names, root_depth, color=bar_colors, edgecolor="white")
    for bar, val in zip(bars, root_depth):
        axes[1].text(val + 0.02, bar.get_y() + bar.get_height()/2,
                     f"{val}m", va="center", fontsize=9, fontweight="bold")
    axes[1].set_xlabel("Root Depth (m)")
    axes[1].set_title("Root Depth\n(Coconut roots go 2.5m deep)")
    axes[1].axvline(1.0, color="gray", ls="--", lw=1, alpha=0.5)

    # Tilt angle recommendation
    bars = axes[2].bar(crop_names, tilt_angle, color=bar_colors, edgecolor="white")
    for bar, val in zip(bars, tilt_angle):
        axes[2].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f"{val}°", ha="center", fontsize=9, fontweight="bold")
    axes[2].axhline(45, color="red", ls="--", lw=1.5, alpha=0.7, label="Fixed 45° baseline")
    axes[2].set_ylabel("Optimal Tilt Angle (degrees)")
    axes[2].set_title("Crop-Specific Optimal Tilt\n(Coconut=25° for under-canopy)")
    axes[2].legend(fontsize=9)
    axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig("charts/sprinkler_coconut.png", bbox_inches="tight")
    plt.close()
    print("  ✓ charts/sprinkler_coconut.png")


# ══════════════════════════════════════════════════════════════════
# CHART 6 — End-to-End Flow
# ══════════════════════════════════════════════════════════════════
def chart_end_to_end():
    print("  Generating end-to-end flow chart...")

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("End-to-End System Flow — Novel Adaptive Sprinkler Integration",
                 fontweight="bold", fontsize=13)

    boxes = [
        (1.0, "ESP32\nSensor\nReads soil\nmoisture",           "#D6EAF8", "#2E86AB"),
        (3.5, "LSTM\nPredicts\nmoisture drop\nin 2 hours",     "#D5F5E3", "#3BB273"),
        (6.0, "Fog\nScheduler\nMPC+RL decides\nirrigate now",  "#FEF9E7", "#F4A261"),
        (8.5, "Tilt Angle\nOptimiser\nFinds best angle\nfor field", "#F9EBEA", "#E84855"),
        (11.0,"Sprinkler\nFires\nObstacle-aware\nwater spread", "#EBD5F5", "#9B5DE5"),
        (13.5,"Soil\nUpdate\nMoisture rises\nto target",       "#D5F5E3", "#3BB273"),
    ]

    for i, (x, label, fc, ec) in enumerate(boxes):
        rect = mpatches.FancyBboxPatch((x-0.9, 1.2), 1.8, 2.4,
            boxstyle="round,pad=0.15", fc=fc, ec=ec, lw=2)
        ax.add_patch(rect)
        ax.text(x, 2.4, label, ha="center", va="center",
                fontsize=8, fontweight="bold")
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 1.05, 2.4), xytext=(x + 0.92, 2.4),
                arrowprops=dict(arrowstyle="->", color="#555", lw=2))

    # Data labels below arrows
    labels = ["30-min\nreadings", "Predicted\nmoisture", "Irrigation\ndecision",
              "Optimal\ntilt angle", "Water\ndistribution"]
    for i, (lbl, (x, *_)) in enumerate(zip(labels, boxes)):
        ax.text(x + 1.25, 1.0, lbl, ha="center", fontsize=7,
                color="#555", style="italic")

    # Novel badge
    ax.text(8.0, 4.6, "★ NOVEL CONTRIBUTION: Steps 4 & 5 are new — no existing fog paper has this",
            ha="center", fontsize=10, fontweight="bold", color="#E84855",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FDEDEC", ec="#E84855"))

    plt.tight_layout()
    plt.savefig("charts/sprinkler_end_to_end.png", bbox_inches="tight")
    plt.close()
    print("  ✓ charts/sprinkler_end_to_end.png")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nGenerating novel sprinkler charts...\n")
    chart_tilt_analysis()
    chart_field_maps()
    chart_water_savings()
    chart_uniformity()
    chart_coconut()
    chart_end_to_end()

    print("\n" + "=" * 55)
    print("  All charts saved to charts/")
    print("=" * 55)
    print("  sprinkler_tilt_analysis.png — Range & coverage per angle")
    print("  sprinkler_field_maps.png    — 2D water distribution maps")
    print("  sprinkler_water_savings.png — Smart vs fixed 45° savings")
    print("  sprinkler_uniformity.png    — Christiansen CU per plot")
    print("  sprinkler_coconut.png       — Coconut hyperparameters")
    print("  sprinkler_end_to_end.png    — Full system flow diagram")