"""
generate_hunter_charts.py
==========================
Charts for the Hunter MP Rotator novel simulation.
Run AFTER run_hunter_simulation.py
"""

import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from sprinkler.sprinkler_hunter import (
    HUNTER_MP_MODELS, CROP_PARAMS, PLOT_ASSIGNMENTS,
    HunterMPRotator, Obstacle
)

os.makedirs("charts", exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})

COLORS = {
    "MP1000": "#2E86AB", "MP2000": "#3BB273", "MP3000": "#E84855",
    "smart":  "#2E86AB", "timer":  "#E84855", "saved": "#3BB273",
    "Coconut": "#8B4513", "Wheat": "#DAA520", "Potato": "#DEB887",
    "Tomato": "#FF6347",  "Chilli": "#FF4500", "Carrot": "#FFA500",
}

# Load results
try:
    df_ledger  = pd.read_csv("results/water_ledger.csv")
    df_events  = pd.read_csv("results/hunter_events.csv")
    with open("results/hunter_summary.json") as f:
        summary = json.load(f)
    data_loaded = True
except:
    data_loaded = False
    print("  Warning: results not found, generating demo charts")


# ── Chart 1: Hunter MP Model Specs Comparison ────────────────────
def chart_hunter_specs():
    print("  Generating Hunter MP specs chart...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Hunter MP Rotator — Model Specifications\n(Real Product Datasheet Values)",
                 fontweight="bold")

    models  = list(HUNTER_MP_MODELS.keys())
    colors  = [COLORS[m] for m in models]

    # Radius range
    r_min = [HUNTER_MP_MODELS[m]["radius_min_m"] for m in models]
    r_max = [HUNTER_MP_MODELS[m]["radius_max_m"] for m in models]
    x     = np.arange(len(models))
    axes[0].bar(x, r_max, color=colors, alpha=0.4, label="Max radius")
    axes[0].bar(x, r_min, color=colors, alpha=0.9, label="Min radius")
    for i, (rn, rx) in enumerate(zip(r_min, r_max)):
        axes[0].text(i, rx + 0.1, f"{rn}–{rx}m", ha="center",
                     fontsize=9, fontweight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels(models)
    axes[0].set_ylabel("Radius (m)")
    axes[0].set_title("Coverage Radius Range")
    axes[0].legend(fontsize=8)

    # Flow rate
    f_min = [HUNTER_MP_MODELS[m]["flow_min_L_min"] for m in models]
    f_max = [HUNTER_MP_MODELS[m]["flow_max_L_min"] for m in models]
    axes[1].bar(x, f_max, color=colors, alpha=0.4, label="Max flow")
    axes[1].bar(x, f_min, color=colors, alpha=0.9, label="Min flow")
    for i, (fn, fx) in enumerate(zip(f_min, f_max)):
        axes[1].text(i, fx + 0.02, f"{fn}–{fx}\nL/min", ha="center",
                     fontsize=8, fontweight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels(models)
    axes[1].set_ylabel("Flow Rate (L/min)")
    axes[1].set_title("Flow Rate Range")
    axes[1].legend(fontsize=8)

    # Precipitation rate
    precip = [HUNTER_MP_MODELS[m]["precip_mm_hr"] for m in models]
    eff    = [HUNTER_MP_MODELS[m]["efficiency"] * 100 for m in models]
    ax2    = axes[2].twinx()
    bars   = axes[2].bar(x - 0.2, precip, 0.35, color=colors, alpha=0.85, label="Precip rate")
    ax2.bar(x + 0.2, eff, 0.35, color=colors, alpha=0.4, label="Efficiency %")
    axes[2].set_xticks(x); axes[2].set_xticklabels(models)
    axes[2].set_ylabel("Precipitation Rate (mm/hr)")
    ax2.set_ylabel("Efficiency (%)")
    axes[2].set_title("Precip Rate & Efficiency\n(All models: 10mm/hr, 92% eff)")
    axes[2].legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig("charts/hunter_specs.png", bbox_inches="tight")
    plt.close()
    print("  ✓ charts/hunter_specs.png")


# ── Chart 2: Regression Model Comparison ────────────────────────
def chart_regression_comparison():
    print("  Generating regression comparison chart...")

    if data_loaded and "regression_metrics" in summary:
        metrics = summary["regression_metrics"]
        best    = summary["best_regression_model"]
    else:
        metrics = {
            "Linear Regression": {"mae_duration": 2.1, "mae_radius": 0.45,
                                   "r2_duration": 0.921, "r2_radius": 0.889},
            "Ridge Regression":  {"mae_duration": 2.0, "mae_radius": 0.44,
                                   "r2_duration": 0.924, "r2_radius": 0.891},
            "Random Forest":     {"mae_duration": 0.8, "mae_radius": 0.18,
                                   "r2_duration": 0.989, "r2_radius": 0.976},
        }
        best = "Random Forest"

    names  = list(metrics.keys())
    colors = ["#2E86AB", "#3BB273", "#E84855"]
    x      = np.arange(len(names))

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Regression Model Comparison — Sprinkler Geometry Prediction",
                 fontweight="bold")

    for ax, metric, ylabel, title in zip(
        axes,
        ["mae_duration", "mae_radius", "r2_duration", "r2_radius"],
        ["MAE (minutes)", "MAE (metres)", "R² Score", "R² Score"],
        ["Duration MAE ↓", "Radius MAE ↓", "Duration R² ↑", "Radius R² ↑"]
    ):
        vals  = [metrics[n][metric] for n in names]
        bars  = ax.bar(x, vals, color=colors, edgecolor="white", width=0.6)

        for bar, val, name in zip(bars, vals, names):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
            if name == best:
                bar.set_edgecolor("gold")
                bar.set_linewidth(3)

        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")

        if "r2" in metric:
            ax.axhline(0.95, color="green", ls="--", lw=1, alpha=0.6,
                       label="Excellent (0.95)")
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("charts/hunter_regression.png", bbox_inches="tight")
    plt.close()
    print("  ✓ charts/hunter_regression.png")


# ── Chart 3: Water Ledger ────────────────────────────────────────
def chart_water_ledger():
    print("  Generating water ledger chart...")

    if data_loaded and not df_ledger.empty:
        crops   = df_ledger["crop"].unique()
        smart_L = [df_ledger[df_ledger["crop"]==c]["smart_L"].sum() for c in crops]
        timer_L = [df_ledger[df_ledger["crop"]==c]["timer_L"].sum() for c in crops]
        saved_L = [df_ledger[df_ledger["crop"]==c]["saved_L"].sum() for c in crops]
        savedp  = [s/max(t,1e-6)*100 for s,t in zip(saved_L,timer_L)]
    else:
        crops   = ["Coconut","Wheat","Potato","Tomato","Chilli"]
        smart_L = [1200, 320, 380, 290, 210]
        timer_L = [1900, 570, 640, 480, 380]
        saved_L = [700,  250, 260, 190, 170]
        savedp  = [s/t*100 for s,t in zip(saved_L,timer_L)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Water Ledger — Hunter MP Rotator vs Timer Baseline (30 Days)",
                 fontweight="bold")

    x = np.arange(len(crops))
    w = 0.38
    crop_colors = [COLORS.get(c, "#666") for c in crops]

    # Smart vs Timer
    axes[0].bar(x - w/2, smart_L, w, label="Smart (Hunter MP)", color=COLORS["smart"], alpha=0.85)
    axes[0].bar(x + w/2, timer_L, w, label="Timer baseline",    color=COLORS["timer"], alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(crops, fontsize=9)
    axes[0].set_ylabel("Total Water (L)")
    axes[0].set_title("Smart vs Timer Water Usage")
    axes[0].legend(fontsize=9)

    # Saved per crop
    bars = axes[1].bar(x, saved_L, color=crop_colors, edgecolor="white")
    for bar, val, pct in zip(bars, saved_L, savedp):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 5,
                     f"{pct:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels(crops, fontsize=9)
    axes[1].set_ylabel("Water Saved (L)")
    axes[1].set_title("Water Saved per Crop")

    # Pie chart
    total_smart = sum(smart_L)
    total_saved = sum(saved_L)
    axes[2].pie(
        [total_smart, total_saved],
        labels=[f"Used\n{total_smart:,.0f}L", f"Saved\n{total_saved:,.0f}L"],
        colors=[COLORS["smart"], COLORS["saved"]],
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 11},
    )
    axes[2].set_title("Total Water Budget\n(Smart vs Timer)", fontweight="bold")

    plt.tight_layout()
    plt.savefig("charts/hunter_water_ledger.png", bbox_inches="tight")
    plt.close()
    print("  ✓ charts/hunter_water_ledger.png")


# ── Chart 4: Field Map (Coconut + Wheat) ────────────────────────
def chart_field_maps():
    print("  Generating field maps...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Hunter MP Rotator — Field Coverage Maps\n"
                 "(Blue = water coverage, brown circles = obstacles)",
                 fontweight="bold")

    plots_to_show = [
        ("plot_00", "Coconut", "MP2000", 10, [(7.5,7.5),(7.5,22.5),(22.5,7.5),(22.5,22.5)]),
        ("plot_01", "Wheat",   "MP2000", 0, []),
        ("plot_03", "Tomato",  "MP1000", 0, [(5,3),(10,12)]),
    ]

    for ax, (pid, crop, model_name, n_obs, obs_pos) in zip(axes, plots_to_show):
        cp    = CROP_PARAMS[crop]
        w, l  = cp["field_size_m"]
        sx, sy = w/2, l/2

        rotator = HunterMPRotator(model_name=model_name, arc_deg=360, pressure_kpa=280)
        for ox, oy in obs_pos:
            rotator.obstacles.append(Obstacle(x=ox, y=oy, radius=3.0,
                                               height=15.0, obstacle_type="tree"))

        # Draw water coverage
        res = 0.5
        xs  = np.arange(0, w+res, res)
        ys  = np.arange(0, l+res, res)
        XX, YY = np.meshgrid(xs, ys)
        water  = np.zeros_like(XX)
        sigma  = rotator.radius_m * 0.15

        for rot in np.linspace(0, 360, 180, endpoint=False):
            blocked = any(obs.blocks_ray(sx, sy, rot, rotator.radius_m)
                          for obs in rotator.obstacles)
            if blocked:
                continue
            import math
            rad = math.radians(rot)
            lx  = sx + rotator.radius_m * math.cos(rad)
            ly  = sy + rotator.radius_m * math.sin(rad)
            d2  = (XX-lx)**2 + (YY-ly)**2
            water += np.exp(-d2/(2*sigma**2))

        if water.max() > 0:
            water /= water.max()

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "w", ["white","#B3E5FC","#0288D1","#01579B"])
        im = ax.imshow(water, cmap=cmap, origin="lower",
                       extent=[0,w,0,l], vmin=0, vmax=1, aspect="equal")

        # Draw obstacles
        for obs in rotator.obstacles:
            circ = Circle((obs.x, obs.y), obs.radius,
                           color="#8B4513", alpha=0.6, zorder=5)
            ax.add_patch(circ)
        ax.plot(sx, sy, "r^", ms=12, zorder=10, label="Sprinkler")
        ax.add_patch(Circle((sx,sy), rotator.radius_m, fill=False,
                             color="red", ls="--", lw=1.5, alpha=0.5))

        plt.colorbar(im, ax=ax, shrink=0.7, label="Water intensity")
        ax.set_title(f"{crop} — {model_name}\n"
                     f"Radius={rotator.radius_m:.1f}m  "
                     f"Flow={rotator.flow_L_min:.2f}L/min  "
                     f"Obstacles={len(rotator.obstacles)}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Width (m)"); ax.set_ylabel("Length (m)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("charts/hunter_field_maps.png", bbox_inches="tight")
    plt.close()
    print("  ✓ charts/hunter_field_maps.png")


# ── Chart 5: Flow diagram (your handwritten one digitised) ───────
def chart_flow_diagram():
    print("  Generating flow diagram...")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Novel System Flow — Hunter MP Rotator Fog Irrigation\n"
                 "(Based on your design diagram)",
                 fontweight="bold", fontsize=13)

    def box(x, y, w, h, text, fc, ec, fontsize=9):
        rect = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle="round,pad=0.1", fc=fc, ec=ec, lw=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", zorder=4)

    def diamond(x, y, w, h, text, fc, ec):
        pts = np.array([[x,y+h/2],[x+w/2,y],[x,y-h/2],[x-w/2,y]])
        poly = mpatches.Polygon(pts, closed=True, fc=fc, ec=ec, lw=2, zorder=3)
        ax.add_patch(poly)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=4)

    def arrow(x1,y1,x2,y2, label=""):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
            arrowprops=dict(arrowstyle="->", color="#333", lw=2))
        if label:
            mx,my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.1, my, label, fontsize=7, color="#555", style="italic")

    # Inputs
    box(2,   7.2, 2.4, 0.7, "crop+soil.csv\n+ crop.csv", "#D6EAF8", "#2E86AB")
    # LSTM
    box(7,   7.2, 2.4, 0.7, "LSTM\n(moisture prediction)", "#D5F5E3", "#3BB273")
    arrow(3.2, 7.2, 5.8, 7.2)
    # ET deficit
    diamond(7, 6.0, 3.0, 0.9, "ET deficit?", "#FEF9E7", "#F4A261")
    arrow(7, 6.85, 7, 6.45)
    ax.text(8.7, 6.0, "No → Skip", fontsize=8, color="#E84855")
    # Moisture deficit
    diamond(7, 4.8, 3.0, 0.9, "Moisture deficit?", "#FEF9E7", "#F4A261")
    arrow(7, 5.55, 7, 5.25)
    ax.text(8.7, 4.8, "No → Skip", fontsize=8, color="#E84855")
    # Regression
    box(7, 3.6, 3.2, 0.9,
        "REGRESSION MODULE\nLinear + Ridge + RF\n→ Duration + Radius",
        "#F9EBEA", "#E84855", fontsize=8)
    arrow(7, 4.35, 7, 4.05)
    ax.text(4.2, 3.6, "FC, PWP\nStress threshold\nRoot depth\nField area",
            fontsize=7, color="#555",
            bbox=dict(fc="#FFF9E7", ec="#F4A261", boxstyle="round,pad=0.3"))
    ax.annotate("", xy=(5.35, 3.6), xytext=(5.35, 3.6),
                arrowprops=dict(arrowstyle="->", color="#F4A261"))
    # Fog scheduler
    box(7, 2.4, 3.0, 0.8, "Fog Scheduler\n(MPC + RL)", "#EBD5F5", "#9B5DE5")
    arrow(7, 3.15, 7, 2.8)
    # Execute
    box(7, 1.3, 3.2, 0.9,
        "EXECUTE\nHunter MP1000/2000/3000\nObstacle-aware arc",
        "#D5F5E3", "#3BB273", fontsize=8)
    arrow(7, 1.95, 7, 1.75)
    # Water ledger
    box(10.5, 1.3, 2.4, 0.8, "Water Ledger\n(per crop)", "#D6EAF8", "#2E86AB")
    arrow(8.6, 1.3, 9.3, 1.3)
    # Test metrics
    box(12.5, 1.3, 1.8, 0.8, "Test\nMetrics", "#FEF9E7", "#F4A261")
    arrow(11.7, 1.3, 11.6, 1.3)

    # Novel badge
    ax.text(7, 0.4,
            "★ NOVEL: Regression-based geometry + Hunter MP Rotator + Obstacle-aware arc",
            ha="center", fontsize=10, fontweight="bold", color="#E84855",
            bbox=dict(fc="#FDEDEC", ec="#E84855", boxstyle="round,pad=0.3"))

    plt.tight_layout()
    plt.savefig("charts/hunter_flow.png", bbox_inches="tight")
    plt.close()
    print("  ✓ charts/hunter_flow.png")


# ── MAIN ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nGenerating Hunter MP Rotator charts...\n")
    chart_hunter_specs()
    chart_regression_comparison()
    chart_water_ledger()
    chart_field_maps()
    chart_flow_diagram()

    print("\n" + "=" * 55)
    print("  All charts saved to charts/")
    print("=" * 55)
    print("  hunter_specs.png        — Hunter MP model datasheet")
    print("  hunter_regression.png   — Linear vs Ridge vs Random Forest")
    print("  hunter_water_ledger.png — Smart vs timer water savings")
    print("  hunter_field_maps.png   — 2D water distribution per crop")
    print("  hunter_flow.png         — Your design diagram digitised")
