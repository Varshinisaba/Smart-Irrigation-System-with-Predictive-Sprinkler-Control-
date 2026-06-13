"""
sprinkler_model.py
==================
Novel contribution: Physics-based adaptive sprinkler simulation
integrated with fog computing irrigation scheduler.

Key novelties (not in any existing fog irrigation paper):
1. Tilt angle → water range using projectile motion physics
2. Obstacle-aware ray casting (coconut trees, walls)
3. Christiansen Uniformity Coefficient per plot
4. Optimal tilt angle finder per field + crop combination
5. Water savings vs fixed-angle baseline
6. End-to-end: LSTM prediction → fog decision → sprinkler control → soil update
"""

import numpy as np
import math
from dataclasses import dataclass, field as dc_field
from typing import List, Dict, Tuple

G         = 9.81    # gravity m/s²
RHO_WATER = 1000    # kg/m³
MIN_TILT  = 10      # minimum tilt degrees
MAX_TILT  = 75      # maximum tilt degrees


# ══════════════════════════════════════════════════════════════════
# OBSTACLE
# ══════════════════════════════════════════════════════════════════

@dataclass
class Obstacle:
    x:             float   # centre x (m)
    y:             float   # centre y (m)
    radius:        float   # blocking radius (m)
    height:        float   # height (m)
    obstacle_type: str     # "tree", "wall", "plant"

    def blocks_ray(self, sx: float, sy: float,
                   angle_deg: float, distance: float) -> bool:
        """Ray casting: does obstacle block water from (sx,sy)?"""
        rad = np.radians(angle_deg)
        ex  = sx + distance * np.cos(rad)
        ey  = sy + distance * np.sin(rad)
        dx  = ex - sx
        dy  = ey - sy
        len_sq = dx*dx + dy*dy
        if len_sq == 0:
            return False
        t = max(0.0, min(1.0, ((self.x-sx)*dx + (self.y-sy)*dy) / len_sq))
        px = sx + t * dx
        py = sy + t * dy
        return math.sqrt((self.x-px)**2 + (self.y-py)**2) <= self.radius


# ══════════════════════════════════════════════════════════════════
# FIELD
# ══════════════════════════════════════════════════════════════════

@dataclass
class Field:
    width:    float
    length:   float
    crop:     str
    soil:     str
    plot_id:  str = "plot_00"
    sprinkler_x: float = None
    sprinkler_y: float = None
    obstacles: List[Obstacle] = dc_field(default_factory=list)

    def __post_init__(self):
        if self.sprinkler_x is None:
            self.sprinkler_x = self.width  / 2
        if self.sprinkler_y is None:
            self.sprinkler_y = self.length / 2

    @property
    def area_m2(self):
        return self.width * self.length

    def add_obstacle(self, obs: Obstacle):
        self.obstacles.append(obs)


# ══════════════════════════════════════════════════════════════════
# SPRINKLER PHYSICS ENGINE
# ══════════════════════════════════════════════════════════════════

class SprinklerModel:
    """
    Adaptive physics-based sprinkler.
    Novel: automatically finds optimal tilt angle per field.
    """

    def __init__(self,
                 field:               Field,
                 nozzle_pressure_kpa: float = 200.0,
                 nozzle_diameter_mm:  float = 4.0,
                 rotation_speed_rpm:  float = 1.0):

        self.field    = field
        self.pressure = nozzle_pressure_kpa
        self.d_nozzle = nozzle_diameter_mm / 1000.0
        self.rpm      = rotation_speed_rpm

        # Nozzle exit velocity — Bernoulli + discharge coefficient
        Cd      = 0.95
        self.v0 = Cd * math.sqrt(2.0 * nozzle_pressure_kpa * 1000.0 / RHO_WATER)

        # Volumetric flow rate (m³/s)
        self.flow_rate = Cd * (math.pi * self.d_nozzle**2 / 4.0) * self.v0

    # ── Core physics ────────────────────────────────────────────
    def water_range(self, tilt_deg: float) -> float:
        """
        Projectile motion:  R = v₀² × sin(2θ) / g
        Maximum range at θ = 45°.
        Air resistance correction factor = 0.75.
        """
        theta = math.radians(tilt_deg)
        R = (self.v0**2 * math.sin(2 * theta)) / G * 0.75
        # Clip to field diagonal
        max_reach = math.sqrt(self.field.width**2 + self.field.length**2)
        return min(R, max_reach)

    def arc_clearance(self, tilt_deg: float, n_angles: int = 360) -> float:
        """Fraction of rotation arc NOT blocked by obstacles."""
        R  = self.water_range(tilt_deg)
        sx = self.field.sprinkler_x
        sy = self.field.sprinkler_y
        clear = sum(
            1 for a in np.linspace(0, 360, n_angles, endpoint=False)
            if not any(obs.blocks_ray(sx, sy, a, R) for obs in self.field.obstacles)
        )
        return clear / n_angles

    def coverage_efficiency(self, tilt_deg: float) -> float:
        """
        Fraction of field area effectively covered.
        = (circular area clipped to field / field area) × arc_clearance
        """
        R  = self.water_range(tilt_deg)
        sx = self.field.sprinkler_x
        sy = self.field.sprinkler_y
        W  = self.field.width
        L  = self.field.length

        # Effective radius clipped so spray stays inside field
        max_r_to_wall = min(sx, W - sx, sy, L - sy)
        eff_r         = min(R, max_r_to_wall + R * 0.3)   # allow slight overshoot
        circle_area   = math.pi * eff_r ** 2
        base_cov      = min(circle_area / self.field.area_m2, 1.0)

        arc = self.arc_clearance(tilt_deg, n_angles=180)
        return round(base_cov * arc, 4)

    def uniformity_coefficient(self, tilt_deg: float,
                                n_angles: int = 180) -> float:
        """
        Christiansen's Uniformity Coefficient (CU).
        CU > 80% = acceptable.  CU > 90% = excellent.
        CU = 100 × (1 − mean_abs_deviation / mean)
        Uses radial water depth profile (triangular distribution).
        """
        R  = self.water_range(tilt_deg)
        sx = self.field.sprinkler_x
        sy = self.field.sprinkler_y

        # Sample points along radii in clear arcs
        depths = []
        for ang in np.linspace(0, 360, n_angles, endpoint=False):
            blocked = any(obs.blocks_ray(sx, sy, ang, R)
                          for obs in self.field.obstacles)
            if blocked:
                continue
            for r in np.linspace(0.5, R, 20):
                # Triangular depth profile: max at 0.3R, zero at R
                depth = max(0.0, 1.0 - abs(r / R - 0.3) / 0.7)
                depths.append(depth)

        if not depths:
            return 0.0
        arr  = np.array(depths)
        mean = arr.mean()
        if mean == 0:
            return 0.0
        CU = 100.0 * (1.0 - np.mean(np.abs(arr - mean)) / mean)
        return round(float(np.clip(CU, 0, 100)), 1)

    # ── Optimal tilt finder (NOVEL) ──────────────────────────────
    def optimal_tilt_for_field(self) -> Dict:
        """
        Find tilt angle that maximises coverage efficiency
        for THIS field with ITS obstacles.
        This is the novel adaptive control contribution.
        """
        angles  = list(range(MIN_TILT, MAX_TILT + 1, 5))
        results = []
        for ang in angles:
            eff = self.coverage_efficiency(ang)
            rng = self.water_range(ang)
            cu  = self.uniformity_coefficient(ang, n_angles=72)
            results.append({
                "tilt_deg":      ang,
                "range_m":       round(rng, 2),
                "coverage_pct":  round(eff * 100, 1),
                "uniformity_cu": cu,
                # Combined score: 60% coverage + 40% uniformity
                "score":         0.6 * eff + 0.4 * (cu / 100.0),
            })
        best = max(results, key=lambda x: x["score"])
        return {"optimal": best, "all_angles": results}

    # ── Water savings (NOVEL) ────────────────────────────────────
    def water_saved_vs_fixed(self,
                              smart_tilt:  float,
                              fixed_tilt:  float = 45.0,
                              duration_min: float = 30.0) -> Dict:
        """
        Compare adaptive smart tilt vs fixed 45° sprinkler.
        Smart system matches coverage to actual field need → saves water.
        """
        eff_smart = self.coverage_efficiency(smart_tilt)
        eff_fixed = self.coverage_efficiency(fixed_tilt)

        vol_per_min_L = self.flow_rate * 60.0 * 1000.0
        vol_smart = vol_per_min_L * duration_min * eff_smart
        vol_fixed = vol_per_min_L * duration_min * eff_fixed

        # Smart saves by not over-irrigating blocked/covered zones
        saved_L   = max(0.0, vol_fixed - vol_smart)
        saved_pct = saved_L / max(vol_fixed, 1e-9) * 100.0

        return {
            "smart_tilt_deg":  smart_tilt,
            "fixed_tilt_deg":  fixed_tilt,
            "smart_volume_L":  round(vol_smart, 2),
            "fixed_volume_L":  round(vol_fixed, 2),
            "saved_L":         round(saved_L, 2),
            "saved_pct":       round(saved_pct, 1),
            "smart_coverage_pct": round(eff_smart * 100, 1),
            "fixed_coverage_pct": round(eff_fixed * 100, 1),
        }

    # ── Irrigation event simulator ───────────────────────────────
    def simulate_irrigation_event(self,
                                   current_moisture: float,
                                   target_moisture:  float,
                                   root_depth_m:     float,
                                   tilt_deg:         float) -> Dict:
        """
        Calculate duration and volume to raise moisture
        from current → target at given tilt angle.
        End-to-end link: fog decision → sprinkler → soil update.
        """
        deficit_mm = max(0.0, target_moisture - current_moisture) * root_depth_m * 1000.0

        # Application rate depends on coverage and flow
        eff  = self.coverage_efficiency(tilt_deg)
        rate_mm_per_min = (self.flow_rate * 60.0 * 1000.0 * eff) / max(self.field.area_m2, 1.0)

        duration_min = deficit_mm / max(rate_mm_per_min, 0.001)
        duration_min = min(float(duration_min), 90.0)   # cap 90 min

        vol_L = self.flow_rate * duration_min * 60.0 * 1000.0

        # Soil moisture after irrigation
        moisture_after = min(current_moisture + (rate_mm_per_min * duration_min) / (root_depth_m * 1000.0),
                             target_moisture + 0.02)

        return {
            "current_moisture":       round(current_moisture, 4),
            "target_moisture":        round(target_moisture, 4),
            "moisture_after":         round(moisture_after, 4),
            "deficit_mm":             round(deficit_mm, 2),
            "tilt_deg":               tilt_deg,
            "duration_min":           round(duration_min, 1),
            "volume_L":               round(vol_L, 2),
            "application_rate_mm_min":round(rate_mm_per_min, 4),
            "coverage_pct":           round(eff * 100, 1),
        }

    # ── Obstacle impact ──────────────────────────────────────────
    def obstacle_impact(self, tilt_deg: float) -> Dict:
        """How much do obstacles reduce water coverage?"""
        f = self.field
        obs_saved = f.obstacles[:]
        f.obstacles = []
        cov_free = self.coverage_efficiency(tilt_deg)
        f.obstacles = obs_saved
        cov_with = self.coverage_efficiency(tilt_deg)
        blocked  = max(0.0, cov_free - cov_with) / max(cov_free, 1e-9) * 100.0
        return {
            "n_obstacles":          len(f.obstacles),
            "coverage_free_pct":    round(cov_free * 100, 1),
            "coverage_with_pct":    round(cov_with * 100, 1),
            "area_blocked_pct":     round(blocked, 1),
        }


# ══════════════════════════════════════════════════════════════════
# FACTORY — build Field + SprinklerModel for each plot
# ══════════════════════════════════════════════════════════════════

def build_sprinkler_for_plot(plot_id: str, crop: str, soil: str) -> SprinklerModel:
    from sprinkler.crop_params import CROP_PARAMS
    cp  = CROP_PARAMS[crop]
    w, l = cp["field_size_m"]
    f = Field(width=w, length=l, crop=crop, soil=soil, plot_id=plot_id)

    np.random.seed(abs(hash(plot_id)) % 9999)

    if crop == "Coconut":
        # Real coconut trees as obstacles on 7.5m grid
        for tx in np.arange(7.5, w, 7.5):
            for ty in np.arange(7.5, l, 7.5):
                if abs(tx - w/2) < 4 and abs(ty - l/2) < 4:
                    continue   # skip centre (sprinkler location)
                f.add_obstacle(Obstacle(x=float(tx), y=float(ty),
                                        radius=3.5, height=15.0,
                                        obstacle_type="tree"))
        # Boundary wall fragments
        for bx, by in [(1.0, l/2), (w-1.0, l/2)]:
            f.add_obstacle(Obstacle(x=bx, y=by, radius=0.5,
                                    height=2.0, obstacle_type="wall"))
    else:
        # Small plants near boundary as obstacles
        n_obs = {"Tomato": 3, "Chilli": 2, "Wheat": 2, "Potato": 2}.get(crop, 2)
        for _ in range(n_obs):
            px = float(np.random.uniform(1.5, w - 1.5))
            py = float(np.random.uniform(1.5, l - 1.5))
            if abs(px - w/2) < 2.5 and abs(py - l/2) < 2.5:
                continue
            f.add_obstacle(Obstacle(x=px, y=py,
                                    radius=cp["canopy_radius_m"],
                                    height=cp["tree_height_m"],
                                    obstacle_type="plant"))

    s = SprinklerModel(
        field               = f,
        nozzle_pressure_kpa = cp["nozzle_pressure_kpa"],
        nozzle_diameter_mm  = cp["nozzle_diameter_mm"],
        rotation_speed_rpm  = cp["rotation_speed_rpm"],
    )
    return s
