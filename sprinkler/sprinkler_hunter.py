"""
sprinkler_hunter.py
====================
Hunter MP Rotator based sprinkler simulation.
Replaces old sprinkler_model.py completely.

Novel contributions:
1. Real Hunter MP Rotator specs (MP1000 / MP2000 / MP3000)
2. Regression-based geometry module (Linear, Ridge, Random Forest)
3. Evapotranspiration deficit → moisture deficit → sprinkler selection
4. Obstacle-aware coverage (coconut trees, walls)
5. Water ledger tracking per plot
6. Fog scheduler integration

Flow (matches your diagram):
  crop+soil CSV + crop CSV
        ↓
  LSTM (moisture prediction)
        ↓
  ET deficit check
        ↓
  Moisture deficit check
        ↓
  Regression → sprinkler geometry (duration + coverage)
        ↓
  Fog scheduler
        ↓
  Execute → Water ledger → Test metrics
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# HUNTER MP ROTATOR SPECS (real product datasheet values)
# ══════════════════════════════════════════════════════════════════

HUNTER_MP_MODELS = {
    "MP1000": {
        "radius_min_m":    2.4,
        "radius_max_m":    3.7,
        "flow_min_L_min":  0.10,
        "flow_max_L_min":  0.38,
        "precip_mm_hr":    10.0,
        "pressure_min_kpa":170,
        "pressure_max_kpa":410,
        "arc_options_deg": [90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
        "efficiency":      0.92,
        "best_for":        "small plots 2.4-3.7m radius",
        "color":           "#2E86AB",
    },
    "MP2000": {
        "radius_min_m":    3.7,
        "radius_max_m":    6.1,
        "flow_min_L_min":  0.38,
        "flow_max_L_min":  0.95,
        "precip_mm_hr":    10.0,
        "pressure_min_kpa":170,
        "pressure_max_kpa":410,
        "arc_options_deg": [90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
        "efficiency":      0.92,
        "best_for":        "medium plots 3.7-6.1m radius",
        "color":           "#3BB273",
    },
    "MP3000": {
        "radius_min_m":    6.1,
        "radius_max_m":    10.7,
        "flow_min_L_min":  0.95,
        "flow_max_L_min":  2.65,
        "precip_mm_hr":    10.0,
        "pressure_min_kpa":205,
        "pressure_max_kpa":410,
        "arc_options_deg": [90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
        "efficiency":      0.92,
        "best_for":        "large plots 6.1-10.7m radius",
        "color":           "#E84855",
    },
}

# ══════════════════════════════════════════════════════════════════
# CROP PARAMS — all Tamil Nadu crops including Coconut
# ══════════════════════════════════════════════════════════════════

CROP_PARAMS = {
    "Coconut": {
        "fc": 0.45, "pwp": 0.18, "optimal_low": 0.28, "optimal_high": 0.40,
        "root_depth_m": 2.5, "kc": 1.10, "water_per_day_L": 175,
        "field_size_m": (30, 30), "canopy_radius_m": 3.5,
        "recommended_model": "MP2000", "arc_deg": 360,
        "stress_threshold": 0.22,
        "growth_stages": {
            "seedling": {"fc": 0.38, "pwp": 0.15},
            "young":    {"fc": 0.40, "pwp": 0.16},
            "mature":   {"fc": 0.45, "pwp": 0.18},
            "bearing":  {"fc": 0.45, "pwp": 0.18},
        },
    },
    "Wheat": {
        "fc": 0.34, "pwp": 0.14, "optimal_low": 0.22, "optimal_high": 0.29,
        "root_depth_m": 1.0, "kc": 0.85, "water_per_day_L": 6,
        "field_size_m": (20, 20), "canopy_radius_m": 0.2,
        "recommended_model": "MP2000", "arc_deg": 360,
        "stress_threshold": 0.18,
        "growth_stages": {
            "germination": {"fc": 0.30, "pwp": 0.12},
            "tillering":   {"fc": 0.34, "pwp": 0.14},
            "heading":     {"fc": 0.34, "pwp": 0.14},
            "maturity":    {"fc": 0.30, "pwp": 0.12},
        },
    },
    "Potato": {
        "fc": 0.38, "pwp": 0.16, "optimal_low": 0.25, "optimal_high": 0.32,
        "root_depth_m": 0.6, "kc": 1.10, "water_per_day_L": 8,
        "field_size_m": (20, 20), "canopy_radius_m": 0.3,
        "recommended_model": "MP2000", "arc_deg": 360,
        "stress_threshold": 0.20,
        "growth_stages": {
            "emergence":  {"fc": 0.35, "pwp": 0.14},
            "vegetative": {"fc": 0.38, "pwp": 0.16},
            "tuber":      {"fc": 0.38, "pwp": 0.16},
            "maturity":   {"fc": 0.32, "pwp": 0.14},
        },
    },
    "Tomato": {
        "fc": 0.35, "pwp": 0.15, "optimal_low": 0.23, "optimal_high": 0.30,
        "root_depth_m": 0.8, "kc": 1.05, "water_per_day_L": 7,
        "field_size_m": (15, 15), "canopy_radius_m": 0.4,
        "recommended_model": "MP1000", "arc_deg": 360,
        "stress_threshold": 0.19,
        "growth_stages": {
            "seedling":   {"fc": 0.32, "pwp": 0.13},
            "vegetative": {"fc": 0.35, "pwp": 0.15},
            "flowering":  {"fc": 0.35, "pwp": 0.15},
            "fruiting":   {"fc": 0.35, "pwp": 0.15},
        },
    },
    "Chilli": {
        "fc": 0.33, "pwp": 0.14, "optimal_low": 0.21, "optimal_high": 0.28,
        "root_depth_m": 0.5, "kc": 0.90, "water_per_day_L": 5,
        "field_size_m": (15, 15), "canopy_radius_m": 0.25,
        "recommended_model": "MP1000", "arc_deg": 360,
        "stress_threshold": 0.17,
        "growth_stages": {
            "seedling":   {"fc": 0.30, "pwp": 0.12},
            "vegetative": {"fc": 0.33, "pwp": 0.14},
            "flowering":  {"fc": 0.33, "pwp": 0.14},
            "fruiting":   {"fc": 0.30, "pwp": 0.13},
        },
    },
    "Carrot": {
        "fc": 0.36, "pwp": 0.15, "optimal_low": 0.23, "optimal_high": 0.30,
        "root_depth_m": 0.4, "kc": 0.90, "water_per_day_L": 5,
        "field_size_m": (15, 15), "canopy_radius_m": 0.2,
        "recommended_model": "MP1000", "arc_deg": 360,
        "stress_threshold": 0.18,
        "growth_stages": {
            "seedling":   {"fc": 0.33, "pwp": 0.13},
            "vegetative": {"fc": 0.36, "pwp": 0.15},
            "maturity":   {"fc": 0.34, "pwp": 0.14},
        },
    },
    # TN crops
    "paddy":     {"fc": 0.40, "pwp": 0.20, "optimal_low": 0.30, "optimal_high": 0.38,
                  "root_depth_m": 0.4, "kc": 1.20, "water_per_day_L": 15,
                  "field_size_m": (25, 25), "canopy_radius_m": 0.15,
                  "recommended_model": "MP3000", "arc_deg": 360,
                  "stress_threshold": 0.25, "growth_stages": {}},
    "sugarcane": {"fc": 0.38, "pwp": 0.18, "optimal_low": 0.26, "optimal_high": 0.34,
                  "root_depth_m": 1.5, "kc": 1.10, "water_per_day_L": 20,
                  "field_size_m": (25, 25), "canopy_radius_m": 0.5,
                  "recommended_model": "MP3000", "arc_deg": 360,
                  "stress_threshold": 0.22, "growth_stages": {}},
    "groundnut": {"fc": 0.30, "pwp": 0.14, "optimal_low": 0.20, "optimal_high": 0.26,
                  "root_depth_m": 0.6, "kc": 0.80, "water_per_day_L": 6,
                  "field_size_m": (20, 20), "canopy_radius_m": 0.3,
                  "recommended_model": "MP2000", "arc_deg": 360,
                  "stress_threshold": 0.17, "growth_stages": {}},
    "cotton":    {"fc": 0.32, "pwp": 0.15, "optimal_low": 0.21, "optimal_high": 0.28,
                  "root_depth_m": 1.0, "kc": 0.90, "water_per_day_L": 8,
                  "field_size_m": (20, 20), "canopy_radius_m": 0.4,
                  "recommended_model": "MP2000", "arc_deg": 360,
                  "stress_threshold": 0.18, "growth_stages": {}},
}

PLOT_ASSIGNMENTS = [
    {"plot": "plot_00", "crop": "Coconut", "soil": "Red Soil",      "stage": "bearing"},
    {"plot": "plot_01", "crop": "Wheat",   "soil": "Black Soil",    "stage": "tillering"},
    {"plot": "plot_02", "crop": "Potato",  "soil": "Alluvial Soil", "stage": "tuber"},
    {"plot": "plot_03", "crop": "Tomato",  "soil": "Red Soil",      "stage": "fruiting"},
    {"plot": "plot_04", "crop": "Chilli",  "soil": "Clay Soil",     "stage": "flowering"},
    {"plot": "plot_05", "crop": "Wheat",   "soil": "Loam Soil",     "stage": "heading"},
    {"plot": "plot_06", "crop": "Potato",  "soil": "Chalky Soil",   "stage": "vegetative"},
    {"plot": "plot_07", "crop": "Tomato",  "soil": "Sandy Soil",    "stage": "flowering"},
    {"plot": "plot_08", "crop": "Chilli",  "soil": "Black Soil",    "stage": "fruiting"},
]


# ══════════════════════════════════════════════════════════════════
# OBSTACLE
# ══════════════════════════════════════════════════════════════════

@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    height: float
    obstacle_type: str

    def blocks_ray(self, sx, sy, angle_deg, distance) -> bool:
        import math
        rad = math.radians(angle_deg)
        ex, ey = sx + distance * math.cos(rad), sy + distance * math.sin(rad)
        dx, dy = ex - sx, ey - sy
        len_sq = dx*dx + dy*dy
        if len_sq == 0:
            return False
        t = max(0.0, min(1.0, ((self.x-sx)*dx + (self.y-sy)*dy) / len_sq))
        px, py = sx + t*dx, sy + t*dy
        return math.sqrt((self.x-px)**2 + (self.y-py)**2) <= self.radius


# ══════════════════════════════════════════════════════════════════
# HUNTER MP ROTATOR MODEL
# ══════════════════════════════════════════════════════════════════

class HunterMPRotator:
    """
    Physics + datasheet based Hunter MP Rotator simulation.
    Selects correct model (MP1000/2000/3000) based on plot size.
    """

    def __init__(self, model_name: str, arc_deg: float = 360,
                 pressure_kpa: float = 280.0):
        assert model_name in HUNTER_MP_MODELS, f"Unknown model: {model_name}"
        self.model_name  = model_name
        self.specs       = HUNTER_MP_MODELS[model_name]
        self.arc_deg     = arc_deg
        self.pressure    = pressure_kpa
        self.obstacles: List[Obstacle] = []

        # Interpolate flow and radius based on pressure
        p_range = self.specs["pressure_max_kpa"] - self.specs["pressure_min_kpa"]
        p_frac  = np.clip((pressure_kpa - self.specs["pressure_min_kpa"]) / max(p_range,1), 0, 1)

        self.radius_m   = self.specs["radius_min_m"] + p_frac * (
            self.specs["radius_max_m"] - self.specs["radius_min_m"])
        self.flow_L_min = self.specs["flow_min_L_min"] + p_frac * (
            self.specs["flow_max_L_min"] - self.specs["flow_min_L_min"])

        self.precip_rate_mm_hr = self.specs["precip_mm_hr"]
        self.efficiency        = self.specs["efficiency"]

    def arc_clearance(self, sx: float, sy: float, n_angles: int = 360) -> float:
        """Fraction of arc not blocked by obstacles."""
        arc_start = 0
        arc_end   = self.arc_deg
        angles    = np.linspace(arc_start, arc_end, n_angles, endpoint=False)
        clear = sum(
            1 for a in angles
            if not any(obs.blocks_ray(sx, sy, a, self.radius_m)
                       for obs in self.obstacles)
        )
        return clear / n_angles

    def coverage_area_m2(self, sx: float, sy: float,
                          field_w: float, field_l: float) -> float:
        """Effective coverage area accounting for arc, obstacles, field bounds."""
        import math
        arc_frac   = self.arc_clearance(sx, sy)
        circle_area = math.pi * self.radius_m**2 * (self.arc_deg / 360)
        field_area  = field_w * field_l
        # Clip radius to field boundary
        max_r = min(sx, field_w-sx, sy, field_l-sy)
        eff_r = min(self.radius_m, max_r * 1.4)
        eff_area = math.pi * eff_r**2 * (self.arc_deg / 360)
        return float(min(eff_area * arc_frac, field_area))

    def duration_for_deficit(self, deficit_mm: float) -> float:
        """
        Minutes needed to apply deficit_mm of water.
        Uses actual Hunter precipitation rate.
        """
        if deficit_mm <= 0:
            return 0.0
        # precip_rate_mm_hr × efficiency × duration_hr = deficit_mm
        duration_hr  = deficit_mm / (self.precip_rate_mm_hr * self.efficiency)
        duration_min = duration_hr * 60
        return round(min(float(duration_min), 90.0), 1)   # cap at 90 min

    def volume_for_duration(self, duration_min: float) -> float:
        """Litres used in given duration."""
        return round(self.flow_L_min * duration_min, 2)

    def select_best_arc(self, sx: float, sy: float) -> float:
        """Choose the arc option that maximises clear coverage."""
        best_arc  = 360
        best_clear = 0.0
        for arc in self.specs["arc_options_deg"]:
            self.arc_deg = arc
            c = self.arc_clearance(sx, sy, n_angles=72)
            if c > best_clear:
                best_clear = c
                best_arc   = arc
        self.arc_deg = best_arc
        return best_arc


def select_hunter_model(field_w: float, field_l: float) -> str:
    """Auto-select Hunter model based on field size."""
    max_dim = max(field_w, field_l) / 2   # max radius needed
    if max_dim <= 3.7:
        return "MP1000"
    elif max_dim <= 6.1:
        return "MP2000"
    else:
        return "MP3000"


# ══════════════════════════════════════════════════════════════════
# REGRESSION GEOMETRY MODULE (Novel)
# ══════════════════════════════════════════════════════════════════

class SprinklerRegressionModule:
    """
    Novel: Regression-based sprinkler geometry predictor.
    Trains on synthetic irrigation scenarios and predicts:
      - Optimal irrigation duration (minutes)
      - Expected coverage radius (metres)

    Three models compared: Linear, Ridge, Random Forest
    """

    def __init__(self):
        self.models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression":  Ridge(alpha=1.0),
            "Random Forest":     RandomForestRegressor(
                n_estimators=100, random_state=42, max_depth=8),
        }
        self.scaler_X   = StandardScaler()
        self.scaler_y   = StandardScaler()
        self.is_trained = False
        self.metrics    = {}
        self.best_model_name = None

    def _generate_training_data(self, n_samples: int = 2000) -> Tuple:
        """
        Generate synthetic irrigation scenarios for regression training.
        Features: moisture_deficit, fc, pwp, stress_threshold,
                  root_depth, field_area, et_rate, kc
        Targets: duration_min, coverage_radius_m
        """
        np.random.seed(42)
        rows = []
        for _ in range(n_samples):
            # Random crop parameters
            fc        = np.random.uniform(0.25, 0.50)
            pwp       = fc - np.random.uniform(0.10, 0.25)
            current_m = np.random.uniform(pwp, fc)
            target_m  = np.random.uniform(current_m, fc)
            deficit   = max(0, target_m - current_m)
            stress_th = pwp + 0.4 * (fc - pwp)
            root_d    = np.random.uniform(0.3, 3.0)
            field_a   = np.random.uniform(100, 1000)
            et_rate   = np.random.uniform(2, 8)
            kc        = np.random.uniform(0.7, 1.3)

            # Physics-based labels (ground truth for regression)
            deficit_mm    = deficit * root_d * 1000
            precip_rate   = 10.0 * 0.92   # Hunter MP efficiency
            duration_min  = min((deficit_mm / precip_rate) * 60, 90)

            # Radius derived from field area (matched to actual input feature)
            radius = np.sqrt(field_a / np.pi) * 0.8
            radius = np.clip(radius, 2.4, 10.7)

            rows.append([
                deficit, fc, pwp, stress_th,
                root_d, field_a, et_rate, kc,
                duration_min, radius
            ])

        arr = np.array(rows)
        X   = arr[:, :8]
        y   = arr[:, 8:]   # [duration_min, radius_m]
        return X, y

    def train(self) -> Dict:
        """Train all 3 regression models and compare."""
        print("  Training sprinkler regression models...")
        X, y = self._generate_training_data(2000)

        # 80/20 split
        split    = int(0.8 * len(X))
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        X_tr_s = self.scaler_X.fit_transform(X_tr)
        X_te_s = self.scaler_X.transform(X_te)

        print(f"  {'Model':<22} {'MAE_dur':>10} {'MAE_rad':>10} {'R2_dur':>8} {'R2_rad':>8}")
        print("  " + "-" * 62)

        best_r2 = -999
        for name, model in self.models.items():
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_te_s)

            mae_dur = mean_absolute_error(y_te[:, 0], y_pred[:, 0])
            mae_rad = mean_absolute_error(y_te[:, 1], y_pred[:, 1])
            r2_dur  = r2_score(y_te[:, 0], y_pred[:, 0])
            r2_rad  = r2_score(y_te[:, 1], y_pred[:, 1])

            self.metrics[name] = {
                "mae_duration": round(mae_dur, 3),
                "mae_radius":   round(mae_rad, 3),
                "r2_duration":  round(r2_dur,  4),
                "r2_radius":    round(r2_rad,  4),
            }
            print(f"  {name:<22} {mae_dur:>10.3f} {mae_rad:>10.3f} "
                  f"{r2_dur:>8.4f} {r2_rad:>8.4f}")

            avg_r2 = (r2_dur + r2_rad) / 2
            if avg_r2 > best_r2:
                best_r2 = avg_r2
                self.best_model_name = name

        self.is_trained = True
        print(f"  Best model: {self.best_model_name} (avg R²={best_r2:.4f})")
        return self.metrics

    def predict(self, moisture_deficit: float, fc: float, pwp: float,
                stress_threshold: float, root_depth_m: float,
                field_area_m2: float, et_rate: float,
                kc: float) -> Dict:
        """
        Predict irrigation duration and coverage radius.
        Uses best model from training.
        """
        if not self.is_trained:
            self.train()

        X = np.array([[moisture_deficit, fc, pwp, stress_threshold,
                        root_depth_m, field_area_m2, et_rate, kc]])
        X_s   = self.scaler_X.transform(X)
        model = self.models[self.best_model_name]
        pred  = model.predict(X_s)[0]

        duration_min = float(np.clip(pred[0], 0, 90))
        radius_m     = float(np.clip(pred[1], 2.4, 10.7))

        return {
            "duration_min":   round(duration_min, 1),
            "radius_m":       round(radius_m, 2),
            "model_used":     self.best_model_name,
            "all_predictions": {
                name: model.predict(X_s)[0].tolist()
                for name, model in self.models.items()
            }
        }


# ══════════════════════════════════════════════════════════════════
# WATER LEDGER
# ══════════════════════════════════════════════════════════════════

class WaterLedger:
    """
    Tracks water usage per plot per day.
    Novel: stores actual vs timer baseline vs threshold baseline.
    """
    def __init__(self):
        self.entries: List[Dict] = []

    def record(self, day: int, plot_id: str, crop: str,
               smart_L: float, timer_L: float,
               duration_min: float, radius_m: float,
               model_used: str, trigger: str):
        self.entries.append({
            "day":          day,
            "plot_id":      plot_id,
            "crop":         crop,
            "smart_L":      round(smart_L, 2),
            "timer_L":      round(timer_L, 2),
            "saved_L":      round(max(0, timer_L - smart_L), 2),
            "duration_min": duration_min,
            "radius_m":     radius_m,
            "model_used":   model_used,
            "trigger":      trigger,
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)

    def summary(self) -> Dict:
        df = self.to_dataframe()
        if df.empty:
            return {}
        total_smart = df["smart_L"].sum()
        total_timer = df["timer_L"].sum()
        total_saved = df["saved_L"].sum()
        return {
            "total_smart_L":  round(total_smart, 1),
            "total_timer_L":  round(total_timer, 1),
            "total_saved_L":  round(total_saved, 1),
            "saved_pct":      round(total_saved / max(total_timer, 1e-6) * 100, 1),
            "total_events":   len(df),
            "avg_duration_min": round(df["duration_min"].mean(), 1),
            "avg_radius_m":   round(df["radius_m"].mean(), 2),
        }


# ══════════════════════════════════════════════════════════════════
# MAIN SPRINKLER CONTROLLER
# ══════════════════════════════════════════════════════════════════

class HunterSprinklerController:
    """
    Integrates Hunter MP Rotator + Regression module + Water Ledger.
    Called by fog scheduler after irrigation decision.
    """

    def __init__(self):
        self.regression = SprinklerRegressionModule()
        self.ledger     = WaterLedger()
        self.sprinklers: Dict[str, HunterMPRotator] = {}
        self._setup_done = False

    def setup(self, plot_assignments: List[Dict]):
        """Build Hunter sprinkler for each plot."""
        print("\n  Setting up Hunter MP Rotators per plot ...")
        print(f"  {'Plot':<10} {'Crop':<10} {'Model':<10} "
              f"{'Radius(m)':>10} {'Flow(L/min)':>12} {'Arc':>6}")
        print("  " + "-" * 62)

        for pa in plot_assignments:
            pid   = pa["plot"]
            crop  = pa["crop"]
            cp    = CROP_PARAMS.get(crop, CROP_PARAMS["Wheat"])
            w, l  = cp["field_size_m"]
            model_name = cp.get("recommended_model",
                                 select_hunter_model(w, l))
            arc   = cp.get("arc_deg", 360)

            rotator = HunterMPRotator(
                model_name   = model_name,
                arc_deg      = arc,
                pressure_kpa = 280.0,
            )

            # Add coconut tree obstacles for coconut plot
            if crop == "Coconut":
                for tx in np.arange(7.5, w, 7.5):
                    for ty in np.arange(7.5, l, 7.5):
                        if abs(tx - w/2) < 4 and abs(ty - l/2) < 4:
                            continue
                        rotator.obstacles.append(Obstacle(
                            x=float(tx), y=float(ty),
                            radius=3.5, height=15.0,
                            obstacle_type="tree"))

            self.sprinklers[pid] = rotator
            print(f"  {pid:<10} {crop:<10} {model_name:<10} "
                  f"{rotator.radius_m:>10.1f} {rotator.flow_L_min:>12.2f} "
                  f"{rotator.arc_deg:>5}°")

        # Train regression models
        print()
        self.regression.train()
        self._setup_done = True

    def execute_irrigation(self,
                            day: int,
                            plot_id: str,
                            crop: str,
                            stage: str,
                            current_moisture: float,
                            et_rate_mm_day: float,
                            trigger: str = "FOG_SCHEDULER") -> Dict:
        """
        Full pipeline per your diagram:
        1. Check ET deficit
        2. Check moisture deficit
        3. Regression → geometry
        4. Hunter MP fires
        5. Water ledger update
        """
        cp    = CROP_PARAMS.get(crop, CROP_PARAMS["Wheat"])
        stage_params = cp["growth_stages"].get(stage, {})
        fc    = stage_params.get("fc",  cp["fc"])
        pwp   = stage_params.get("pwp", cp["pwp"])
        opt_low   = cp["optimal_low"]
        stress_th = cp["stress_threshold"]

        # ── Step 1: ET deficit check ──────────────────────────────
        et_deficit = et_rate_mm_day * cp["kc"]   # crop water demand
        et_met     = current_moisture > opt_low   # is ET being met?

        # ── Step 2: Moisture deficit check ───────────────────────
        moisture_deficit = max(0.0, opt_low + 0.05 - current_moisture)

        if moisture_deficit <= 0 and et_met:
            return {"irrigate": False, "reason": "No deficit"}

        # ── Step 3: Regression → geometry ────────────────────────
        w, l = cp["field_size_m"]
        reg_result = self.regression.predict(
            moisture_deficit = moisture_deficit,
            fc               = fc,
            pwp              = pwp,
            stress_threshold = stress_th,
            root_depth_m     = cp["root_depth_m"],
            field_area_m2    = w * l,
            et_rate          = et_rate_mm_day,
            kc               = cp["kc"],
        )

        duration_min = reg_result["duration_min"]
        radius_m     = reg_result["radius_m"]

        if duration_min < 1.0:
            return {"irrigate": False, "reason": "Duration too short"}

        # ── Step 4: Hunter MP fires ───────────────────────────────
        rotator = self.sprinklers.get(plot_id)
        if rotator:
            sx, sy = w / 2, l / 2
            # Auto-select best arc given obstacles
            best_arc = rotator.select_best_arc(sx, sy)
            coverage_m2 = rotator.coverage_area_m2(sx, sy, w, l)

            # Use physics-based duration if regression gives unrealistic value
            phys_deficit_mm = moisture_deficit * cp["root_depth_m"] * 1000
            phys_duration   = rotator.duration_for_deficit(phys_deficit_mm)

            # Take weighted average: 60% regression + 40% physics
            final_duration = 0.6 * duration_min + 0.4 * phys_duration
            final_duration = round(min(final_duration, 90.0), 1)

            smart_L = rotator.volume_for_duration(final_duration)
        else:
            coverage_m2    = w * l * 0.7
            final_duration = duration_min
            smart_L        = final_duration * 0.5   # fallback

        
        daily_water_L = cp.get("water_per_day_L", 10)
        timer_L = daily_water_L * 2.0   # timer system runs twice regardless of deficit

        # ── Step 5: Water ledger ──────────────────────────────────
        self.ledger.record(
            day=day, plot_id=plot_id, crop=crop,
            smart_L=smart_L, timer_L=timer_L,
            duration_min=final_duration,
            radius_m=radius_m,
            model_used=reg_result["model_used"],
            trigger=trigger,
        )

        return {
            "irrigate":        True,
            "duration_min":    final_duration,
            "volume_L":        smart_L,
            "radius_m":        radius_m,
            "coverage_m2":     round(coverage_m2, 1),
            "hunter_model":    rotator.model_name if rotator else "MP2000",
            "arc_deg":         rotator.arc_deg if rotator else 360,
            "regression_model":reg_result["model_used"],
            "et_deficit_mm":   round(et_deficit, 2),
            "moisture_deficit":round(moisture_deficit, 4),
            "timer_L":         timer_L,
            "saved_L":         round(max(0, timer_L - smart_L), 2),
            "trigger":         trigger,
        }
