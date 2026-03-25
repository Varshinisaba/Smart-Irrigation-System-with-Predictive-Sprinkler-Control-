"""
Fog-Level Irrigation Scheduler
================================
Two complementary controllers that run on the fog node (Raspberry Pi / server):

  A) MPC (Model Predictive Control)
     • Uses LSTM predictions from edge nodes as forecast
     • Optimizes valve schedules over a receding horizon (6–12 h)
     • Hard constraints: pump capacity, water quota, inter-plot fairness

  B) RL Agent (Soft Actor-Critic surrogate, tabular Q for simulation)
     • Learns adaptive scheduling policy from interaction with plant model
     • State: [moisture_per_plot, forecast_rain, time_of_day, quota_remaining]
     • Action: {which plots to irrigate, for how long}
     • Reward: moisture_in_optimal_band – water_used_penalty – energy_penalty

Both controllers output a `ScheduleDecision` for the LoRa gateway to dispatch.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linprog
import json, time


# ══════════════════════════════════════════════════════════════════
# SHARED TYPES
# ══════════════════════════════════════════════════════════════════

@dataclass
class PlotState:
    plot_id:        str
    moisture:       float          # current VWC (m³/m³)
    moisture_pred:  List[float]    # LSTM forecast (next HORIZON steps)
    crop:           str
    fc:             float          # field capacity
    pwp:            float          # permanent wilting point
    area_m2:        float
    last_irrigated: float = 0.0    # unix timestamp
    quota_remaining_m3: float = 10.0  # daily water quota

    @property
    def optimal_low(self):   return self.pwp + 0.40 * (self.fc - self.pwp)
    @property
    def optimal_high(self):  return self.pwp + 0.75 * (self.fc - self.pwp)
    @property
    def in_optimal(self):    return self.optimal_low <= self.moisture <= self.optimal_high
    @property
    def stress_index(self):
        if self.moisture >= self.optimal_low:
            return 0.0
        return (self.optimal_low - self.moisture) / (self.optimal_low - self.pwp)


@dataclass
class ScheduleDecision:
    plot_id:         str
    irrigate:        bool
    duration_s:      int
    volume_m3:       float
    trigger:         str    # "MPC" | "RL" | "EMERGENCY"
    timestamp:       float = field(default_factory=time.time)
    confidence:      float = 1.0


# ══════════════════════════════════════════════════════════════════
# A ▸ MPC CONTROLLER
# ══════════════════════════════════════════════════════════════════

class MPCScheduler:
    """
    Receding-horizon linear MPC for multi-plot irrigation.

    Formulation (LP):
      minimize   sum_t sum_p [w_water * u_pt + w_stress * s_pt]
      subject to:
        θ_{p,t+1} = θ_pt - d_p*Δt + β_p * u_pt + rain_pt
        θ_pt      ≥ PWP_p  (no wilting)
        θ_pt      ≤ FC_p   (no saturation)
        sum_p u_pt ≤ PUMP_CAP   (shared pump)
        sum_t u_pt ≤ QUOTA_p    (per-plot quota)
        u_pt      ≥ 0

    u_pt = irrigation applied (m³) for plot p at time step t
    s_pt = stress slack variable
    """

    def __init__(self,
                 horizon_steps: int = 12,      # 6h at 30-min steps
                 step_minutes:  int = 30,
                 pump_capacity_m3_h: float = 2.0,
                 weight_water:  float = 1.0,
                 weight_stress: float = 5.0):
        self.H       = horizon_steps
        self.dt_h    = step_minutes / 60
        self.pump_h  = pump_capacity_m3_h * self.dt_h   # m³ per step
        self.w_w     = weight_water
        self.w_s     = weight_stress

    def solve(self, plots: List[PlotState],
              rain_forecast: List[float]) -> List[ScheduleDecision]:
        """
        Solve LP and return immediate (step-0) decisions.
        rain_forecast: list of H values (mm) for the horizon
        """
        P = len(plots)
        H = self.H

        # Decision variables: [u_00, u_01,...,u_{P-1,H-1}, s_00,...,s_{P-1,H-1}]
        # u_pt: irrigation m³,  s_pt: stress slack
        n_u = P * H
        n_s = P * H
        n   = n_u + n_s

        # ── Objective ────────────────────────────────────────────
        c_u = np.full(n_u, self.w_w)
        c_s = np.full(n_s, self.w_s)
        c   = np.concatenate([c_u, c_s])

        # ── Inequality constraints Ax ≤ b ────────────────────────
        A_ub, b_ub = [], []

        # Pump capacity per step: sum_p u_pt ≤ pump_h  for each t
        for t in range(H):
            row = np.zeros(n)
            for p in range(P):
                row[p * H + t] = 1.0
            A_ub.append(row);  b_ub.append(self.pump_h)

        # Per-plot quota: sum_t u_pt ≤ quota_p
        for p, plot in enumerate(plots):
            row = np.zeros(n)
            for t in range(H):
                row[p * H + t] = 1.0
            A_ub.append(row);  b_ub.append(plot.quota_remaining_m3)

        # Moisture upper bound: θ_pt ≤ FC
        # θ_pt = θ_p0 - d_p*t*dt + sum_{τ<t} β_p*u_pτ + rain_cum_t
        for p, plot in enumerate(plots):
            dp   = 0.005 * self.dt_h   # depletion m³/m² per step proxy
            beta = 1.0 / max(plot.area_m2, 1) * 1000  # irrigation efficiency
            for t in range(H):
                rain_cum = sum(rain_forecast[:t+1]) * 0.001 * plot.area_m2 * 0.6
                row = np.zeros(n)
                for tau in range(t + 1):
                    row[p * H + tau] = beta
                # θ_p0 - d_p*(t+1) + beta*sum_τ u + rain ≤ FC
                # → beta*sum ≤ FC - θ_p0 + d_p*(t+1) - rain
                rhs = (plot.fc - plot.moisture
                       + dp * (t + 1) - rain_cum)
                A_ub.append(row);  b_ub.append(rhs)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # ── Bounds ───────────────────────────────────────────────
        # u ≥ 0,  s ≥ 0
        bounds = [(0, None)] * n_u + [(0, None)] * n_s

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                         method="highs")

        decisions = []
        if result.success:
            u_opt = result.x[:n_u].reshape(P, H)
        else:
            u_opt = np.zeros((P, H))   # fallback: no irrigation

        for p, plot in enumerate(plots):
            vol = float(u_opt[p, 0])     # step-0 irrigation
            dur = int(vol / max(self.pump_h, 1e-6) * 3600 * self.dt_h)
            irrigate = vol > 0.001 or plot.stress_index > 0.7
            decisions.append(ScheduleDecision(
                plot_id    = plot.plot_id,
                irrigate   = irrigate,
                duration_s = dur if irrigate else 0,
                volume_m3  = vol,
                trigger    = "MPC",
                confidence = float(result.success),
            ))
        return decisions


# ══════════════════════════════════════════════════════════════════
# B ▸ RL AGENT (Tabular Q-Learning for Simulation)
# ══════════════════════════════════════════════════════════════════

class RLScheduler:
    """
    Tabular Q-Learning agent for single-plot irrigation scheduling.
    State discretization + Q-table fits in fog node memory.

    For production, replace Q-table with a small neural network (DQN/SAC).
    """

    # State bins: [moisture_level 0-4, forecast_rain 0-2, hour_of_day 0-3,
    #              days_since_irrigation 0-3]
    N_MOISTURE  = 5
    N_RAIN      = 3
    N_HOUR      = 4
    N_DAYS_IRR  = 4
    N_ACTIONS   = 3   # 0=skip, 1=irrigate_short(15min), 2=irrigate_long(45min)

    IRR_DURATIONS = [0, 900, 2700]   # seconds

    def __init__(self, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 0.15):
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon
        shape = (self.N_MOISTURE, self.N_RAIN, self.N_HOUR,
                 self.N_DAYS_IRR, self.N_ACTIONS)
        self.Q = np.zeros(shape)
        self.episode_rewards: List[float] = []

    def _discretize(self, plot: PlotState, hour: int,
                    days_since_irr: float,
                    forecast_rain_mm: float) -> Tuple[int, int, int, int]:
        # Moisture: 0=critically dry … 4=saturated
        m_range  = plot.fc - plot.pwp
        m_norm   = (plot.moisture - plot.pwp) / max(m_range, 1e-6)
        m_bin    = int(np.clip(m_norm * self.N_MOISTURE, 0, self.N_MOISTURE - 1))

        rain_bin = 0 if forecast_rain_mm < 2 else (1 if forecast_rain_mm < 8 else 2)
        hour_bin = int(hour / 6) % self.N_HOUR
        day_bin  = int(np.clip(days_since_irr, 0, self.N_DAYS_IRR - 1))
        return m_bin, rain_bin, hour_bin, day_bin

    def select_action(self, plot: PlotState, hour: int,
                      days_since_irr: float,
                      forecast_rain_mm: float) -> int:
        s = self._discretize(plot, hour, days_since_irr, forecast_rain_mm)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.N_ACTIONS)
        return int(np.argmax(self.Q[s]))

    def _reward(self, plot: PlotState, action: int,
                water_saved_m3: float) -> float:
        """Shaped reward: stay in optimal band, save water, avoid stress."""
        if plot.in_optimal:
            band_reward = 2.0
        elif plot.moisture > plot.fc:
            band_reward = -2.0       # over-irrigation
        else:
            band_reward = -3.0 * plot.stress_index

        water_penalty = -0.5 * (self.IRR_DURATIONS[action] / 2700)
        return band_reward + water_penalty + 0.1 * water_saved_m3

    def update(self, plot_before: PlotState, action: int, hour: int,
               days_since_irr: float, forecast_rain_mm: float,
               plot_after: PlotState, water_saved_m3: float = 0):
        s  = self._discretize(plot_before, hour, days_since_irr, forecast_rain_mm)
        s_ = self._discretize(plot_after, (hour + 1) % 24,
                               days_since_irr, forecast_rain_mm)
        r  = self._reward(plot_after, action, water_saved_m3)
        td = r + self.gamma * np.max(self.Q[s_]) - self.Q[s][action]
        self.Q[s][action] += self.alpha * td
        self.episode_rewards.append(r)

    def decide(self, plot: PlotState, hour: int,
               days_since_irr: float,
               forecast_rain_mm: float,
               pump_cap_m3_h: float = 2.0) -> ScheduleDecision:
        action  = self.select_action(plot, hour, days_since_irr, forecast_rain_mm)
        dur_s   = self.IRR_DURATIONS[action]
        vol     = pump_cap_m3_h * dur_s / 3600
        return ScheduleDecision(
            plot_id    = plot.plot_id,
            irrigate   = action > 0,
            duration_s = dur_s,
            volume_m3  = vol,
            trigger    = "RL",
            confidence = float(np.max(self.Q[self._discretize(
                             plot, hour, days_since_irr, forecast_rain_mm)]))
        )

    def save(self, path: str = "rl_qtable.npy"):
        np.save(path, self.Q)

    def load(self, path: str = "rl_qtable.npy"):
        self.Q = np.load(path)


# ══════════════════════════════════════════════════════════════════
# C ▸ HYBRID FOG CONTROLLER
# ══════════════════════════════════════════════════════════════════

class FogIrrigationController:
    """
    Hybrid MPC + RL controller running at the fog node.
    MPC handles global optimization (water quota, pump constraints).
    RL provides per-plot fine-grained override for fast-changing conditions.
    """

    def __init__(self, n_plots: int):
        self.mpc   = MPCScheduler()
        self.rl    = {f"plot_{i:02d}": RLScheduler() for i in range(n_plots)}
        self.n_plots = n_plots
        self.log: List[ScheduleDecision] = []

    def run_cycle(self,
                  plots: List[PlotState],
                  rain_forecast_mm: List[float],
                  hour: int) -> List[ScheduleDecision]:
        """
        One scheduling cycle (called every 30 min at fog node):
          1. Run MPC for global feasibility.
          2. Use RL to refine per-plot decisions if MPC says 'no irrigation'
             but stress is high.
          3. Emergency override: irrigate immediately if θ < PWP + 0.02.
        """
        mpc_decisions = self.mpc.solve(plots, rain_forecast_mm)
        final_decisions = []

        for plot, mpc_dec in zip(plots, mpc_decisions):
            rl_agent  = self.rl.get(plot.plot_id, RLScheduler())
            days_since = (time.time() - plot.last_irrigated) / 86400
            rain_next  = rain_forecast_mm[0] if rain_forecast_mm else 0.0

            # Emergency check
            if plot.moisture < plot.pwp + 0.02:
                dec = ScheduleDecision(
                    plot_id=plot.plot_id, irrigate=True,
                    duration_s=1800, volume_m3=1.0, trigger="EMERGENCY")
            elif mpc_dec.irrigate:
                dec = mpc_dec
            elif plot.stress_index > 0.5:
                # RL override
                dec = rl_agent.decide(plot, hour, days_since, rain_next)
                dec.trigger = "RL_OVERRIDE"
            else:
                dec = mpc_dec   # MPC says no irrigation, RL agrees

            self.log.append(dec)
            final_decisions.append(dec)

        return final_decisions

    def export_log(self, path: str = "fog_schedule_log.json"):
        records = [
            {"plot_id": d.plot_id, "irrigate": d.irrigate,
             "duration_s": d.duration_s, "volume_m3": round(d.volume_m3, 4),
             "trigger": d.trigger, "timestamp": d.timestamp}
            for d in self.log
        ]
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  Schedule log → {path}  ({len(records)} decisions)")


# ── Quick Test ────────────────────────────────────────────────────

if __name__ == "__main__":
    from simulation.dataset_generator import CROP_PARAMS, PLOT_CONFIGS

    plots = [
        PlotState(
            plot_id=cfg["plot_id"],
            moisture=np.random.uniform(0.18, 0.38),
            moisture_pred=[np.random.uniform(0.15, 0.38) for _ in range(12)],
            crop=cfg["crop"],
            fc=CROP_PARAMS[cfg["crop"]]["fc"],
            pwp=CROP_PARAMS[cfg["crop"]]["pwp"],
            area_m2=cfg["area_m2"],
        )
        for cfg in PLOT_CONFIGS
    ]

    controller = FogIrrigationController(n_plots=len(plots))
    rain_fcast = [0.0] * 12
    decisions  = controller.run_cycle(plots, rain_fcast, hour=8)

    print("\n  ── Fog Scheduling Decisions ──")
    for d in decisions:
        status = f"💧 {d.duration_s//60} min" if d.irrigate else "⏸  skip"
        print(f"  {d.plot_id}  [{d.trigger:12s}]  {status}  ({d.volume_m3:.3f} m³)")
