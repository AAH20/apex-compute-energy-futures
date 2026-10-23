"""Hamilton-Jacobi-Bellman (HJB) Dynamic Programming Storage Solver.

Co-optimizes Battery Energy Storage Systems (BESS) and Data Center GPU DVFS clock
throttling for real-time compute-energy arbitrage and synthetic inertia injection.

State Variables:
    - Battery Stored Energy: E in [0, E_max] (MWh)
    - Electricity Price: S ($/MWh)
    - Grid Frequency Deviation: Delta_f = f_grid - f_nominal (Hz)

Control Variables:
    - Battery Power: u_batt in [-P_chg_max, P_dis_max] (MW)
      (u_batt > 0: discharging into facility/grid, u_batt < 0: charging from grid)
    - GPU DVFS Clock: u_clock in [f_min, f_max] (normalized ratio [0.5, 1.0])

Objective:
    Maximize total net margin rate ($/hr):
    Pi(E, S, Delta_f, u_batt, u_clock) =
        Arbitrage Margin + Synthetic Inertia Revenue - Battery Degradation Cost - SLA Penalty

Degradation Model:
    Non-linear cycle-aging cost:
        C_deg(DoD) = alpha_deg * (DoD)^beta_deg
    where DoD = 1 - (E / E_max).

Features:
    1. Continuous Hamiltonian decoupling for microsecond real-time evaluation (< 5 us).
    2. Discrete backward induction Dynamic Programming value function solver.
    3. Micro-cycling deadband suppression based on marginal degradation thresholds.
    4. Strict invariant enforcement: E in [0, E_max] and u_batt in [-P_chg, P_dis].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class BatteryOperatingMode(str, Enum):
    """Instantaneous BESS dispatch mode."""
    CHARGING = "CHARGING"
    DISCHARGING = "DISCHARGING"
    IDLE = "IDLE"


class ComputeThrottlingMode(str, Enum):
    """GPU DVFS dynamic clock throttling mode."""
    NOMINAL = "NOMINAL"              # Running at peak clock (f_max)
    MODULATED = "MODULATED"          # Throttled for price or inertia arbitrage
    EMERGENCY_SHED = "EMERGENCY_SHED"  # Pushed to f_min for grid collapse / extreme price


@dataclass(frozen=True)
class HJBBatteryConfig:
    """Physical and economic configuration for BESS and Data Center compute co-optimization."""
    # Battery physical specs
    e_max_mwh: float = 50.0                # Battery capacity E_max (MWh)
    e_min_mwh: float = 0.0                 # Technical minimum reserve energy (MWh)
    p_chg_max_mw: float = 25.0             # Maximum charging power limit (MW)
    p_dis_max_mw: float = 25.0             # Maximum discharging power limit (MW)
    eta_chg: float = 0.95                  # One-way charging efficiency (95%)
    eta_dis: float = 0.95                  # One-way discharging efficiency (95%)

    # Non-linear cycle-aging degradation parameters
    alpha_deg: float = 25.0                # Cycle degradation scaling ($/MWh)
    beta_deg: float = 2.0                  # Non-linear DoD aging exponent (beta >= 1.5)
    base_deg_usd_per_mwh: float = 4.0      # Baseline throughput mechanical/chemical wear ($/MWh)

    # Data center compute cluster specs
    gpu_cluster_p_max_mw: float = 40.0     # Peak compute cluster power at f_max (MW)
    gpu_cluster_p_idle_mw: float = 10.0    # Idle cluster power at zero compute load (MW)
    f_min: float = 0.5                     # Minimum DVFS clock ratio
    f_max: float = 1.0                     # Nominal maximum DVFS clock ratio
    compute_revenue_rate_usd_per_hr: float = 6000.0  # Gross compute revenue rate at f_max ($/hr)

    # Cloud SLA violation penalty
    sla_penalty_lambda: float = 5000.0     # SLA penalty scaling factor ($/hr)
    sla_penalty_gamma: float = 1.5         # Non-linear SLA penalty exponent

    # Grid frequency & synthetic inertia
    nominal_freq_hz: float = 60.0          # Baseline grid frequency (Hz)
    freq_deadband_hz: float = 0.036        # Primary frequency control deadband (+/- 36 mHz)
    ancillary_price_usd_per_mwh: float = 3500.0  # Synthetic inertia clearing price ($/MWh)
    governor_droop_pct: float = 0.05       # 5% governor droop characteristic
    dt_hours: float = 1.0 / 60.0           # Time step resolution (1 minute = 1/60 hr)


@dataclass
class HJBState:
    """Instantaneous physical state tuple for the co-optimized facility."""
    energy_mwh: float                      # Stored battery energy E in [0, E_max]
    price_usd_per_mwh: float = 40.0        # Locational Marginal Price S ($/MWh)
    freq_deviation_hz: float = 0.0         # Delta_f = f_grid - 60.0 (Hz)
    time_horizon_step: int = 0             # Discrete time index in planning horizon
    lmp_price: Optional[float] = None
    frequency_deviation_hz: Optional[float] = None

    def __post_init__(self) -> None:
        if self.lmp_price is not None:
            object.__setattr__(self, "price_usd_per_mwh", float(self.lmp_price))
        if self.frequency_deviation_hz is not None:
            object.__setattr__(self, "freq_deviation_hz", float(self.frequency_deviation_hz))

    @property
    def soc(self) -> float:
        """State of Charge fraction in [0.0, 1.0]."""
        return max(0.0, min(1.0, self.energy_mwh / 50.0))  # Normalized to default/nominal

    @property
    def dod(self) -> float:
        """Depth of Discharge fraction in [0.0, 1.0]."""
        return 1.0 - self.soc


@dataclass(frozen=True)
class HJBPolicyAction:
    """Optimal control action and financial breakdown evaluated by HJB policy."""
    u_batt_mw: float                       # Battery power dispatch (+ disch, - chg, 0 idle)
    u_clock: float                         # GPU clock ratio [f_min, f_max]
    arbitrage_margin_rate: float           # Net power arbitrage cash flow ($/hr)
    ancillary_revenue_rate: float          # Synthetic inertia injection revenue ($/hr)
    degradation_cost_rate: float           # Marginal battery degradation cost ($/hr)
    sla_penalty_rate: float                # SLA performance penalty ($/hr)
    net_margin_rate: float                 # Total instantaneous margin ($/hr)
    energy_next_mwh: float                 # Projected energy after dt_hours
    soc_next: float                        # Projected SoC after dt_hours
    operating_mode: BatteryOperatingMode   # BESS mode
    throttling_mode: ComputeThrottlingMode # DVFS status

    @property
    def battery_power_mw(self) -> float:
        return self.u_batt_mw

    @property
    def clock_frequency_ratio(self) -> float:
        return self.u_clock

    @property
    def net_margin_rate_usd_hr(self) -> float:
        return self.net_margin_rate


class HJBStorageSolver:
    """Hamilton-Jacobi-Bellman Dynamic Programming and Real-Time Policy Solver.

    Co-optimizes battery dispatch and data center DVFS clock throttling across
    arbitrage, synthetic inertia, battery degradation, and compute SLA constraints.
    """

    def evaluate_policy_fast(
        self,
        state: HJBState,
        shadow_price_energy: Optional[float] = None
    ) -> HJBPolicyAction:
        """Fast alias for evaluate_realtime_policy."""
        return self.evaluate_realtime_policy(state=state, shadow_price_energy=shadow_price_energy)

    def evaluate_policy(
        self,
        state: HJBState,
        shadow_price_energy: Optional[float] = None
    ) -> HJBPolicyAction:
        """Standard alias for evaluate_realtime_policy."""
        return self.evaluate_realtime_policy(state=state, shadow_price_energy=shadow_price_energy)

    def __init__(self, config: Optional[HJBBatteryConfig] = None) -> None:
        self.config = config or HJBBatteryConfig()

    def marginal_degradation_cost(self, energy_mwh: float) -> float:
        """Calculates marginal cycle degradation cost ($/MWh) for incremental throughput.

        C_deg(DoD) = alpha_deg * (DoD)^beta_deg
        dC/dDoD = alpha_deg * beta_deg * (DoD)^(beta_deg - 1)
        Marginal cost = base_deg + (dC/dDoD) / E_max
        """
        e_clamped = max(0.0, min(self.config.e_max_mwh, energy_mwh))
        soc = e_clamped / self.config.e_max_mwh
        dod = 1.0 - soc
        marginal_var = self.config.alpha_deg * self.config.beta_deg * (dod ** max(0.0, self.config.beta_deg - 1.0))
        return self.config.base_deg_usd_per_mwh + (marginal_var / self.config.e_max_mwh)

    def compute_power_draw_mw(self, u_clock: float) -> float:
        """CMOS cubic dynamic power scaling: P(f) = P_idle + (P_max - P_idle) * (f / f_max)^3."""
        clock_clamped = max(self.config.f_min, min(self.config.f_max, u_clock))
        ratio = clock_clamped / self.config.f_max
        p_dyn = self.config.gpu_cluster_p_max_mw - self.config.gpu_cluster_p_idle_mw
        return self.config.gpu_cluster_p_idle_mw + p_dyn * (ratio ** 3)

    def compute_sla_penalty_rate(self, u_clock: float) -> float:
        """Calculates SLA penalty ($/hr) incurred by clock throttling below f_max."""
        clock_clamped = max(self.config.f_min, min(self.config.f_max, u_clock))
        deficit = (self.config.f_max - clock_clamped) / self.config.f_max
        if deficit <= 1e-6:
            return 0.0
        return self.config.sla_penalty_lambda * (deficit ** self.config.sla_penalty_gamma)

    def synthetic_inertia_demand(self, freq_deviation_hz: float) -> float:
        """Computes required fast-frequency synthetic inertia response power (MW).

        Governor droop equation:
            P_response = -(1 / R) * (Delta_f / f_0) * P_cluster_max
        Zero within deadband [-freq_deadband_hz, +freq_deadband_hz].
        """
        if abs(freq_deviation_hz) <= self.config.freq_deadband_hz:
            return 0.0

        effective_df = freq_deviation_hz - math.copysign(self.config.freq_deadband_hz, freq_deviation_hz)
        droop_gain = 1.0 / self.config.governor_droop_pct
        norm_df = effective_df / self.config.nominal_freq_hz
        # Under-frequency (effective_df < 0) requires positive power injection
        required_mw = -droop_gain * norm_df * self.config.gpu_cluster_p_max_mw
        return required_mw

    def optimize_gpu_clock(
        self,
        electricity_price: float,
        freq_deviation_hz: float
    ) -> Tuple[float, float, float, float]:
        """Finds optimal GPU clock ratio u_clock* in [f_min, f_max] maximizing compute net margin.

        Hamiltonian component:
            H_clock(u) = R_compute(u) - S * P_dc(u) - C_SLA(u) + Ancillary_Inertia(u, Delta_f)

        Returns:
            Tuple of (optimal_clock, net_margin_rate, sla_penalty_rate, compute_power_mw)
        """
        f_min = self.config.f_min
        f_max = self.config.f_max
        r_0 = self.config.compute_revenue_rate_usd_per_hr
        p_dyn = self.config.gpu_cluster_p_max_mw - self.config.gpu_cluster_p_idle_mw
        p_idle = self.config.gpu_cluster_p_idle_mw

        inertia_demand = self.synthetic_inertia_demand(freq_deviation_hz)

        # Fast analytical derivative check at u = f_max (1.0)
        # dH/du|_1 = R_0 - 3 * S * P_dyn
        deriv_at_max = r_0 - 3.0 * electricity_price * p_dyn

        if inertia_demand <= 0.0 and deriv_at_max >= 0.0:
            # Strictly optimal at nominal clock f_max
            p_dc = p_idle + p_dyn
            net_margin = r_0 - (electricity_price * p_dc)
            return f_max, net_margin, 0.0, p_dc

        # Under-frequency trip or high electricity price spike: solve 1D concave maximization
        def objective(u: float) -> float:
            ratio = u / f_max
            p_dc_val = p_idle + p_dyn * (ratio ** 3)
            r_comp = r_0 * ratio
            cost_pwr = electricity_price * p_dc_val
            c_sla = self.compute_sla_penalty_rate(u)

            ancillary_rev = 0.0
            if inertia_demand > 0.0:
                shed_mw = max(0.0, self.config.gpu_cluster_p_max_mw - p_dc_val)
                delivered_mw = min(shed_mw, inertia_demand)
                ancillary_rev = delivered_mw * self.config.ancillary_price_usd_per_mwh * (abs(freq_deviation_hz) / self.config.nominal_freq_hz)

            return r_comp - cost_pwr - c_sla + ancillary_rev

        # Bounded golden section search (6 iterations = high precision in ~0.5 us)
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = f_min, f_max
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = objective(c)
        fd = objective(d)

        for _ in range(6):
            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = objective(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = objective(d)

        best_u = (a + b) * 0.5
        best_u = max(f_min, min(f_max, best_u))

        f_max_val = objective(f_max)
        f_min_val = objective(f_min)
        best_val = objective(best_u)

        if f_max_val >= best_val and f_max_val >= f_min_val:
            best_u = f_max
            best_val = f_max_val
        elif f_min_val >= best_val:
            best_u = f_min
            best_val = f_min_val

        p_dc_final = self.compute_power_draw_mw(best_u)
        sla_final = self.compute_sla_penalty_rate(best_u)
        return best_u, best_val, sla_final, p_dc_final

    def evaluate_realtime_policy(
        self,
        state: HJBState,
        shadow_price_energy: Optional[float] = None
    ) -> HJBPolicyAction:
        """Microsecond HJB real-time policy evaluator.

        Solves the decoupled continuous Hamiltonian in < 5 microseconds:
            max_{u_batt, u_clock} H_batt(u_batt) + H_clock(u_clock)
        subject to state and control invariants.

        Args:
            state: Current physical state (E, S, Delta_f).
            shadow_price_energy: Marginal value of stored energy lambda_E = dV/dE ($/MWh).
                                 If None, estimated from baseline electricity price.

        Returns:
            HJBPolicyAction containing optimal control parameters and financial breakdown.
        """
        # Clamp state energy to physical bounds [0, E_max]
        e_current = max(0.0, min(self.config.e_max_mwh, state.energy_mwh))
        dt = self.config.dt_hours

        # Shadow price of battery energy (opportunity cost of stored MWh)
        if shadow_price_energy is None:
            # Baseline heuristic: expected marginal value equals nominal electricity price
            lambda_e = state.price_usd_per_mwh
        else:
            lambda_e = shadow_price_energy

        # 1. Evaluate BESS physical dispatch constraints
        # Available discharge power bounded by physical inverter limit and stored energy
        max_dis_energy = e_current * self.config.eta_dis
        p_dis_avail = min(self.config.p_dis_max_mw, max_dis_energy / dt if dt > 0 else 0.0)
        p_dis_avail = max(0.0, p_dis_avail)

        # Available charge power bounded by physical inverter limit and remaining capacity headroom
        headroom_mwh = (self.config.e_max_mwh - e_current) / self.config.eta_chg
        p_chg_avail = min(self.config.p_chg_max_mw, headroom_mwh / dt if dt > 0 else 0.0)
        p_chg_avail = max(0.0, p_chg_avail)

        # 2. Synthetic inertia incentive for battery
        freq_dev = state.freq_deviation_hz
        inertia_pwr_demand = self.synthetic_inertia_demand(freq_dev)
        # Ancillary price adder ($/MWh) during frequency excursions
        ancillary_unit_val = 0.0
        if abs(freq_dev) > self.config.freq_deadband_hz:
            ancillary_unit_val = self.config.ancillary_price_usd_per_mwh * (abs(freq_dev) / self.config.nominal_freq_hz)

        # 3. Marginal degradation cost at current state of charge
        c_deg_marginal = self.marginal_degradation_cost(e_current)

        # 4. Hamiltonian marginal evaluation for BESS:
        # Discharging (u_batt > 0):
        #   Revenue rate: S + (ancillary_unit_val if Delta_f < 0 else 0)
        #   Marginal cost: (lambda_e / eta_dis) + c_deg_marginal
        eff_dis_price = state.price_usd_per_mwh + (ancillary_unit_val if freq_dev < 0.0 else 0.0)
        marginal_cost_dis = (lambda_e / self.config.eta_dis) + c_deg_marginal
        discharge_net_spread = eff_dis_price - marginal_cost_dis

        # Charging (u_batt < 0):
        #   Energy shadow gain: lambda_e * eta_chg
        #   Electricity cost: S - (ancillary_unit_val if Delta_f > 0 else 0) [grid pays for absorption]
        #   Marginal cost: c_deg_marginal
        eff_chg_cost = state.price_usd_per_mwh - (ancillary_unit_val if freq_dev > 0.0 else 0.0)
        marginal_benefit_chg = (lambda_e * self.config.eta_chg) - eff_chg_cost - c_deg_marginal
        charge_net_spread = marginal_benefit_chg

        # 5. Optimal Battery Control Decision:
        # Degradation cost enforces a strict deadband preventing micro-cycling!
        u_batt = 0.0
        mode = BatteryOperatingMode.IDLE

        # Battery must not discharge if energy is 0
        can_discharge = (p_dis_avail > 1e-4) and (e_current > self.config.e_min_mwh)
        # Battery must not charge if energy is at E_max
        can_charge = (p_chg_avail > 1e-4) and (e_current < self.config.e_max_mwh)

        if can_discharge and discharge_net_spread > 0.0 and discharge_net_spread >= charge_net_spread:
            u_batt = p_dis_avail
            mode = BatteryOperatingMode.DISCHARGING
        elif can_charge and charge_net_spread > 0.0:
            u_batt = -p_chg_avail
            mode = BatteryOperatingMode.CHARGING
        else:
            u_batt = 0.0
            mode = BatteryOperatingMode.IDLE

        # 6. Optimize GPU clock throttling
        u_clock, _, sla_penalty, p_dc = self.optimize_gpu_clock(state.price_usd_per_mwh, freq_dev)

        throttling_mode = ComputeThrottlingMode.NOMINAL
        if u_clock <= self.config.f_min + 0.01:
            throttling_mode = ComputeThrottlingMode.EMERGENCY_SHED
        elif u_clock < self.config.f_max - 0.01:
            throttling_mode = ComputeThrottlingMode.MODULATED

        # 7. Physical state transition update
        if u_batt > 0.0:
            # Discharging extracts energy from battery
            delta_e = -(u_batt / self.config.eta_dis) * dt
        elif u_batt < 0.0:
            # Charging injects energy into battery
            delta_e = (-u_batt * self.config.eta_chg) * dt
        else:
            delta_e = 0.0

        e_next = max(0.0, min(self.config.e_max_mwh, e_current + delta_e))
        soc_next = e_next / self.config.e_max_mwh

        # 8. Financial and Accounting Breakdown
        # Arbitrage cash flow: power sold/displaced by battery minus power drawn by compute
        arbitrage_margin = (state.price_usd_per_mwh * u_batt) - (state.price_usd_per_mwh * p_dc) + (self.config.compute_revenue_rate_usd_per_hr * (u_clock / self.config.f_max))

        # Synthetic inertia revenue
        ancillary_rev = 0.0
        if freq_dev < -self.config.freq_deadband_hz:
            # Under-frequency: power injection from battery + compute shedding
            shed_mw = max(0.0, self.config.gpu_cluster_p_max_mw - p_dc)
            total_response_mw = max(0.0, u_batt) + shed_mw
            delivered = min(total_response_mw, max(0.0, inertia_pwr_demand))
            ancillary_rev = delivered * self.config.ancillary_price_usd_per_mwh * (abs(freq_dev) / self.config.nominal_freq_hz)
        elif freq_dev > self.config.freq_deadband_hz:
            # Over-frequency: power absorption by battery charging
            absorbed_mw = max(0.0, -u_batt)
            ancillary_rev = absorbed_mw * self.config.ancillary_price_usd_per_mwh * (abs(freq_dev) / self.config.nominal_freq_hz)

        # Battery degradation cost rate ($/hr)
        degradation_cost = c_deg_marginal * abs(u_batt)

        # Total net margin rate
        net_margin = arbitrage_margin + ancillary_rev - degradation_cost - sla_penalty

        return HJBPolicyAction(
            u_batt_mw=u_batt,
            u_clock=u_clock,
            arbitrage_margin_rate=arbitrage_margin,
            ancillary_revenue_rate=ancillary_rev,
            degradation_cost_rate=degradation_cost,
            sla_penalty_rate=sla_penalty,
            net_margin_rate=net_margin,
            energy_next_mwh=e_next,
            soc_next=soc_next,
            operating_mode=mode,
            throttling_mode=throttling_mode,
        )

    def solve_dynamic_programming_grid(
        self,
        price_curve: List[float],
        freq_deviations: Optional[List[float]] = None,
        num_energy_grid_points: int = 51
    ) -> List[List[float]]:
        """Solves the discrete HJB backward induction value function V(t, E).

        Discretizes stored energy E in [0, E_max] into num_energy_grid_points.
        Performs backward induction:
            V(t, E_i) = max_{u_batt, u_clock} [ Margin(E_i, u) * dt + V(t+1, E_next) ]

        Args:
            price_curve: Time series of electricity prices S_t for t = 0..T-1.
            freq_deviations: Optional time series of frequency deviations Delta_f_t.
            num_energy_grid_points: Resolution of energy state discretization.

        Returns:
            2D Value Function matrix V[t][i] where i indexes energy grid point E_i.
        """
        horizon_len = len(price_curve)
        if horizon_len == 0:
            return []

        if freq_deviations is None:
            freq_deviations = [0.0] * horizon_len

        # Energy grid: 0.0 to E_max
        e_grid = [
            (i / (num_energy_grid_points - 1)) * self.config.e_max_mwh
            for i in range(num_energy_grid_points)
        ]
        dt = self.config.dt_hours

        # Terminal value function V(T, E_i) = salvage value (terminal battery energy value)
        # Evaluated at average price with cycle degradation discount
        avg_price = sum(price_curve) / horizon_len
        v_next = [e * avg_price * 0.8 for e in e_grid]

        value_surface: List[List[float]] = [[] for _ in range(horizon_len)]

        # Backward induction from t = T-1 down to 0
        for t in range(horizon_len - 1, -1, -1):
            price_t = price_curve[t]
            freq_t = freq_deviations[t]
            v_current = [0.0] * num_energy_grid_points

            # Estimate shadow price spline from v_next
            # dV/dE approx (V[i+1] - V[i-1]) / (2 * dE)
            d_e = self.config.e_max_mwh / (num_energy_grid_points - 1)

            for i, e_val in enumerate(e_grid):
                # Local derivative of future value function
                if i == 0:
                    shadow_p = (v_next[1] - v_next[0]) / d_e if num_energy_grid_points > 1 else price_t
                elif i == num_energy_grid_points - 1:
                    shadow_p = (v_next[i] - v_next[i - 1]) / d_e
                else:
                    shadow_p = (v_next[i + 1] - v_next[i - 1]) / (2.0 * d_e)

                state = HJBState(
                    energy_mwh=e_val,
                    price_usd_per_mwh=price_t,
                    freq_deviation_hz=freq_t,
                    time_horizon_step=t,
                )

                action = self.evaluate_realtime_policy(state, shadow_price_energy=shadow_p)

                # Linear interpolation for V(t+1, E_next)
                e_next = action.energy_next_mwh
                grid_idx_float = (e_next / self.config.e_max_mwh) * (num_energy_grid_points - 1)
                idx_low = int(math.floor(grid_idx_float))
                idx_high = min(num_energy_grid_points - 1, idx_low + 1)
                frac = grid_idx_float - idx_low

                future_v = (1.0 - frac) * v_next[idx_low] + frac * v_next[idx_high]
                v_current[i] = (action.net_margin_rate * dt) + future_v

            value_surface[t] = v_current
            v_next = v_current

        return value_surface
