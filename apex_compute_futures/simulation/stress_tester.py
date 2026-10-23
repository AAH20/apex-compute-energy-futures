"""Institutional Multi-Scenario Crisis & Black-Swan Stress Testing Engine.

Evaluates high-impact tail risks, extreme market dislocations, and physical grid
contingencies across compute-energy portfolios. Computes Value at Risk (VaR_0.999),
Expected Shortfall (CVaR_0.999), Maximum Drawdown (MDD), Recovery Time, and System
Survival Probabilities under high-dimensional Monte Carlo path simulations.
Zero third-party dependencies (pure Python standard library).
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import random
import time
from typing import Dict, List, Optional, Tuple


class StressScenarioType(str, Enum):
    """Categorical classification of institutional crisis scenarios."""
    WINTER_STORM_URI_2 = "WINTER_STORM_URI_2"
    FRONTIER_MODEL_LAUNCH = "FRONTIER_MODEL_LAUNCH"
    SUBSEA_TERABIT_PARTITION = "SUBSEA_TERABIT_PARTITION"
    CUSTOM = "CUSTOM"


@dataclass(frozen=True)
class StressScenario:
    """Quantitative definition of an institutional crisis stress scenario."""
    scenario_id: str
    scenario_type: StressScenarioType
    name: str
    description: str
    duration_hours: float
    
    # Grid & Physical Energy Shocks
    grid_frequency_hz: float             # e.g., 59.35 Hz (vs nominal 60.00 Hz)
    power_lmp_per_mwh: float             # e.g., $9,000/MWh (vs nominal $35-$45/MWh)
    datacenter_curtailment_pct: float    # e.g., 45% mandatory load shed
    
    # Compute Spot & Forward Market Shocks
    gpu_supply_withdrawal_count: int    # e.g., 60,000 GPUs withdrawn instantaneously
    retail_preemption_rate: float        # e.g., 0.85 (85% preemption on spot instances)
    forward_basis_shock_bps: float       # e.g., +600 bps forward-spot basis widening
    
    # Network Fabric & Telemetry Shocks
    network_latency_spike_ms: float      # e.g., 350.0 ms (vs nominal 12.0 ms)
    packet_loss_pct: float               # e.g., 8.5%
    regional_arbitrage_divergence_bps: float # e.g., +450 bps
    
    # Capital & Portfolio Boundary Constraints
    initial_portfolio_value_usd: float = 10_000_000.0
    capital_reserve_floor_usd: float = 2_000_000.0  # Basel / Vault mandatory reserve floor
    portfolio_gpu_capacity: int = 10_000             # Equivalent H100 fleet size
    committed_forward_contracts_mw: float = 25.0     # Power commitment in MW


@dataclass
class PathTrajectory:
    """Individual simulated Monte Carlo equity and drawdown trajectory."""
    path_id: int
    equity_series: List[float]
    drawdown_series: List[float]
    max_drawdown: float
    max_drawdown_usd: float
    recovered: bool
    recovery_time_hours: float
    survived: bool
    final_equity_usd: float
    total_loss_usd: float


@dataclass
class StressTestResult:
    """Comprehensive institutional stress-testing evaluation report."""
    scenario: StressScenario
    num_paths: int
    confidence_level: float               # e.g., 0.999
    
    # Risk Measures at Tail
    parametric_var_999: float            # VaR at 99.9% in USD
    historical_var_999: float            # Empirical VaR at 99.9% in USD
    cvar_999: float                      # Expected Shortfall (CVaR) in USD
    
    # Drawdown & Solvency
    max_drawdown_pct: float              # Portfolio Maximum Drawdown (%)
    max_drawdown_usd: float              # Portfolio Maximum Drawdown ($)
    avg_recovery_time_hours: float       # Mean time to recover prior high (hours)
    system_survival_probability: float   # Probability equity > capital_reserve_floor
    
    # PnL & Equity Distribution
    worst_case_loss_usd: float           # Max loss across all paths
    median_pnl_usd: float                # 50th percentile PnL
    p01_equity_usd: float                # 1st percentile terminal equity
    p05_equity_usd: float                # 5th percentile terminal equity
    p50_equity_usd: float                # Median terminal equity
    p95_equity_usd: float                # 95th percentile terminal equity
    p99_equity_usd: float                # 99th percentile terminal equity
    
    # Runtime & Visualization Data
    execution_time_ms: float
    fan_chart_percentiles: Dict[str, List[float]]  # Time-stepped ribbons for SVG
    sample_trajectories: List[List[float]]         # Representative paths for chart overlay

    @property
    def var_parametric_999(self) -> float:
        return self.parametric_var_999

    @property
    def cvar_parametric_999(self) -> float:
        return self.cvar_999

    @property
    def max_drawdown(self) -> float:
        return self.max_drawdown_pct / 100.0


def _rational_norm_inv(p: float) -> float:
    """High-precision Beasley-Springer-Moro / Acklam approximation of inverse standard normal CDF."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    
    # Coefficients in rational approximations
    a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return num / den
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        return num / den
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return -num / den


class InstitutionalStressTestingEngine:
    """Institutional Multi-Scenario Crisis & Black-Swan Stress Testing Engine."""

    def __init__(self, default_seed: Optional[int] = 42):
        self.default_seed = default_seed

    @classmethod
    def create_winter_storm_uri_scenario(
        cls,
        initial_capital: float = 10_000_000.0,
        reserve_floor: float = 2_000_000.0,
    ) -> StressScenario:
        """Generates the 'Winter Storm Uri 2.0' ERCOT grid frequency and LMP crisis."""
        return StressScenario(
            scenario_id="SCEN_URI_2.0",
            scenario_type=StressScenarioType.WINTER_STORM_URI_2,
            name="Winter Storm Uri 2.0 (ERCOT Grid Collapse)",
            description=(
                "ERCOT grid frequency plunges to 59.35 Hz (approaching 59.30 Hz UFLS Stage 1). "
                "Wholesale power LMP caps out at $9,000/MWh for 48 consecutive hours. "
                "ISO issues emergency 45% mandatory datacenter power curtailment order."
            ),
            duration_hours=48.0,
            grid_frequency_hz=59.35,
            power_lmp_per_mwh=9000.0,
            datacenter_curtailment_pct=45.0,
            gpu_supply_withdrawal_count=12_000,
            retail_preemption_rate=0.40,
            forward_basis_shock_bps=350.0,
            network_latency_spike_ms=45.0,
            packet_loss_pct=1.5,
            regional_arbitrage_divergence_bps=220.0,
            initial_portfolio_value_usd=initial_capital,
            capital_reserve_floor_usd=reserve_floor,
            portfolio_gpu_capacity=10_000,
            committed_forward_contracts_mw=25.0,
        )

    @classmethod
    def create_frontier_model_shock_scenario(
        cls,
        initial_capital: float = 10_000_000.0,
        reserve_floor: float = 2_000_000.0,
    ) -> StressScenario:
        """Generates the 'Frontier Model Launch Shock' instantaneous GPU liquidity drain."""
        return StressScenario(
            scenario_id="SCEN_FRONTIER_SHOCK",
            scenario_type=StressScenarioType.FRONTIER_MODEL_LAUNCH,
            name="Frontier Model Launch Shock (Compute Squeeze)",
            description=(
                "Tier-1 AI Lab instantaneously acquires 60,000 GPUs from secondary & spot venues. "
                "Retail & unreserved spot preemption rate explodes to 85%. "
                "Forward-spot basis spreads blow out by +600 bps with massive short squeeze."
            ),
            duration_hours=48.0,
            grid_frequency_hz=59.98,
            power_lmp_per_mwh=65.0,
            datacenter_curtailment_pct=0.0,
            gpu_supply_withdrawal_count=60_000,
            retail_preemption_rate=0.85,
            forward_basis_shock_bps=600.0,
            network_latency_spike_ms=30.0,
            packet_loss_pct=0.8,
            regional_arbitrage_divergence_bps=480.0,
            initial_portfolio_value_usd=initial_capital,
            capital_reserve_floor_usd=reserve_floor,
            portfolio_gpu_capacity=10_000,
            committed_forward_contracts_mw=25.0,
        )

    @classmethod
    def create_subsea_partition_scenario(
        cls,
        initial_capital: float = 10_000_000.0,
        reserve_floor: float = 2_000_000.0,
    ) -> StressScenario:
        """Generates the 'Subsea Terabit Partition' transatlantic fiber severance."""
        return StressScenario(
            scenario_id="SCEN_SUBSEA_PARTITION",
            scenario_type=StressScenarioType.SUBSEA_TERABIT_PARTITION,
            name="Subsea Terabit Partition (Cross-Venue Routing Severance)",
            description=(
                "Dual TAT-14 / Dunant subsea cable cuts trigger transatlantic routing detour. "
                "Cross-region latency surges +350 ms, packet loss hits 8.5%, desynchronizing "
                "cross-venue order book arbitrage between US-East and EU-West."
            ),
            duration_hours=48.0,
            grid_frequency_hz=59.99,
            power_lmp_per_mwh=45.0,
            datacenter_curtailment_pct=5.0,
            gpu_supply_withdrawal_count=8_000,
            retail_preemption_rate=0.30,
            forward_basis_shock_bps=280.0,
            network_latency_spike_ms=350.0,
            packet_loss_pct=8.5,
            regional_arbitrage_divergence_bps=650.0,
            initial_portfolio_value_usd=initial_capital,
            capital_reserve_floor_usd=reserve_floor,
            portfolio_gpu_capacity=10_000,
            committed_forward_contracts_mw=25.0,
        )

    def run_scenario(
        self,
        scenario_or_type: Union[StressScenarioType, StressScenario],
        num_paths: int = 10_000,
        time_steps: int = 48,
        confidence_level: float = 0.999,
        seed: Optional[int] = None,
    ) -> StressTestResult:
        """Executes stress test from scenario instance or scenario type enum."""
        if isinstance(scenario_or_type, StressScenario):
            scen = scenario_or_type
        elif scenario_or_type == StressScenarioType.WINTER_STORM_URI_2:
            scen = self.create_winter_storm_uri_scenario()
        elif scenario_or_type == StressScenarioType.FRONTIER_MODEL_LAUNCH:
            scen = self.create_frontier_model_shock_scenario()
        elif scenario_or_type == StressScenarioType.SUBSEA_TERABIT_PARTITION:
            scen = self.create_subsea_partition_scenario()
        else:
            scen = self.create_winter_storm_uri_scenario()

        return self.run_stress_test(
            scenario=scen,
            num_paths=num_paths,
            time_steps=time_steps,
            confidence_level=confidence_level,
            seed=seed,
        )

    def run_stress_test(
        self,
        scenario: StressScenario,
        num_paths: int = 10_000,
        time_steps: int = 48,
        confidence_level: float = 0.999,
        seed: Optional[int] = None,
    ) -> StressTestResult:
        """Executes high-dimensional Monte Carlo crisis path simulation."""
        t_start = time.perf_counter()
        rng = random.Random(seed if seed is not None else self.default_seed)
        
        dt_hours = scenario.duration_hours / time_steps
        v0 = scenario.initial_portfolio_value_usd
        v_floor = scenario.capital_reserve_floor_usd

        # Scenario shock impact calibration per hour
        hourly_energy_penalty = 0.0
        hourly_compute_penalty = 0.0
        hourly_network_penalty = 0.0

        if scenario.scenario_type == StressScenarioType.WINTER_STORM_URI_2:
            # LMP delta above baseline $40/MWh for committed MW
            excess_lmp = max(0.0, scenario.power_lmp_per_mwh - 40.0)
            curtailment_loss_mw = scenario.committed_forward_contracts_mw * (scenario.datacenter_curtailment_pct / 100.0)
            # Power cost surge + SLA delivery default penalty ($250/MWh-equivalent)
            hourly_energy_penalty = (excess_lmp * scenario.committed_forward_contracts_mw) * 0.12 + (curtailment_loss_mw * 250.0)
            hourly_compute_penalty = (scenario.portfolio_gpu_capacity * (scenario.retail_preemption_rate * 0.05)) * 4.5
        elif scenario.scenario_type == StressScenarioType.FRONTIER_MODEL_LAUNCH:
            # 85% preemption on short spot inventory requires emergency buy-in at forward+600bps
            hourly_compute_penalty = (scenario.gpu_supply_withdrawal_count * 0.02) * (scenario.forward_basis_shock_bps * 0.05)
            hourly_network_penalty = scenario.regional_arbitrage_divergence_bps * 2.5
        elif scenario.scenario_type == StressScenarioType.SUBSEA_TERABIT_PARTITION:
            # Latency spike causes missed quotes, adverse selection, packet loss drops execution legs
            hourly_network_penalty = (scenario.network_latency_spike_ms * 8.0) + (scenario.packet_loss_pct * 1500.0) + (scenario.regional_arbitrage_divergence_bps * 3.5)
        else:
            # General parametric shock
            hourly_energy_penalty = scenario.power_lmp_per_mwh * 1.5
            hourly_compute_penalty = scenario.forward_basis_shock_bps * 10.0
            hourly_network_penalty = scenario.network_latency_spike_ms * 10.0

        total_hourly_drag = (hourly_energy_penalty + hourly_compute_penalty + hourly_network_penalty)
        vol_per_step = math.sqrt(dt_hours) * (v0 * 0.008)

        paths_equity: List[List[float]] = []
        path_losses: List[float] = []
        path_max_drawdowns: List[float] = []
        path_drawdown_dollars: List[float] = []
        path_recovery_times: List[float] = []
        path_survivals: List[bool] = []

        # Fast simulation loop
        for p_idx in range(num_paths):
            current_v = v0
            series = [current_v]
            peak = current_v
            mdd = 0.0
            mdd_dollars = 0.0
            survived = True
            trough_step = 0
            trough_val = current_v
            recovered = True
            recovery_steps = 0

            for step in range(1, time_steps + 1):
                # Crisis profile: peak severity during middle 60% of crisis, initial onset spike
                progress = step / time_steps
                severity = math.sin(math.pi * progress) ** 0.8
                step_drag = total_hourly_drag * dt_hours * (0.4 + 1.2 * severity)
                
                # Gaussian diffusion innovation
                z = rng.gauss(0.0, 1.0)
                innovation = z * vol_per_step

                # Injected tail jump at step 2 (contingency arrival)
                contingency_jump = 0.0
                if step == 2:
                    contingency_jump = total_hourly_drag * dt_hours * 2.8

                current_v = current_v - step_drag - contingency_jump + innovation
                if current_v < 0.0:
                    current_v = 0.0

                series.append(current_v)

                if current_v > peak:
                    peak = current_v
                    if trough_val < peak and trough_step > 0 and recovery_steps == 0:
                        recovery_steps = step - trough_step
                else:
                    dd_dollars = peak - current_v
                    dd_pct = dd_dollars / peak if peak > 0.0 else 1.0
                    if dd_pct > mdd:
                        mdd = dd_pct
                        mdd_dollars = dd_dollars
                        trough_step = step
                        trough_val = current_v

                if current_v < v_floor:
                    survived = False

            if peak > trough_val and recovery_steps == 0:
                recovered = False
                recovery_hours = scenario.duration_hours
            else:
                recovery_hours = recovery_steps * dt_hours

            paths_equity.append(series)
            path_loss = v0 - current_v
            path_losses.append(path_loss)
            path_max_drawdowns.append(mdd)
            path_drawdown_dollars.append(mdd_dollars)
            path_recovery_times.append(recovery_hours)
            path_survivals.append(survived)

        # Statistical calculations
        path_losses.sort()
        n = len(path_losses)
        
        # Historical VaR at confidence_level
        var_index = min(int(confidence_level * n), n - 1)
        hist_var = max(0.0, path_losses[var_index])

        # Historical CVaR (Expected Shortfall)
        tail_losses = path_losses[var_index:]
        hist_cvar = sum(tail_losses) / len(tail_losses) if tail_losses else hist_var

        # Parametric VaR & CVaR
        mean_loss = sum(path_losses) / n
        var_loss = sum((x - mean_loss) ** 2 for x in path_losses) / (n - 1 if n > 1 else 1)
        std_loss = math.sqrt(var_loss)
        z_tail = _rational_norm_inv(confidence_level)
        param_var = max(0.0, mean_loss + z_tail * std_loss)
        
        # Standard normal PDF for CVaR
        phi_z = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z_tail * z_tail)
        alpha_tail = 1.0 - confidence_level
        param_cvar = max(param_var, mean_loss + std_loss * (phi_z / alpha_tail))

        # Drawdown & Solvency stats
        max_dd_pct = max(path_max_drawdowns) * 100.0
        max_dd_usd = max(path_drawdown_dollars)
        avg_recovery = sum(path_recovery_times) / n
        survival_prob = (sum(1 for s in path_survivals if s) / n) * 100.0

        # Percentile slices of terminal equity
        terminal_equities = sorted([p[-1] for p in paths_equity])
        p01 = terminal_equities[int(0.01 * n)]
        p05 = terminal_equities[int(0.05 * n)]
        p50 = terminal_equities[int(0.50 * n)]
        p95 = terminal_equities[min(int(0.95 * n), n - 1)]
        p99 = terminal_equities[min(int(0.99 * n), n - 1)]

        worst_loss = path_losses[-1]
        median_pnl = v0 - p50

        # Construct time-stepped fan chart ribbons for visualizer
        fan_chart_percentiles: Dict[str, List[float]] = {
            "p01": [],
            "p05": [],
            "p25": [],
            "p50": [],
            "p75": [],
            "p95": [],
            "p99": [],
        }

        for step in range(time_steps + 1):
            step_values = sorted([paths_equity[p][step] for p in range(n)])
            fan_chart_percentiles["p01"].append(step_values[int(0.01 * n)])
            fan_chart_percentiles["p05"].append(step_values[int(0.05 * n)])
            fan_chart_percentiles["p25"].append(step_values[int(0.25 * n)])
            fan_chart_percentiles["p50"].append(step_values[int(0.50 * n)])
            fan_chart_percentiles["p75"].append(step_values[int(0.75 * n)])
            fan_chart_percentiles["p95"].append(step_values[min(int(0.95 * n), n - 1)])
            fan_chart_percentiles["p99"].append(step_values[min(int(0.99 * n), n - 1)])

        # Pick 5 representative paths for visualization overlays
        sample_indices = [0, int(0.25 * n), int(0.50 * n), int(0.75 * n), min(int(0.99 * n), n - 1)]
        sample_paths = [paths_equity[idx] for idx in sample_indices]

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return StressTestResult(
            scenario=scenario,
            num_paths=num_paths,
            confidence_level=confidence_level,
            parametric_var_999=param_var,
            historical_var_999=hist_var,
            cvar_999=hist_cvar,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_usd=max_dd_usd,
            avg_recovery_time_hours=avg_recovery,
            system_survival_probability=survival_prob,
            worst_case_loss_usd=worst_loss,
            median_pnl_usd=-median_pnl,
            p01_equity_usd=p01,
            p05_equity_usd=p05,
            p50_equity_usd=p50,
            p95_equity_usd=p95,
            p99_equity_usd=p99,
            execution_time_ms=t_elapsed_ms,
            fan_chart_percentiles=fan_chart_percentiles,
            sample_trajectories=sample_paths,
        )
