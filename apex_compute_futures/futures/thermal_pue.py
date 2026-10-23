"""First-Principles Thermal Dynamics, Liquid Cooling Loop, and Dynamic PUE Drift Model.

Models the microsecond thermodynamic envelope and real-time PUE of high-density
AI server racks (H100 / Blackwell SXM5) cooled via single-phase liquid loops:
1. Silicon Die Thermal Dynamics:
   C_th * d(T_die)/dt = P_chip(V, f_clk) - (T_die - T_coolant) / R_th
2. Liquid Cooling Loop Enthalpy Balance:
   Q = m_dot * C_p * (T_hot - T_cold)
3. Real-Time Dynamic PUE:
   PUE(t) = 1.0 + (P_chiller + P_pumps + P_fans) / P_compute
4. Microsecond Thermal Headroom Solver:
   Solves max safe overclock frequency f_clk before T_die > 85°C or coolant saturation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ThermalConfig:
    """Thermodynamic parameters for AI accelerator compute racks."""
    # Die & package thermal properties
    c_th: float = 120.0             # Silicon die + cold plate thermal mass (J/K)
    r_th: float = 0.035             # Junction-to-coolant thermal resistance (K/W)
    t_die_limit: float = 85.0       # Max safe junction temperature (°C)
    
    # Liquid cooling loop properties
    c_p: float = 4184.0             # Coolant specific heat capacity (J/(kg*K), water)
    mass_flow_rate: float = 2.0     # Coolant mass flow rate per rack (kg/s)
    t_cold: float = 25.0            # Coolant inlet/supply temperature (°C)
    t_coolant_limit: float = 60.0   # Coolant return saturation limit (°C)
    gpus_per_rack: int = 8          # Accelerator density per rack
    
    # Electrical DVFS and power model
    base_freq_ghz: float = 1.41     # Base core clock frequency (GHz)
    base_voltage_v: float = 0.85    # Base core operating voltage (V)
    base_power_watts: float = 700.0 # Base GPU TDP (W) at base freq/volt
    leakage_power_base: float = 80.0# Static leakage power (W) at 40°C reference
    v_min: float = 0.70             # Minimum operating voltage (V)
    v_max: float = 1.15             # Maximum overdrive voltage (V)
    freq_min_ghz: float = 0.60      # Minimum clock frequency (GHz)
    freq_max_ghz: float = 2.40      # Maximum overdrive frequency (GHz)
    dvfs_slope: float = 0.25        # Voltage-frequency scaling slope (V/GHz)
    leakage_temp_coeff: float = 0.015 # Exponential leakage coefficient per °C
    
    # Facility infrastructure constants
    pump_base_power_watts: float = 450.0   # Pumping power at nominal flow (W)
    fan_base_power_watts: float = 350.0    # Radiator/CDU fan power at nominal heat (W)
    nominal_heat_kw: float = 6.0           # Nominal rack heat load (kW)
    ambient_temp_c: float = 30.0           # Outside ambient dry-bulb temperature (°C)
    chiller_carnot_eff: float = 0.50       # Realistic chiller Carnot efficiency fraction
    chiller_approach_temp: float = 5.0     # Heat exchanger approach temp delta (°C)


@dataclass(frozen=True)
class PUEBreakdown:
    """Detailed breakdown of instantaneous datacenter cooling overhead and PUE."""
    pue: float
    p_compute_watts: float
    p_cooling_overhead_watts: float
    p_chiller_watts: float
    p_pumps_watts: float
    p_fans_watts: float
    cop: float
    economizer_active: bool


@dataclass(frozen=True)
class ThermalHeadroom:
    """Microsecond thermal headroom solver output."""
    max_safe_frequency_ghz: float
    max_safe_power_per_gpu_watts: float
    steady_state_die_temp_c: float
    coolant_return_temp_c: float
    thermal_headroom_c: float
    limiting_constraint: str         # "die_temperature", "coolant_saturation", or "dvfs_ceiling"
    pue_at_headroom: float


class ThermalPUEModel:
    """Physical thermodynamics and dynamic PUE engine for AI data center infrastructure."""

    def __init__(self, config: Optional[ThermalConfig] = None) -> None:
        self.config = config or ThermalConfig()
        # Thermal time constant tau = R_th * C_th (seconds)
        self.tau_th = self.config.r_th * self.config.c_th

    def compute_voltage_for_frequency(self, freq_ghz: float) -> float:
        """Computes operating voltage via linear DVFS curve clamped to [v_min, v_max]."""
        v = self.config.base_voltage_v + self.config.dvfs_slope * (freq_ghz - self.config.base_freq_ghz)
        return min(self.config.v_max, max(self.config.v_min, v))

    def compute_chip_power(
        self,
        freq_ghz: float,
        t_die_c: float,
        voltage_v: Optional[float] = None
    ) -> float:
        """Calculates total GPU power dissipation: P_chip = P_dynamic(V, f) + P_leakage(V, T_die)."""
        cfg = self.config
        v = voltage_v if voltage_v is not None else self.compute_voltage_for_frequency(freq_ghz)
        
        # Dynamic power: P_dyn = C_eff * V^2 * f
        dyn_base = cfg.base_power_watts - cfg.leakage_power_base
        p_dyn = dyn_base * ((v / cfg.base_voltage_v) ** 2) * (freq_ghz / cfg.base_freq_ghz)
        
        # Static leakage power: P_leak = V * I_leak * exp(beta * (T_die - T_ref))
        temp_delta = t_die_c - 40.0
        p_leak = cfg.leakage_power_base * (v / cfg.base_voltage_v) * math.exp(cfg.leakage_temp_coeff * temp_delta)
        
        return p_dyn + p_leak

    def step_die_temperature(
        self,
        t_die_current_c: float,
        power_watts: float,
        dt_seconds: float,
        t_coolant_c: Optional[float] = None
    ) -> float:
        """Solves C_th * dT/dt = P_chip - (T_die - T_coolant) / R_th over dt using exact exponential integration.
        
        T_die(t + dt) = T_steady + (T_die(t) - T_steady) * exp(-dt / tau_th)
        where T_steady = T_coolant + power_watts * R_th.
        Unconditionally stable for any dt > 0.
        """
        cfg = self.config
        t_cool = cfg.t_cold if t_coolant_c is None else t_coolant_c
        t_steady = t_cool + power_watts * cfg.r_th
        
        if dt_seconds <= 0.0:
            return t_die_current_c
            
        decay = math.exp(-dt_seconds / self.tau_th)
        return t_steady + (t_die_current_c - t_steady) * decay

    def enthalpy_balance(
        self,
        heat_load_watts: float,
        mass_flow_rate_kg_s: Optional[float] = None,
        t_cold_c: Optional[float] = None
    ) -> Tuple[float, float]:
        """Calculates liquid cooling loop enthalpy balance:
        Q = m_dot * C_p * (T_hot - T_cold)
        
        Returns:
            Tuple[T_hot_c, delta_t_c]
        """
        cfg = self.config
        m_dot = mass_flow_rate_kg_s if mass_flow_rate_kg_s is not None else cfg.mass_flow_rate
        t_inlet = t_cold_c if t_cold_c is not None else cfg.t_cold
        
        if m_dot <= 0.0:
            raise ValueError("Coolant mass flow rate must be positive.")
            
        delta_t = heat_load_watts / (m_dot * cfg.c_p)
        t_hot = t_inlet + delta_t
        return t_hot, delta_t

    def compute_required_mass_flow(
        self,
        heat_load_watts: float,
        max_delta_t_c: float
    ) -> float:
        """Calculates minimum coolant mass flow rate required to maintain loop delta_T <= max_delta_t_c."""
        if max_delta_t_c <= 0.0:
            raise ValueError("max_delta_t_c must be strictly positive.")
        return heat_load_watts / (self.config.c_p * max_delta_t_c)

    def calculate_pue(
        self,
        p_compute_rack_watts: float,
        ambient_temp_c: Optional[float] = None,
        mass_flow_rate_kg_s: Optional[float] = None
    ) -> PUEBreakdown:
        """Calculates real-time PUE and cooling component power breakdown.
        
        PUE(t) = 1.0 + (P_chiller + P_pumps + P_fans) / P_compute
        
        Models:
        - P_pumps: Scales with (m_dot / m_dot_base)^3 via pump affinity laws.
        - P_fans: Scales quadratically with normalized heat rejection load.
        - P_chiller: Carnot-based COP with water-side economizer bypass threshold.
        """
        cfg = self.config
        t_amb = ambient_temp_c if ambient_temp_c is not None else cfg.ambient_temp_c
        m_dot = mass_flow_rate_kg_s if mass_flow_rate_kg_s is not None else cfg.mass_flow_rate
        
        p_comp = max(100.0, p_compute_rack_watts)
        total_heat_watts = p_comp  # Thermodynamic conservation: 100% IT electricity turns into heat
        
        # 1. Pumping Power (Affinity Law: P_pump proportional to flow^3)
        flow_ratio = m_dot / max(0.1, cfg.mass_flow_rate)
        p_pumps = cfg.pump_base_power_watts * (flow_ratio ** 3)
        
        # 2. Radiator / CDU Fan Power (P_fans proportional to (Q / Q_nominal)^2)
        heat_ratio = (total_heat_watts / 1000.0) / max(0.5, cfg.nominal_heat_kw)
        p_fans = cfg.fan_base_power_watts * max(0.05, heat_ratio ** 2)
        
        # 3. Chiller Power & Economizer Mode
        # Economizer triggers if outside ambient is significantly cooler than supply water
        economizer_threshold = cfg.t_cold - 5.0
        if t_amb <= economizer_threshold:
            # Free cooling / economizer mode active; compressors uncoupled
            economizer_active = True
            p_chiller = 0.0
            cop = float("inf")
        else:
            economizer_active = False
            t_lift = max(2.0, (t_amb - cfg.t_cold) + cfg.chiller_approach_temp)
            carnot_ideal_cop = (cfg.t_cold + 273.15) / t_lift
            cop = max(1.8, cfg.chiller_carnot_eff * carnot_ideal_cop)
            p_chiller = total_heat_watts / cop
            
        p_overhead = p_chiller + p_pumps + p_fans
        pue = 1.0 + (p_overhead / p_comp)
        
        return PUEBreakdown(
            pue=round(pue, 4),
            p_compute_watts=round(p_comp, 2),
            p_cooling_overhead_watts=round(p_overhead, 2),
            p_chiller_watts=round(p_chiller, 2),
            p_pumps_watts=round(p_pumps, 2),
            p_fans_watts=round(p_fans, 2),
            cop=round(cop, 3) if cop != float("inf") else float("inf"),
            economizer_active=economizer_active
        )

    def solve_thermal_headroom(
        self,
        t_die_current_c: float,
        dt_burst_seconds: Optional[float] = None,
        ambient_temp_c: Optional[float] = None
    ) -> ThermalHeadroom:
        """Solves microsecond thermal headroom to find maximum safe operating frequency.
        
        Constraints:
        1. Silicon die junction temp T_die <= cfg.t_die_limit (default 85.0°C).
        2. Coolant return temp T_hot <= cfg.t_coolant_limit (default 60.0°C).
        
        If dt_burst_seconds is provided, solves transient headroom over a burst interval;
        otherwise solves continuous steady-state headroom.
        
        Uses fast bounded bisection achieving < 0.001 GHz precision in < 15 iterations.
        """
        cfg = self.config
        t_amb = ambient_temp_c if ambient_temp_c is not None else cfg.ambient_temp_c
        
        # Search interval [freq_min, freq_max]
        low = cfg.freq_min_ghz
        high = cfg.freq_max_ghz
        
        def evaluate_candidate(freq: float) -> Tuple[bool, float, float, float]:
            # Initial guess for die temp to compute power
            t_eval = max(t_die_current_c, cfg.t_cold + 20.0)
            # 2 Picard iterations for self-consistent leakage power vs die temperature
            for _ in range(2):
                p_gpu = self.compute_chip_power(freq, t_eval)
                t_eval = cfg.t_cold + p_gpu * cfg.r_th
                
            p_gpu = self.compute_chip_power(freq, t_eval)
            p_rack = p_gpu * cfg.gpus_per_rack
            
            # Coolant return temperature
            t_hot, _ = self.enthalpy_balance(p_rack)
            
            # Die temperature at end of interval
            if dt_burst_seconds is not None and dt_burst_seconds > 0.0:
                t_die = self.step_die_temperature(t_die_current_c, p_gpu, dt_burst_seconds, cfg.t_cold)
            else:
                t_die = cfg.t_cold + p_gpu * cfg.r_th
                
            safe = (t_die <= cfg.t_die_limit) and (t_hot <= cfg.t_coolant_limit)
            return safe, t_die, t_hot, p_gpu

        # Check bounds
        safe_high, t_die_h, t_hot_h, p_h = evaluate_candidate(high)
        if safe_high:
            # Can run at max frequency without exceeding limits
            pue_res = self.calculate_pue(p_h * cfg.gpus_per_rack, ambient_temp_c=t_amb)
            return ThermalHeadroom(
                max_safe_frequency_ghz=high,
                max_safe_power_per_gpu_watts=round(p_h, 2),
                steady_state_die_temp_c=round(t_die_h, 2),
                coolant_return_temp_c=round(t_hot_h, 2),
                thermal_headroom_c=round(cfg.t_die_limit - t_die_h, 2),
                limiting_constraint="dvfs_ceiling",
                pue_at_headroom=pue_res.pue
            )
            
        safe_low, t_die_l, t_hot_l, p_l = evaluate_candidate(low)
        if not safe_low:
            # Even minimum frequency exceeds limits (e.g. ambient/coolant already boiling)
            limit_cause = "coolant_saturation" if t_hot_l > cfg.t_coolant_limit else "die_temperature"
            pue_res = self.calculate_pue(p_l * cfg.gpus_per_rack, ambient_temp_c=t_amb)
            return ThermalHeadroom(
                max_safe_frequency_ghz=low,
                max_safe_power_per_gpu_watts=round(p_l, 2),
                steady_state_die_temp_c=round(t_die_l, 2),
                coolant_return_temp_c=round(t_hot_l, 2),
                thermal_headroom_c=round(cfg.t_die_limit - t_die_l, 2),
                limiting_constraint=limit_cause,
                pue_at_headroom=pue_res.pue
            )

        # Microsecond bisection
        for _ in range(16):
            mid = 0.5 * (low + high)
            safe_mid, _, _, _ = evaluate_candidate(mid)
            if safe_mid:
                low = mid
            else:
                high = mid
                
        # Conservative floor to guarantee strict compliance with limits
        best_freq = math.floor(low * 1000.0) / 1000.0
        safe_f, t_die_f, t_hot_f, p_f = evaluate_candidate(best_freq)
        while not safe_f and best_freq > cfg.freq_min_ghz:
            best_freq = round(best_freq - 0.001, 3)
            safe_f, t_die_f, t_hot_f, p_f = evaluate_candidate(best_freq)
        
        limit_reason = "coolant_saturation" if t_hot_f >= cfg.t_coolant_limit - 0.2 else "die_temperature"
        pue_res = self.calculate_pue(p_f * cfg.gpus_per_rack, ambient_temp_c=t_amb)
        
        return ThermalHeadroom(
            max_safe_frequency_ghz=best_freq,
            max_safe_power_per_gpu_watts=round(p_f, 2),
            steady_state_die_temp_c=round(t_die_f, 2),
            coolant_return_temp_c=round(t_hot_f, 2),
            thermal_headroom_c=round(cfg.t_die_limit - t_die_f, 2),
            limiting_constraint=limit_reason,
            pue_at_headroom=pue_res.pue
        )
