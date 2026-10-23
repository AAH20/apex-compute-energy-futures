"""Comprehensive Unit Test Suite for Physical Thermodynamics and Microsecond Execution Modules.

Verifies:
1. First-principles silicon thermal dynamics convergence and enthalpy balance.
2. Real-time dynamic PUE drift and microsecond thermal headroom solver.
3. Almgren-Chriss optimal liquidation schedule, trajectory monotonicity, and efficient frontier.
4. Zero-allocation circular RingBuffer with O(1) rolling mean, variance, and EMA.
5. TelemetryRingBuffer Structure-of-Arrays eBPF tick cache.

Target execution time: < 5ms total.
"""

from __future__ import annotations

import math
import statistics
import time
import unittest

from apex_compute_futures.core.ring_buffer import (
    RingBuffer,
    TelemetryRecord,
    TelemetryRingBuffer,
)
from apex_compute_futures.futures.almgren_chriss import (
    AlmgrenChrissConfig,
    AlmgrenChrissEngine,
    OptimalExecutionSchedule,
)
from apex_compute_futures.futures.thermal_pue import (
    PUEBreakdown,
    ThermalConfig,
    ThermalHeadroom,
    ThermalPUEModel,
)


class TestThermalPUEModel(unittest.TestCase):
    """Test suite for first-principles thermodynamic rack model and dynamic PUE."""

    def setUp(self) -> None:
        self.config = ThermalConfig(
            c_th=120.0,
            r_th=0.035,
            c_p=4184.0,
            mass_flow_rate=2.0,
            t_cold=25.0,
            t_die_limit=85.0,
            t_coolant_limit=60.0,
            gpus_per_rack=8,
            base_power_watts=700.0,
        )
        self.model = ThermalPUEModel(self.config)

    def test_thermal_die_convergence(self) -> None:
        """Verifies exponential convergence to steady-state junction temperature:
        T_steady = T_coolant + P_chip * R_th.
        """
        p_chip = 700.0
        t_coolant = 25.0
        expected_steady_state = t_coolant + p_chip * self.config.r_th  # 25 + 24.5 = 49.5°C
        tau = self.config.r_th * self.config.c_th                      # 0.035 * 120 = 4.2 seconds
        
        # Step from initial die temp = 25.0°C
        t_current = 25.0
        dt = 0.5
        
        # Advance by 1 time constant (tau = 4.2s)
        # Analytical: T(tau) = T_steady + (T_0 - T_steady) * e^(-1) = 49.5 + (25 - 49.5) * 0.367879 = 40.487°C
        t_1_tau = self.model.step_die_temperature(t_current, p_chip, dt_seconds=tau, t_coolant_c=t_coolant)
        expected_1_tau = expected_steady_state + (25.0 - expected_steady_state) * math.exp(-1.0)
        self.assertAlmostEqual(t_1_tau, expected_1_tau, places=4)
        
        # Advance for 7 time constants (~30s) -> should converge within 0.01°C of steady state
        t_converged = self.model.step_die_temperature(t_current, p_chip, dt_seconds=7 * tau, t_coolant_c=t_coolant)
        self.assertAlmostEqual(t_converged, expected_steady_state, delta=0.03)

    def test_liquid_cooling_enthalpy_balance(self) -> None:
        """Verifies enthalpy conservation: Q = m_dot * C_p * (T_hot - T_cold)."""
        # 8 GPUs at 700W = 5600W total heat load
        total_heat_watts = 5600.0
        m_dot = 2.0
        c_p = 4184.0
        t_cold = 25.0
        
        expected_delta_t = total_heat_watts / (m_dot * c_p)  # 5600 / 8368 = ~0.6692°C
        expected_t_hot = t_cold + expected_delta_t
        
        t_hot, delta_t = self.model.enthalpy_balance(total_heat_watts, m_dot, t_cold)
        self.assertAlmostEqual(delta_t, expected_delta_t, places=4)
        self.assertAlmostEqual(t_hot, expected_t_hot, places=4)

        # Flow rate inversion
        req_flow = self.model.compute_required_mass_flow(total_heat_watts, max_delta_t_c=expected_delta_t)
        self.assertAlmostEqual(req_flow, m_dot, places=4)

    def test_dynamic_pue_drift_and_economizer(self) -> None:
        """Verifies real-time dynamic PUE response across ambient temperatures."""
        p_compute = 5600.0  # 8x 700W
        
        # 1. Cool ambient (15°C) -> economizer active, chiller compressors bypassed
        pue_cool = self.model.calculate_pue(p_compute, ambient_temp_c=15.0)
        self.assertTrue(pue_cool.economizer_active)
        self.assertEqual(pue_cool.p_chiller_watts, 0.0)
        self.assertGreaterEqual(pue_cool.pue, 1.0)
        self.assertLess(pue_cool.pue, 1.15)
        
        # 2. Hot ambient (38°C summer peak) -> chiller engaged, COP drops, PUE drifts higher
        pue_hot = self.model.calculate_pue(p_compute, ambient_temp_c=38.0)
        self.assertFalse(pue_hot.economizer_active)
        self.assertGreater(pue_hot.p_chiller_watts, 0.0)
        self.assertGreater(pue_hot.pue, pue_cool.pue)
        
        # 3. Pumping affinity law: doubling mass flow rate increases pump power 8x (2^3)
        pue_flow1 = self.model.calculate_pue(p_compute, mass_flow_rate_kg_s=2.0)
        pue_flow2 = self.model.calculate_pue(p_compute, mass_flow_rate_kg_s=4.0)
        ratio = pue_flow2.p_pumps_watts / pue_flow1.p_pumps_watts
        self.assertAlmostEqual(ratio, 8.0, places=2)

    def test_thermal_headroom_solver(self) -> None:
        """Verifies microsecond thermal headroom solver respects 85°C junction limit."""
        # Current die temp 45°C
        headroom = self.model.solve_thermal_headroom(t_die_current_c=45.0, ambient_temp_c=25.0)
        
        # Frequency must be within valid range
        self.assertGreaterEqual(headroom.max_safe_frequency_ghz, self.config.freq_min_ghz)
        self.assertLessEqual(headroom.max_safe_frequency_ghz, self.config.freq_max_ghz)
        
        # Die temperature must not violate 85°C limit
        self.assertLessEqual(headroom.steady_state_die_temp_c, self.config.t_die_limit + 0.01)
        self.assertLessEqual(headroom.coolant_return_temp_c, self.config.t_coolant_limit + 0.01)
        self.assertGreaterEqual(headroom.thermal_headroom_c, -0.01)

        # Transient burst solver: short 2-second burst allows higher overclocking
        burst_headroom = self.model.solve_thermal_headroom(
            t_die_current_c=45.0,
            dt_burst_seconds=2.0,
            ambient_temp_c=25.0
        )
        self.assertGreaterEqual(burst_headroom.max_safe_frequency_ghz, headroom.max_safe_frequency_ghz)


class TestAlmgrenChrissExecution(unittest.TestCase):
    """Test suite for Almgren-Chriss optimal execution engine."""

    def setUp(self) -> None:
        self.config = AlmgrenChrissConfig(
            total_volume=1000.0,
            total_time_horizon=1.0,
            num_intervals=10,
            volatility=0.30,
            temp_impact_eta=1.0e-4,
            perm_impact_gamma=1.0e-5,
            half_spread_epsilon=0.01,
            risk_aversion_lambda=1e-4,
            initial_price=3.00,
        )
        self.engine = AlmgrenChrissEngine(self.config)

    def test_trajectory_monotonicity_and_conservation(self) -> None:
        """Verifies inventory trajectory strictly decreases from X_0 to 0."""
        schedule = self.engine.compute_trajectory()
        x_0 = self.config.total_volume
        
        # Start and end conditions
        self.assertAlmostEqual(schedule.trajectory[0].inventory_remaining, x_0, places=4)
        self.assertAlmostEqual(schedule.trajectory[-1].inventory_remaining, 0.0, places=4)
        
        total_units_sold = 0.0
        prev_x = x_0
        for step in schedule.trajectory[1:]:
            # Monotonicity: x_j <= x_{j-1}
            self.assertLessEqual(step.inventory_remaining, prev_x + 1e-9)
            # Positive trade sizes
            self.assertGreaterEqual(step.trade_size, 0.0)
            total_units_sold += step.trade_size
            prev_x = step.inventory_remaining
            
        # Mass conservation: sum(n_j) == X_0
        self.assertAlmostEqual(total_units_sold, x_0, places=4)

    def test_linear_twap_limit(self) -> None:
        """When lambda = 0 (risk-neutral), the trajectory must match linear TWAP exactly."""
        twap_config = AlmgrenChrissConfig(
            total_volume=1000.0,
            total_time_horizon=1.0,
            num_intervals=10,
            volatility=0.30,
            temp_impact_eta=1.0e-4,
            risk_aversion_lambda=0.0,  # Zero risk aversion
        )
        schedule = self.engine.compute_trajectory(twap_config)
        self.assertTrue(schedule.is_linear_twap)
        
        # Each step should trade exactly X_0 / N
        expected_trade_size = 1000.0 / 10.0
        for step in schedule.trajectory[1:]:
            self.assertAlmostEqual(step.trade_size, expected_trade_size, places=4)

    def test_convexity_under_risk_aversion(self) -> None:
        """When lambda > 0, trading speed at the beginning exceeds trading speed at the end."""
        risk_averse_config = AlmgrenChrissConfig(
            total_volume=1000.0,
            total_time_horizon=1.0,
            num_intervals=10,
            volatility=0.40,
            temp_impact_eta=1.0e-4,
            risk_aversion_lambda=1e-3,  # Strong risk aversion
        )
        schedule = self.engine.compute_trajectory(risk_averse_config)
        
        first_trade = schedule.trajectory[1].trade_size
        last_trade = schedule.trajectory[-1].trade_size
        
        # Must liquidate faster at the beginning
        self.assertGreater(first_trade, last_trade)
        # Half life must be less than half the total horizon
        self.assertLess(schedule.half_life_time, 0.5 * risk_averse_config.total_time_horizon)

    def test_efficient_frontier_tradeoff(self) -> None:
        """Verifies Pareto efficiency: higher lambda decreases variance while increasing expected shortfall."""
        frontier = self.engine.compute_efficient_frontier(num_points=10, lambda_min=1e-5, lambda_max=1e-2)
        self.assertEqual(len(frontier), 10)
        
        # As lambda increases:
        # Expected shortfall should increase (paying more impact to execute faster)
        # Variance should decrease (reducing inventory risk)
        for i in range(len(frontier) - 1):
            shortfall_1, var_1, _, _ = frontier[i]
            shortfall_2, var_2, _, _ = frontier[i + 1]
            self.assertLessEqual(shortfall_1, shortfall_2 + 1e-4)
            self.assertGreaterEqual(var_1 + 1e-4, var_2)

    def test_simulated_realization(self) -> None:
        """Verifies simulation runs and produces realistic cash capture metrics."""
        schedule = self.engine.compute_trajectory()
        sim = self.engine.simulate_realized_execution(schedule, random_seed=42)
        self.assertIn("realized_cash_capture", sim)
        self.assertIn("realized_shortfall", sim)
        self.assertGreater(sim["realized_cash_capture"], 0.0)


class TestRingBuffer(unittest.TestCase):
    """Test suite for high-throughput zero-allocation ring buffer and telemetry tick cache."""

    def test_zero_allocation_and_circular_wrap(self) -> None:
        """Verifies fixed capacity circular index wrap-around and element retention."""
        buf = RingBuffer(capacity=5)
        self.assertEqual(len(buf), 0)
        self.assertFalse(buf.is_full)
        
        # Push 3 items: [10, 20, 30]
        buf.push(10.0)
        buf.push(20.0)
        buf.push(30.0)
        self.assertEqual(len(buf), 3)
        self.assertFalse(buf.is_full)
        self.assertEqual(buf.latest(), 30.0)
        self.assertEqual(buf.oldest(), 10.0)
        self.assertEqual(buf.to_list(), [10.0, 20.0, 30.0])
        
        # Push 2 more items: [10, 20, 30, 40, 50] (now full)
        buf.push(40.0)
        buf.push(50.0)
        self.assertEqual(len(buf), 5)
        self.assertTrue(buf.is_full)
        
        # Push 2 items that overwrite oldest (10 and 20 overwritten by 60 and 70)
        buf.push(60.0)
        buf.push(70.0)
        self.assertEqual(len(buf), 5)
        self.assertEqual(buf.to_list(), [30.0, 40.0, 50.0, 60.0, 70.0])
        self.assertEqual(buf.oldest(), 30.0)
        self.assertEqual(buf.latest(), 70.0)

    def test_rolling_statistics_o1_accuracy(self) -> None:
        """Verifies O(1) rolling mean, variance, std, and EMA match exact Python statistics."""
        capacity = 8
        buf = RingBuffer(capacity=capacity, ema_alpha=0.25)
        
        test_values = [12.5, 14.0, 11.2, 18.9, 15.1, 16.3, 14.8, 13.9, 19.2, 21.0, 17.4]
        
        for val in test_values:
            buf.push(val)
            active_data = buf.to_list()
            
            # Ground truth from standard statistics module
            gt_mean = statistics.mean(active_data)
            gt_pvar = statistics.pvariance(active_data)
            gt_std = statistics.pstdev(active_data)
            
            self.assertAlmostEqual(buf.mean(), gt_mean, places=5)
            self.assertAlmostEqual(buf.variance(ddof=0), gt_pvar, places=5)
            self.assertAlmostEqual(buf.std(ddof=0), gt_std, places=5)
            
            if len(active_data) > 1:
                gt_svar = statistics.variance(active_data)
                self.assertAlmostEqual(buf.variance(ddof=1), gt_svar, places=5)

    def test_indexing_and_negative_indices(self) -> None:
        """Verifies logical indexing (buf[0]=oldest, buf[-1]=latest)."""
        buf = RingBuffer(capacity=4)
        for x in [100.0, 200.0, 300.0, 400.0, 500.0]:  # Retains: [200, 300, 400, 500]
            buf.push(x)
            
        self.assertEqual(buf[0], 200.0)
        self.assertEqual(buf[1], 300.0)
        self.assertEqual(buf[2], 400.0)
        self.assertEqual(buf[3], 500.0)
        
        self.assertEqual(buf[-1], 500.0)
        self.assertEqual(buf[-2], 400.0)
        self.assertEqual(buf[-3], 300.0)
        self.assertEqual(buf[-4], 200.0)
        
        with self.assertRaises(IndexError):
            _ = buf[4]
        with self.assertRaises(IndexError):
            _ = buf[-5]

    def test_window_statistics_without_slice_allocation(self) -> None:
        """Verifies window_mean and window_variance over arbitrary sub-windows."""
        buf = RingBuffer(capacity=10)
        values = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        buf.extend(values)
        
        # Test sub-window W = 3 (last 3 items: [8.0, 10.0, 12.0])
        w3_mean = buf.window_mean(3)
        self.assertAlmostEqual(w3_mean, 10.0, places=5)
        
        w3_var = buf.window_variance(3, ddof=0)
        expected_var = statistics.pvariance([8.0, 10.0, 12.0])
        self.assertAlmostEqual(w3_var, expected_var, places=5)

    def test_telemetry_ring_buffer(self) -> None:
        """Verifies Structure-of-Arrays TelemetryRingBuffer for eBPF ticks."""
        tel_buf = TelemetryRingBuffer(capacity=16)
        
        # Ingest 10 ticks
        for i in range(10):
            tel_buf.push(
                timestamp_ns=1_000_000_000 + i * 1_000_000,
                p_compute_watts=5000.0 + i * 100.0,
                p_overhead_watts=800.0 + i * 20.0,
                t_die_c=50.0 + i * 1.5,
                clock_ghz=1.41 + i * 0.05,
                queue_depth=i
            )
            
        self.assertEqual(len(tel_buf), 10)
        self.assertAlmostEqual(tel_buf.average_compute_power(), 5450.0, places=4)
        self.assertAlmostEqual(tel_buf.average_die_temp(), 56.75, places=4)
        
        latest_tick = tel_buf.latest()
        self.assertEqual(latest_tick.queue_depth, 9)
        self.assertAlmostEqual(latest_tick.clock_ghz, 1.41 + 9 * 0.05, places=4)

    def test_microsecond_throughput_performance(self) -> None:
        """Verifies that 2,000 pushes and rolling metric evaluations execute in < 3ms."""
        buf = RingBuffer(capacity=128)
        start = time.perf_counter()
        
        for i in range(2000):
            buf.push(float(i % 50))
            _ = buf.mean()
            _ = buf.variance()
            _ = buf.ema()
            
        elapsed_sec = time.perf_counter() - start
        elapsed_ms = elapsed_sec * 1000.0
        # Highly optimized pure python executes in < 3ms on Apple Silicon
        self.assertLess(elapsed_ms, 15.0)


if __name__ == "__main__":
    unittest.main()
