"""Comprehensive Unit Test Suite for Stochastic Control & Smart Order Routing Modules.

Verifies:
1. Hamilton-Jacobi-Bellman (HJB) battery storage solver invariants:
   - Strict adherence to state of charge bounds E in [0, E_max].
   - Inverter power limits u_batt in [-P_chg_max, P_dis_max].
   - Non-linear DoD degradation penalty strictly eliminating unnecessary micro-cycling.
   - Real-time policy evaluation latency < 10 us.
   - Dynamic programming backward induction value function monotonicity.
2. Multi-Venue Smart Order Router (SOR) and Liquidity Knapsack:
   - Exact volume conservation sum v_k* == V_target.
   - Economic risk-adjusted volume shift from spot brokers to Tier-1 Hyperscalers as preemption penalty increases.
   - Venue capacity box constraints 0 <= v_k <= C_k and latency filtering.
   - Microsecond order routing execution speed (< 5 us).
   - Node quantization and discrete budget knapsack allocation.

Target execution time: < 5ms total using standard unittest.
"""

from __future__ import annotations

import math
import time
import unittest
from typing import Dict, List

from apex_compute_futures.futures.hjb_storage_solver import (
    BatteryOperatingMode,
    ComputeThrottlingMode,
    HJBBatteryConfig,
    HJBPolicyAction,
    HJBState,
    HJBStorageSolver,
)
from apex_compute_futures.futures.smart_order_router import (
    ComputeVenue,
    LiquidityKnapsackSolver,
    RoutingSolution,
    SmartOrderRouter,
    VenueTier,
)


class TestHJBStorageSolver(unittest.TestCase):
    """Test suite for HJB Dynamic Programming Solver and real-time policy evaluator."""

    def setUp(self) -> None:
        self.config = HJBBatteryConfig(
            e_max_mwh=50.0,
            e_min_mwh=0.0,
            p_chg_max_mw=25.0,
            p_dis_max_mw=25.0,
            eta_chg=0.95,
            eta_dis=0.95,
            alpha_deg=25.0,
            beta_deg=2.0,
            base_deg_usd_per_mwh=4.0,
            gpu_cluster_p_max_mw=40.0,
            gpu_cluster_p_idle_mw=10.0,
            compute_revenue_rate_usd_per_hr=6000.0,
            sla_penalty_lambda=5000.0,
            nominal_freq_hz=60.0,
            freq_deadband_hz=0.036,
            ancillary_price_usd_per_mwh=3500.0,
            dt_hours=1.0 / 60.0,
        )
        self.solver = HJBStorageSolver(self.config)

    def test_battery_physical_bounds_empty(self) -> None:
        """Verifies that an empty battery (E = 0) NEVER discharges, even during extreme price spikes."""
        state_empty = HJBState(
            energy_mwh=0.0,
            price_usd_per_mwh=1000.0,  # Extreme price spike ($1000/MWh)
            freq_deviation_hz=0.0,
        )
        action = self.solver.evaluate_realtime_policy(state_empty, shadow_price_energy=50.0)

        # Invariant: u_batt <= 0.0 (cannot discharge empty battery)
        self.assertLessEqual(action.u_batt_mw, 0.0)
        self.assertNotEqual(action.operating_mode, BatteryOperatingMode.DISCHARGING)
        # Energy must remain >= 0
        self.assertGreaterEqual(action.energy_next_mwh, 0.0)

    def test_battery_physical_bounds_full(self) -> None:
        """Verifies that a full battery (E = E_max) NEVER charges, even during negative prices."""
        state_full = HJBState(
            energy_mwh=50.0,
            price_usd_per_mwh=-50.0,  # Extreme negative price (grid pays to take power)
            freq_deviation_hz=0.0,
        )
        action = self.solver.evaluate_realtime_policy(state_full, shadow_price_energy=50.0)

        # Invariant: u_batt >= 0.0 (cannot charge full battery)
        self.assertGreaterEqual(action.u_batt_mw, 0.0)
        self.assertNotEqual(action.operating_mode, BatteryOperatingMode.CHARGING)
        # Energy must not exceed E_max
        self.assertLessEqual(action.energy_next_mwh, self.config.e_max_mwh)

    def test_battery_charge_discharge_power_limits(self) -> None:
        """Verifies that power dispatch strictly respects inverter limits [-P_chg_max, P_dis_max]."""
        for price in [-100.0, 10.0, 45.0, 80.0, 300.0, 1500.0]:
            for energy in [5.0, 25.0, 45.0]:
                state = HJBState(energy_mwh=energy, price_usd_per_mwh=price, freq_deviation_hz=0.0)
                action = self.solver.evaluate_realtime_policy(state, shadow_price_energy=50.0)

                self.assertGreaterEqual(action.u_batt_mw, -self.config.p_chg_max_mw)
                self.assertLessEqual(action.u_batt_mw, self.config.p_dis_max_mw)
                self.assertGreaterEqual(action.energy_next_mwh, 0.0)
                self.assertLessEqual(action.energy_next_mwh, self.config.e_max_mwh)

    def test_battery_degradation_prevents_unnecessary_micro_cycling(self) -> None:
        """Verifies that non-linear degradation creates an economic deadband eliminating micro-cycling.

        When the price spread is smaller than the marginal degradation cost, battery must stay IDLE.
        Without degradation, the battery would cycle for even tiny sub-dollar margins.
        """
        # Nominal baseline shadow price = $50.0/MWh
        # Marginal degradation cost at 50% SoC is base_deg + alpha * beta * (0.5)^1 / 50 = 4 + 25*2*0.5/50 = 4.5 $/MWh
        shadow_p = 50.0

        # Case A: Price spread is tiny ($0.50 above shadow price) -> Discharging spread = 50.5 - 50/0.95 - 4.5 < 0
        state_tiny = HJBState(energy_mwh=25.0, price_usd_per_mwh=50.50, freq_deviation_hz=0.0)
        action_with_deg = self.solver.evaluate_realtime_policy(state_tiny, shadow_price_energy=shadow_p)

        self.assertEqual(action_with_deg.operating_mode, BatteryOperatingMode.IDLE)
        self.assertEqual(action_with_deg.u_batt_mw, 0.0)
        self.assertEqual(action_with_deg.degradation_cost_rate, 0.0)

        # Case B: Zero-degradation baseline with 100% round-trip efficiency
        config_no_deg = HJBBatteryConfig(
            e_max_mwh=50.0,
            alpha_deg=0.0,
            beta_deg=1.0,
            base_deg_usd_per_mwh=0.0,
            eta_chg=1.0,
            eta_dis=1.0,
        )
        solver_no_deg = HJBStorageSolver(config_no_deg)
        action_no_deg = solver_no_deg.evaluate_realtime_policy(state_tiny, shadow_price_energy=shadow_p)

        # Without degradation and efficiency losses, even a $0.50 spread triggers discharge
        self.assertEqual(action_no_deg.operating_mode, BatteryOperatingMode.DISCHARGING)
        self.assertGreater(action_no_deg.u_batt_mw, 0.0)

    def test_non_linear_dod_degradation_scaling(self) -> None:
        """Verifies that marginal degradation increases monotonically with Depth of Discharge (DoD)."""
        # High energy (low DoD = 0.1) vs Low energy (high DoD = 0.9)
        c_deg_shallow = self.solver.marginal_degradation_cost(energy_mwh=45.0)  # DoD = 0.1
        c_deg_deep = self.solver.marginal_degradation_cost(energy_mwh=5.0)     # DoD = 0.9

        self.assertGreater(c_deg_deep, c_deg_shallow)
        # Ratio reflects beta_deg = 2.0: (0.9 / 0.1)^1 = 9x difference in variable degradation component
        diff_deep = c_deg_deep - self.config.base_deg_usd_per_mwh
        diff_shallow = c_deg_shallow - self.config.base_deg_usd_per_mwh
        self.assertAlmostEqual(diff_deep / diff_shallow, 9.0, places=3)

    def test_hjb_policy_evaluator_microsecond_speed(self) -> None:
        """Verifies that the real-time policy evaluator executes in < 10 microseconds."""
        state = HJBState(energy_mwh=25.0, price_usd_per_mwh=65.0, freq_deviation_hz=0.0)

        # Warmup
        for _ in range(20):
            self.solver.evaluate_realtime_policy(state)

        start = time.perf_counter_ns()
        n_iters = 200
        for _ in range(n_iters):
            self.solver.evaluate_realtime_policy(state)
        elapsed_us = (time.perf_counter_ns() - start) / (n_iters * 1000.0)

        self.assertLess(elapsed_us, 10.0, f"Policy evaluator took {elapsed_us:.2f} us, expected < 10 us")

    def test_dynamic_programming_grid_value_monotonicity(self) -> None:
        """Verifies that the backward induction value function is monotonic in stored energy."""
        price_curve = [35.0, 40.0, 120.0, 95.0, 30.0]
        v_grid = self.solver.solve_dynamic_programming_grid(price_curve, num_energy_grid_points=11)

        self.assertEqual(len(v_grid), 5)
        for t in range(len(price_curve)):
            # More stored energy must yield greater or equal terminal value
            for i in range(len(v_grid[t]) - 1):
                self.assertGreaterEqual(
                    v_grid[t][i + 1],
                    v_grid[t][i] - 1e-6,
                    f"Value function not monotonic at step {t}, index {i}"
                )

    def test_synthetic_inertia_underfrequency_response(self) -> None:
        """Verifies that grid under-frequency trips trigger synthetic inertia and ancillary revenue."""
        # Significant grid trip: Delta_f = -0.20 Hz (well outside 36 mHz deadband)
        state_trip = HJBState(energy_mwh=25.0, price_usd_per_mwh=60.0, freq_deviation_hz=-0.20)
        action = self.solver.evaluate_realtime_policy(state_trip, shadow_price_energy=50.0)

        self.assertGreater(action.ancillary_revenue_rate, 0.0)
        # Should be discharging or throttling to provide fast frequency injection
        is_stabilizing = (action.u_batt_mw > 0.0) or (action.u_clock < 1.0)
        self.assertTrue(is_stabilizing)


class TestSmartOrderRouter(unittest.TestCase):
    """Test suite for Multi-Venue Smart Order Router and Liquidity Knapsack."""

    def setUp(self) -> None:
        self.router = SmartOrderRouter()
        self.venues = self.router.venues

    def test_allocation_volume_conservation(self) -> None:
        """Verifies that sum v_k* == V_target holds strictly across diverse batch order volumes."""
        target_volumes = [500.0, 1500.0, 3500.0, 5000.0, 8500.0]

        for target in target_volumes:
            sol = self.router.route_order(target, preemption_penalty_usd=25.0)

            self.assertTrue(sol.is_fully_routed)
            self.assertAlmostEqual(sol.total_allocated_gpu_hrs, target, places=6)
            self.assertAlmostEqual(sum(a.allocated_volume_gpu_hrs for a in sol.allocations.values()), target, places=6)
            self.assertAlmostEqual(sol.unallocated_gpu_hrs, 0.0, places=6)

    def test_volume_shifts_to_tier1_as_preemption_penalty_increases(self) -> None:
        """Verifies that volume shifts monotonically from Tier-3 Spot Brokers to Tier-1 Hyperscalers

        as preemption penalty increases.
        """
        v_target = 5000.0

        # 1. Zero preemption penalty ($0/GPU-hr): Spot brokers should receive significant allocation
        sol_cheap = self.router.route_order(v_target, preemption_penalty_usd=0.0)
        tier3_cheap = sol_cheap.tier_distribution_gpu_hrs[VenueTier.TIER_3_SPOT_BROKER]
        tier1_cheap = sol_cheap.tier_distribution_gpu_hrs[VenueTier.TIER_1_HYPERSCALER]

        self.assertGreater(tier3_cheap, 0.0, "Tier-3 spot brokers should receive volume when preemption cost is $0")

        # 2. Extreme preemption penalty ($100/GPU-hr): High preemption penalty makes spot brokers unviable
        sol_risky = self.router.route_order(v_target, preemption_penalty_usd=100.0)
        tier3_risky = sol_risky.tier_distribution_gpu_hrs[VenueTier.TIER_3_SPOT_BROKER]
        tier1_risky = sol_risky.tier_distribution_gpu_hrs[VenueTier.TIER_1_HYPERSCALER]

        # Invariant: Tier-1 volume must increase, Tier-3 volume must decrease
        self.assertGreater(tier1_risky, tier1_cheap)
        self.assertLess(tier3_risky, tier3_cheap)
        self.assertAlmostEqual(tier3_risky, 0.0, places=3)

    def test_venue_capacity_bounds_and_latency_filtering(self) -> None:
        """Verifies that individual venue capacity bounds C_k are strictly observed and latency filtering works."""
        sol = self.router.route_order(12000.0, preemption_penalty_usd=15.0)

        for venue_id, alloc in sol.allocations.items():
            venue = next(v for v in self.venues if v.venue_id == venue_id)
            self.assertGreaterEqual(alloc.allocated_volume_gpu_hrs, 0.0)
            self.assertLessEqual(alloc.allocated_volume_gpu_hrs, venue.max_capacity_gpu_hrs + 1e-9)

        # Test latency filtering: max_latency_ms = 40ms excludes venues with total latency > 40ms
        sol_latency_filtered = self.router.route_order(3000.0, max_latency_ms=40.0)
        for venue_id, alloc in sol_latency_filtered.allocations.items():
            if alloc.allocated_volume_gpu_hrs > 0.0:
                venue = next(v for v in self.venues if v.venue_id == venue_id)
                self.assertLessEqual(venue.total_latency_ms, 40.0)

    def test_sor_microsecond_routing_speed(self) -> None:
        """Verifies that route_order_fast executes in < 5 microseconds for a 5,000 GPU-hour order."""
        # Warmup
        for _ in range(20):
            self.router.route_order_fast(5000.0, preemption_penalty_usd=25.0)

        start = time.perf_counter_ns()
        n_iters = 200
        for _ in range(n_iters):
            self.router.route_order_fast(5000.0, preemption_penalty_usd=25.0)
        elapsed_us = (time.perf_counter_ns() - start) / (n_iters * 1000.0)

        self.assertLess(elapsed_us, 5.0, f"SOR fast routing took {elapsed_us:.2f} us, expected < 5 us")

    def test_liquidity_knapsack_quantization(self) -> None:
        """Verifies discrete GPU node quantization preserves total requested node capacity."""
        sol = self.router.route_order(4000.0, preemption_penalty_usd=20.0)
        # 8-GPU nodes for 10 hours = 80 GPU-hrs/node -> 4000 / 80 = 50 nodes
        duration_hrs = 10.0
        node_allocation = LiquidityKnapsackSolver.quantize_nodes(sol, self.venues, duration_hours=duration_hrs)

        total_nodes = sum(node_allocation.values())
        self.assertEqual(total_nodes, 50)
        # All node counts must be integers >= 0
        for count in node_allocation.values():
            self.assertIsInstance(count, int)
            self.assertGreaterEqual(count, 0)

    def test_liquidity_budget_knapsack(self) -> None:
        """Verifies discrete 0-1 budget knapsack selects candidates without exceeding budget ceiling."""
        max_budget = 15000.0
        selected, val, cost = LiquidityKnapsackSolver.solve_budget_knapsack(
            self.venues, max_budget_usd=max_budget
        )

        self.assertLessEqual(cost, max_budget)
        self.assertGreater(val, 0.0)
        self.assertGreater(len(selected), 0)


if __name__ == "__main__":
    unittest.main()
