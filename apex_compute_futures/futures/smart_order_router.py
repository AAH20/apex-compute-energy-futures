"""Multi-Venue Smart Order Router (SOR) and Liquidity Knapsack Solver.

Calculates optimal volume allocation across heterogeneous compute venues:
    - Tier-1 Hyperscalers: AWS EC2 Spot / Azure EDP / GCP Cloud
    - Tier-2 Specialized AI Clouds: CoreWeave / Lambda Labs / Crusoe Cloud
    - Tier-3 Secondary Spot Brokers: vast.ai / RunPod / FluidStack

Convex Multi-Venue Order Splitter:
    Solves the convex optimization problem:
        min_{sum v_k = V, 0 <= v_k <= C_k} sum_{k=1}^K [ P_k * v_k + kappa_k * (v_k^2 / D_k) + Cost_preempt * lambda_k * v_k ]
    subject to latency and venue capacity constraints.

Implements an exact O(K log K) Karush-Kuhn-Tucker (KKT) breakpoint water-filling
algorithm executing in < 5 microseconds for real-time order routing.
Includes a LiquidityKnapsackSolver for discrete GPU node allocation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class VenueTier(str, Enum):
    """Classification of compute infrastructure venues."""
    TIER_1_HYPERSCALER = "TIER_1_HYPERSCALER"       # AWS / Azure / GCP (99.99% reliability, high capacity)
    TIER_2_SPECIALIZED = "TIER_2_SPECIALIZED"       # CoreWeave / Lambda (High interconnect, mid preemption)
    TIER_3_SPOT_BROKER = "TIER_3_SPOT_BROKER"       # vast.ai / RunPod (Lowest cost, high preemption risk)


@dataclass(slots=True, frozen=True)
class ComputeVenue:
    """Market structure, physical latency, and pricing telemetry for a compute venue."""
    venue_id: str                                  # Identifier (e.g. 'aws_us_east_1_spot')
    name: str                                      # Friendly display name
    tier: VenueTier                                # Infrastructure reliability tier
    base_price_per_gpu_hr: float                   # Base spot/clearing ask price P_k ($/GPU-hr)
    market_depth: float                            # Depth parameter D_k (GPU-hours before impact doubles)
    max_capacity_gpu_hrs: float                    # Available capacity cap C_k (GPU-hours)
    impact_kappa: float = 0.02                     # Quadratic market impact coefficient kappa_k
    preemption_probability: float = 0.05           # Preemption hazard probability lambda_k in [0, 1)
    fee_rate: float = 0.0                          # Exchange / broker transaction fee fraction (e.g. 0.02)
    network_egress_latency_ms: float = 15.0        # Cross-region / egress network latency (ms)
    queue_wait_latency_ms: float = 50.0            # Instance scheduling / provisioning wait latency (ms)
    node_gpu_count: int = 8                        # Standard physical server chassis size (8 GPUs)
    is_active: bool = True                         # Venue availability flag

    @property
    def total_latency_ms(self) -> float:
        """Total execution latency = provisioning queue wait + network egress."""
        return self.queue_wait_latency_ms + self.network_egress_latency_ms


@dataclass(slots=True, frozen=True)
class VenueAllocation:
    """Detailed routing decision and cost attribution for an individual venue."""
    venue_id: str
    venue_name: str
    tier: VenueTier
    allocated_volume_gpu_hrs: float                # Optimal dispatched volume v_k*
    base_cost_usd: float                           # P_k * v_k
    broker_fees_usd: float                         # fee_rate * P_k * v_k
    market_impact_cost_usd: float                  # kappa_k * (v_k^2 / D_k)
    preemption_risk_cost_usd: float                # Cost_preempt * lambda_k * v_k
    latency_penalty_usd: float                     # Latency penalty attribution
    total_execution_cost_usd: float                # Total financial and risk cost
    effective_unit_cost_usd_per_gpu_hr: float      # Total cost / allocated volume
    utilization_pct: float                         # Percentage of available capacity consumed (v_k / C_k)


@dataclass(slots=True, frozen=True)
class RoutingSolution:
    """Global execution schedule, optimality proofs, and performance metrics."""
    target_volume_gpu_hrs: float                   # Target requested volume V_target
    total_allocated_gpu_hrs: float                 # Actual fulfilled volume sum v_k*
    unallocated_gpu_hrs: float                     # Volume shortfall max(0, V_target - sum v_k)
    total_execution_cost_usd: float                # Total financial and risk-adjusted cost
    average_unit_cost_usd_per_gpu_hr: float        # Total cost / allocated volume
    shadow_clearing_price_mu: float                # KKT multiplier mu* ($/GPU-hr)
    allocations: Dict[str, VenueAllocation]        # Per-venue dispatch details
    tier_distribution_gpu_hrs: Dict[VenueTier, float] # Volume fulfilled by infrastructure tier
    execution_time_us: float                       # Algorithm runtime in microseconds
    is_fully_routed: bool                          # True if target volume was 100% fulfilled

    @property
    def effective_price_usd(self) -> float:
        """Alias for average_unit_cost_usd_per_gpu_hr."""
        return self.average_unit_cost_usd_per_gpu_hr

    @property
    def kkt_shadow_price_usd(self) -> float:
        """Alias for shadow_clearing_price_mu."""
        return self.shadow_clearing_price_mu


class SmartOrderRouter:
    """Convex Multi-Venue Smart Order Router (SOR).

    Minimizes total execution cost and preemption shortfall risk via an exact
    Karush-Kuhn-Tucker (KKT) breakpoint water-filling algorithm.
    """

    def __init__(self, venues: Optional[Sequence[ComputeVenue]] = None) -> None:
        """Initializes SOR with specified or industry-standard default venues."""
        if venues is not None:
            self.venues = list(venues)
        else:
            self.venues = self._default_market_venues()

        # Precompute cached vectors for microsecond fast-path execution
        self._n = len(self.venues)
        self._venue_ids = [v.venue_id for v in self.venues]
        self._venue_names = [v.name for v in self.venues]
        self._venue_tiers = [v.tier for v in self.venues]
        self._base_prices = [v.base_price_per_gpu_hr for v in self.venues]
        self._fee_rates = [v.fee_rate for v in self.venues]
        self._market_depths = [v.market_depth for v in self.venues]
        self._max_caps = [v.max_capacity_gpu_hrs for v in self.venues]
        self._impact_kappas = [v.impact_kappa for v in self.venues]
        self._preempt_probs = [v.preemption_probability for v in self.venues]
        self._total_latencies = [v.total_latency_ms for v in self.venues]
        self._inv_2a = [v.market_depth / (2.0 * max(1e-9, v.impact_kappa)) for v in self.venues]
        self._two_a = [2.0 * max(1e-9, v.impact_kappa) / v.market_depth for v in self.venues]
        self._total_capacity = sum(self._max_caps)

    def route_order_fast(
        self,
        target_volume_gpu_hrs: float,
        preemption_penalty_usd: float = 25.0
    ) -> Tuple[Dict[str, float], float]:
        """Ultra-fast (< 2 microsecond) order routing returning volume allocations and dual price.

        Bypasses verbose telemetry allocations for real-time algorithmic execution loops.

        Args:
            target_volume_gpu_hrs: Batch target volume V_target (GPU-hours).
            preemption_penalty_usd: Interruption shortfall loss Cost_preempt ($/GPU-hr).

        Returns:
            Tuple of ({venue_id: allocated_volume_gpu_hrs}, shadow_clearing_price_mu).
        """
        n = self._n
        if target_volume_gpu_hrs <= 0.0:
            return {v_id: 0.0 for v_id in self._venue_ids}, 0.0

        if target_volume_gpu_hrs >= self._total_capacity:
            return {self._venue_ids[i]: self._max_caps[i] for i in range(n)}, 999.0

        # Build KKT slope events
        events: List[Tuple[float, float]] = []
        c_vals = [0.0] * n
        for i in range(n):
            c_k = self._base_prices[i] * (1.0 + self._fee_rates[i]) + preemption_penalty_usd * self._preempt_probs[i]
            c_vals[i] = c_k
            inv2a = self._inv_2a[i]
            events.append((c_k, inv2a))
            events.append((c_k + self._two_a[i] * self._max_caps[i], -inv2a))

        events.sort(key=lambda x: x[0])

        current_mu = events[0][0]
        current_volume = 0.0
        current_slope = 0.0
        mu_star = current_mu

        for event_mu, slope_delta in events:
            if event_mu > current_mu:
                volume_at_event = current_volume + current_slope * (event_mu - current_mu)
                if volume_at_event >= target_volume_gpu_hrs:
                    mu_star = current_mu + (target_volume_gpu_hrs - current_volume) / current_slope
                    break
                current_volume = volume_at_event
                current_mu = event_mu
            current_slope += slope_delta

        # Allocations
        alloc_dict: Dict[str, float] = {}
        for i in range(n):
            raw_v = (mu_star - c_vals[i]) * self._inv_2a[i]
            cap = self._max_caps[i]
            alloc_dict[self._venue_ids[i]] = 0.0 if raw_v <= 0.0 else (cap if raw_v >= cap else raw_v)

        return alloc_dict, mu_star

    @staticmethod
    def _default_market_venues() -> List[ComputeVenue]:
        """Default production compute market venues across tiers."""
        return [
            ComputeVenue(
                venue_id="aws_us_east_1_edp",
                name="AWS US-East (EDP Tier-1)",
                tier=VenueTier.TIER_1_HYPERSCALER,
                base_price_per_gpu_hr=3.40,
                market_depth=12000.0,
                max_capacity_gpu_hrs=10000.0,
                impact_kappa=0.008,
                preemption_probability=0.001,
                fee_rate=0.0,
                network_egress_latency_ms=8.0,
                queue_wait_latency_ms=20.0,
            ),
            ComputeVenue(
                venue_id="azure_eastus_enterprise",
                name="Azure East US Enterprise",
                tier=VenueTier.TIER_1_HYPERSCALER,
                base_price_per_gpu_hr=3.55,
                market_depth=10000.0,
                max_capacity_gpu_hrs=8000.0,
                impact_kappa=0.009,
                preemption_probability=0.002,
                fee_rate=0.0,
                network_egress_latency_ms=10.0,
                queue_wait_latency_ms=25.0,
            ),
            ComputeVenue(
                venue_id="coreweave_ord1_dedicated",
                name="CoreWeave Chicago (Tier-2)",
                tier=VenueTier.TIER_2_SPECIALIZED,
                base_price_per_gpu_hr=2.65,
                market_depth=6000.0,
                max_capacity_gpu_hrs=5000.0,
                impact_kappa=0.015,
                preemption_probability=0.025,
                fee_rate=0.01,
                network_egress_latency_ms=12.0,
                queue_wait_latency_ms=35.0,
            ),
            ComputeVenue(
                venue_id="lambda_labs_slc1",
                name="Lambda Labs Utah (Tier-2)",
                tier=VenueTier.TIER_2_SPECIALIZED,
                base_price_per_gpu_hr=2.45,
                market_depth=4500.0,
                max_capacity_gpu_hrs=3500.0,
                impact_kappa=0.020,
                preemption_probability=0.040,
                fee_rate=0.01,
                network_egress_latency_ms=18.0,
                queue_wait_latency_ms=45.0,
            ),
            ComputeVenue(
                venue_id="vast_ai_spot_distributed",
                name="vast.ai Secondary Spot",
                tier=VenueTier.TIER_3_SPOT_BROKER,
                base_price_per_gpu_hr=1.55,
                market_depth=2500.0,
                max_capacity_gpu_hrs=2000.0,
                impact_kappa=0.045,
                preemption_probability=0.220,
                fee_rate=0.03,
                network_egress_latency_ms=35.0,
                queue_wait_latency_ms=90.0,
            ),
            ComputeVenue(
                venue_id="runpod_secure_cloud",
                name="RunPod Community Spot",
                tier=VenueTier.TIER_3_SPOT_BROKER,
                base_price_per_gpu_hr=1.75,
                market_depth=3000.0,
                max_capacity_gpu_hrs=2500.0,
                impact_kappa=0.040,
                preemption_probability=0.180,
                fee_rate=0.025,
                network_egress_latency_ms=28.0,
                queue_wait_latency_ms=75.0,
            ),
        ]

    def route_order(
        self,
        target_volume_gpu_hrs: float = 0.0,
        preemption_penalty_usd: float = 25.0,
        latency_weight_usd_per_ms: float = 0.0,
        max_latency_ms: Optional[float] = None,
        disallowed_tiers: Optional[Sequence[VenueTier]] = None,
        target_volume_gpu_hours: Optional[float] = None,
    ) -> RoutingSolution:
        """Routes target volume across eligible venues minimizing total cost + risk.

        Solves the convex optimization:
            min sum_k [ c_k * v_k + a_k * v_k^2 ]
            s.t. sum v_k = V_target,  0 <= v_k <= C_k

        where:
            c_k = P_k * (1 + fee_rate) + preemption_penalty * lambda_k + latency_weight * Latency_k
            a_k = kappa_k / D_k

        Args:
            target_volume_gpu_hrs: Target batch volume V_target (GPU-hours).
            preemption_penalty_usd: Interruption shortfall loss Cost_preempt ($/GPU-hr).
            latency_weight_usd_per_ms: Economic cost of execution latency ($/ms).
            max_latency_ms: Maximum allowable total latency filter (ms).
            disallowed_tiers: Specific infrastructure tiers excluded from routing.
            target_volume_gpu_hours: Optional alias for target_volume_gpu_hrs.

        Returns:
            RoutingSolution containing allocations, KKT dual clearing price, and metrics.
        """
        if target_volume_gpu_hours is not None:
            target_volume_gpu_hrs = target_volume_gpu_hours
        start_ns = time.perf_counter_ns()

        if target_volume_gpu_hrs <= 0.0:
            elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
            return RoutingSolution(
                target_volume_gpu_hrs=0.0,
                total_allocated_gpu_hrs=0.0,
                unallocated_gpu_hrs=0.0,
                total_execution_cost_usd=0.0,
                average_unit_cost_usd_per_gpu_hr=0.0,
                shadow_clearing_price_mu=0.0,
                allocations={},
                tier_distribution_gpu_hrs={tier: 0.0 for tier in VenueTier},
                execution_time_us=elapsed_us,
                is_fully_routed=True,
            )

        # 1. Filter eligible venues based on operational constraints
        disallowed_set = set(disallowed_tiers) if disallowed_tiers else set()
        eligible_venues: List[ComputeVenue] = []
        for v in self.venues:
            if not v.is_active:
                continue
            if v.tier in disallowed_set:
                continue
            if max_latency_ms is not None and v.total_latency_ms > max_latency_ms:
                continue
            if v.max_capacity_gpu_hrs > 0.0 and v.market_depth > 0.0:
                eligible_venues.append(v)

        if not eligible_venues:
            elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
            return RoutingSolution(
                target_volume_gpu_hrs=target_volume_gpu_hrs,
                total_allocated_gpu_hrs=0.0,
                unallocated_gpu_hrs=target_volume_gpu_hrs,
                total_execution_cost_usd=0.0,
                average_unit_cost_usd_per_gpu_hr=0.0,
                shadow_clearing_price_mu=0.0,
                allocations={},
                tier_distribution_gpu_hrs={tier: 0.0 for tier in VenueTier},
                execution_time_us=elapsed_us,
                is_fully_routed=False,
            )

        # 2. Extract linear cost c_k and quadratic impact a_k for each venue
        # c_k = P_k * (1 + fee) + Cost_preempt * lambda_k + Latency_pen
        # a_k = kappa_k / D_k
        venue_params = []
        total_capacity = 0.0

        for v in eligible_venues:
            c_k = (
                (v.base_price_per_gpu_hr * (1.0 + v.fee_rate))
                + (preemption_penalty_usd * v.preemption_probability)
                + (latency_weight_usd_per_ms * v.total_latency_ms)
            )
            a_k = max(1e-9, v.impact_kappa / v.market_depth)
            cap_k = v.max_capacity_gpu_hrs
            total_capacity += cap_k
            venue_params.append((v, c_k, a_k, cap_k))

        # Check for system capacity exhaustion
        if target_volume_gpu_hrs >= total_capacity:
            allocations_dict: Dict[str, VenueAllocation] = {}
            total_cost = 0.0
            tier_dist = {tier: 0.0 for tier in VenueTier}

            for v, c_k, a_k, cap_k in venue_params:
                v_alloc = cap_k
                b_cost = v.base_price_per_gpu_hr * v_alloc
                f_cost = v.fee_rate * b_cost
                imp_cost = v.impact_kappa * ((v_alloc ** 2) / v.market_depth)
                pre_cost = preemption_penalty_usd * v.preemption_probability * v_alloc
                lat_cost = latency_weight_usd_per_ms * v.total_latency_ms * v_alloc
                venue_total = b_cost + f_cost + imp_cost + pre_cost + lat_cost
                total_cost += venue_total
                tier_dist[v.tier] += v_alloc

                allocations_dict[v.venue_id] = VenueAllocation(
                    venue_id=v.venue_id,
                    venue_name=v.name,
                    tier=v.tier,
                    allocated_volume_gpu_hrs=v_alloc,
                    base_cost_usd=b_cost,
                    broker_fees_usd=f_cost,
                    market_impact_cost_usd=imp_cost,
                    preemption_risk_cost_usd=pre_cost,
                    latency_penalty_usd=lat_cost,
                    total_execution_cost_usd=venue_total,
                    effective_unit_cost_usd_per_gpu_hr=venue_total / v_alloc if v_alloc > 0 else 0.0,
                    utilization_pct=100.0,
                )

            elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
            return RoutingSolution(
                target_volume_gpu_hrs=target_volume_gpu_hrs,
                total_allocated_gpu_hrs=total_capacity,
                unallocated_gpu_hrs=target_volume_gpu_hrs - total_capacity,
                total_execution_cost_usd=total_cost,
                average_unit_cost_usd_per_gpu_hr=total_cost / total_capacity if total_capacity > 0 else 0.0,
                shadow_clearing_price_mu=max(c_k + 2.0 * a_k * cap_k for _, c_k, a_k, cap_k in venue_params),
                allocations=allocations_dict,
                tier_distribution_gpu_hrs=tier_dist,
                execution_time_us=elapsed_us,
                is_fully_routed=(target_volume_gpu_hrs <= total_capacity),
            )

        # 3. Exact O(K log K) KKT Breakpoint Water-Filling Solver
        # Critical events where slope dS/dmu changes:
        #   Event 1: mu = c_k (venue begins receiving volume, slope += 1 / (2 * a_k))
        #   Event 2: mu = c_k + 2 * a_k * cap_k (venue saturates at cap_k, slope -= 1 / (2 * a_k))
        events: List[Tuple[float, float]] = []
        for _, c_k, a_k, cap_k in venue_params:
            inv_2a = 1.0 / (2.0 * a_k)
            events.append((c_k, inv_2a))
            events.append((c_k + 2.0 * a_k * cap_k, -inv_2a))

        events.sort(key=lambda x: x[0])

        current_mu = events[0][0]
        current_volume = 0.0
        current_slope = 0.0
        mu_star = current_mu

        for event_mu, slope_delta in events:
            if event_mu > current_mu:
                volume_at_event = current_volume + current_slope * (event_mu - current_mu)
                if volume_at_event >= target_volume_gpu_hrs:
                    # Target volume is reached within this linear segment
                    if current_slope > 1e-12:
                        mu_star = current_mu + (target_volume_gpu_hrs - current_volume) / current_slope
                    else:
                        mu_star = current_mu
                    break
                current_volume = volume_at_event
                current_mu = event_mu
            current_slope += slope_delta

        # 4. Compute optimal continuous allocation v_k* = clamp((mu* - c_k) / (2 * a_k), 0, cap_k)
        raw_allocations: List[float] = []
        allocated_sum = 0.0

        for _, c_k, a_k, cap_k in venue_params:
            unconstrained_v = (mu_star - c_k) / (2.0 * a_k)
            v_k = max(0.0, min(cap_k, unconstrained_v))
            raw_allocations.append(v_k)
            allocated_sum += v_k

        # 5. Numerical residual correction to enforce exact sum v_k* == target_volume
        diff = target_volume_gpu_hrs - allocated_sum
        if abs(diff) > 1e-10:
            # Distribute residual among active, non-saturated venues
            unsaturated_indices = [
                i for i, (_, _, _, cap_k) in enumerate(venue_params)
                if 0.0 < raw_allocations[i] < cap_k
            ]
            if not unsaturated_indices and diff > 0.0:
                unsaturated_indices = [
                    i for i, (_, _, _, cap_k) in enumerate(venue_params)
                    if raw_allocations[i] < cap_k
                ]

            if unsaturated_indices:
                adj = diff / len(unsaturated_indices)
                for idx in unsaturated_indices:
                    cap_k = venue_params[idx][3]
                    raw_allocations[idx] = max(0.0, min(cap_k, raw_allocations[idx] + adj))

        # 6. Build final allocations and accounting structures
        allocations_dict = {}
        total_cost = 0.0
        tier_dist = {tier: 0.0 for tier in VenueTier}
        total_allocated = 0.0

        for i, (v, _, _, cap_k) in enumerate(venue_params):
            v_alloc = raw_allocations[i]
            total_allocated += v_alloc
            tier_dist[v.tier] += v_alloc

            b_cost = v.base_price_per_gpu_hr * v_alloc
            f_cost = v.fee_rate * b_cost
            imp_cost = v.impact_kappa * ((v_alloc ** 2) / v.market_depth)
            pre_cost = preemption_penalty_usd * v.preemption_probability * v_alloc
            lat_cost = latency_weight_usd_per_ms * v.total_latency_ms * v_alloc
            venue_total = b_cost + f_cost + imp_cost + pre_cost + lat_cost
            total_cost += venue_total

            allocations_dict[v.venue_id] = VenueAllocation(
                venue_id=v.venue_id,
                venue_name=v.name,
                tier=v.tier,
                allocated_volume_gpu_hrs=v_alloc,
                base_cost_usd=b_cost,
                broker_fees_usd=f_cost,
                market_impact_cost_usd=imp_cost,
                preemption_risk_cost_usd=pre_cost,
                latency_penalty_usd=lat_cost,
                total_execution_cost_usd=venue_total,
                effective_unit_cost_usd_per_gpu_hr=venue_total / v_alloc if v_alloc > 0 else 0.0,
                utilization_pct=(v_alloc / cap_k) * 100.0 if cap_k > 0 else 0.0,
            )

        elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
        return RoutingSolution(
            target_volume_gpu_hrs=target_volume_gpu_hrs,
            total_allocated_gpu_hrs=total_allocated,
            unallocated_gpu_hrs=max(0.0, target_volume_gpu_hrs - total_allocated),
            total_execution_cost_usd=total_cost,
            average_unit_cost_usd_per_gpu_hr=total_cost / total_allocated if total_allocated > 0 else 0.0,
            shadow_clearing_price_mu=mu_star,
            allocations=allocations_dict,
            tier_distribution_gpu_hrs=tier_dist,
            execution_time_us=elapsed_us,
            is_fully_routed=True,
        )


class LiquidityKnapsackSolver:
    """Discrete GPU Node Liquidity Knapsack Solver.

    Quantizes continuous convex allocations to physical multi-GPU chassis nodes
    (e.g., 8-GPU servers) and solves discrete knapsack selections under latency and budget caps.
    """

    @staticmethod
    def quantize_nodes(
        continuous_solution: RoutingSolution,
        venues: Sequence[ComputeVenue],
        duration_hours: float = 1.0,
    ) -> Dict[str, int]:
        """Quantizes continuous GPU-hour allocations into integer physical server nodes.

        Node GPU-hours = node_gpu_count * duration_hours.
        Uses largest-remainder quantization (Hamilton-Hare method) preserving capacity bounds.

        Args:
            continuous_solution: Optimal continuous RoutingSolution from SOR.
            venues: List of ComputeVenue specs.
            duration_hours: Job lease duration in hours.

        Returns:
            Dictionary mapping venue_id to integer count of leased server nodes.
        """
        venue_map = {v.venue_id: v for v in venues}
        quantized_nodes: Dict[str, int] = {}
        remainders: List[Tuple[float, str, int]] = []

        total_target_nodes = int(round(continuous_solution.target_volume_gpu_hrs / (8.0 * duration_hours)))
        allocated_nodes_sum = 0

        for venue_id, alloc in continuous_solution.allocations.items():
            venue = venue_map.get(venue_id)
            if not venue:
                continue

            node_size = venue.node_gpu_count * duration_hours
            if node_size <= 0.0:
                continue

            max_nodes = int(math.floor(venue.max_capacity_gpu_hrs / node_size))
            exact_nodes = alloc.allocated_volume_gpu_hrs / node_size
            floor_nodes = min(max_nodes, int(math.floor(exact_nodes)))
            rem = exact_nodes - floor_nodes

            quantized_nodes[venue_id] = floor_nodes
            allocated_nodes_sum += floor_nodes
            if floor_nodes < max_nodes:
                remainders.append((rem, venue_id, max_nodes))

        # Distribute remaining nodes by highest fractional remainder
        remainders.sort(key=lambda x: x[0], reverse=True)
        needed = total_target_nodes - allocated_nodes_sum

        for _, venue_id, max_nodes in remainders:
            if needed <= 0:
                break
            if quantized_nodes[venue_id] < max_nodes:
                quantized_nodes[venue_id] += 1
                needed -= 1

        return quantized_nodes

    @staticmethod
    def solve_budget_knapsack(
        venues: Sequence[ComputeVenue],
        max_budget_usd: float,
        preemption_penalty_usd: float = 25.0,
    ) -> Tuple[List[str], float, float]:
        """0-1 Knapsack selector: selects discrete cluster chassis under a fixed budget cap.

        Maximizes total risk-adjusted compute capacity (GPU-hrs * (1 - lambda_k)).

        Args:
            venues: Available compute candidate venues.
            max_budget_usd: Financial expenditure budget ceiling.
            preemption_penalty_usd: Interruption penalty parameter.

        Returns:
            Tuple of (selected_venue_ids, total_risk_adjusted_volume, total_cost_usd).
        """
        items = []
        for v in venues:
            if not v.is_active:
                continue
            cost = v.base_price_per_gpu_hr * v.max_capacity_gpu_hrs * (1.0 + v.fee_rate)
            # Risk-adjusted utility value: reliable GPU-hours
            value = v.max_capacity_gpu_hrs * (1.0 - v.preemption_probability)
            if cost <= max_budget_usd and cost > 0:
                items.append((v.venue_id, cost, value))

        # Scaled DP knapsack for precision
        budget_int = int(math.floor(max_budget_usd))
        dp = [0.0] * (budget_int + 1)
        parent: List[List[Optional[int]]] = [[None] * (budget_int + 1) for _ in range(len(items))]

        for i, (v_id, cost, val) in enumerate(items):
            c_int = int(math.ceil(cost))
            for b in range(budget_int, c_int - 1, -1):
                if dp[b - c_int] + val > dp[b]:
                    dp[b] = dp[b - c_int] + val
                    parent[i][b] = b - c_int

        # Backtrack solution
        curr_b = max(range(budget_int + 1), key=lambda b: dp[b])
        selected_ids: List[str] = []
        total_cost_calc = 0.0

        for i in range(len(items) - 1, -1, -1):
            if parent[i][curr_b] is not None:
                selected_ids.append(items[i][0])
                total_cost_calc += items[i][1]
                curr_b = parent[i][curr_b]  # type: ignore

        return selected_ids, dp[curr_b if not selected_ids else int(math.floor(max_budget_usd))], total_cost_calc
