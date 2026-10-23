"""Real-Time Cross-Venue Compute Arbitrage Engine.

Scans spot order books against forward yield curves, computes non-linear
order-book market impact, and formulates sub-millisecond execution quotes.
"""

import math
import time
from typing import Dict, List, Optional
from apex_compute_futures.core.models import ArbitrageQuote, MarketVenue


class ComputeArbitrageEngine:
    """Scans and evaluates risk-adjusted arbitrage spreads across spot compute venues."""

    def __init__(self, slippage_kappa: float = 0.05):
        self.slippage_kappa = slippage_kappa

    def calculate_depth_slippage(self, order_gpu_count: int, venue_depth: int = 64) -> float:
        """Calculates square-root market impact slippage in basis points."""
        if venue_depth <= 0:
            return 500.0  # High penalty if zero depth
        impact = self.slippage_kappa * math.sqrt(order_gpu_count / venue_depth)
        return impact * 10000.0  # bps

    def scan_opportunity(
        self,
        opportunity_id: str,
        venue: MarketVenue,
        node_type: str,
        gpu_count: int,
        spot_ask_rate: float,
        forward_clearance_price: float,
        venue_depth: int = 64
    ) -> Optional[ArbitrageQuote]:
        """Evaluates whether an instantaneous spot quote yields positive net edge."""
        t_start = time.perf_counter_ns()

        if spot_ask_rate >= forward_clearance_price:
            return None  # Negative gross spread

        gross_spread = forward_clearance_price - spot_ask_rate
        gross_margin_bps = (gross_spread / forward_clearance_price) * 10000.0

        slippage_bps = self.calculate_depth_slippage(gpu_count, venue_depth)
        taker_fee_bps = 25.0  # 25 bps broker clearing fee

        net_edge_bps = gross_margin_bps - slippage_bps - taker_fee_bps
        if net_edge_bps <= 0:
            return None

        latency_us = (time.perf_counter_ns() - t_start) / 1000.0

        return ArbitrageQuote(
            opportunity_id=opportunity_id,
            venue=venue,
            node_type=node_type,
            gpu_count=gpu_count,
            spot_ask_rate=round(spot_ask_rate, 4),
            forward_clearance_price=round(forward_clearance_price, 4),
            gross_margin_bps=round(gross_margin_bps, 2),
            estimated_slippage_bps=round(slippage_bps, 2),
            net_edge_bps=round(net_edge_bps, 2),
            latency_microseconds=round(latency_us, 2)
        )
