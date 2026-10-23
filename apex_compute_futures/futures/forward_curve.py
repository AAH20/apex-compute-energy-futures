"""Compute Forward Curve and Yield Surface Engine.

Calculates forward price, cost-of-carry, and basis surfaces across regional
datacenters and maturity horizons (1h to 90d).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
from apex_compute_futures.core.models import ComputeContract, SLATier


@dataclass
class ForwardCurvePoint:
    """Forward price evaluation at maturity T."""
    maturity_hours: float
    forward_rate_per_gpu_hr: float
    cost_of_carry_pct: float
    convenience_yield_pct: float
    basis_spread_usd: float


class ComputeForwardCurve:
    """Calculates forward yield curves using continuous cost-of-carry models."""

    def __init__(
        self,
        region: str,
        base_spot_rate: float = 2.45,
        risk_free_rate_annual: float = 0.045,
        facility_pue: float = 1.25,
        average_power_cost_per_kwh: float = 0.065
    ):
        self.region = region
        self.base_spot_rate = base_spot_rate
        self.risk_free_rate = risk_free_rate_annual
        self.facility_pue = facility_pue
        self.power_cost = average_power_cost_per_kwh

    def get_carry_costs(self, gpu_kw_draw: float = 1.0) -> Tuple[float, float]:
        """Calculates effective continuous storage/carry cost and convenience yield."""
        # Energy cost per GPU-hour = kW * PUE * $/kWh
        energy_cost_per_hr = gpu_kw_draw * self.facility_pue * self.power_cost
        annualized_carry = (energy_cost_per_hr * 8760.0) / (self.base_spot_rate * 8760.0)
        
        # Convenience yield (scarcity premium of holding immediate uncommitted capacity)
        convenience_yield = 0.08  # 8% annualized convenience premium for instant-availability H100s
        return annualized_carry, convenience_yield

    def price_forward(self, maturity_hours: float, sla: SLATier = SLATier.FAULT_TOLERANT) -> ForwardCurvePoint:
        """Computes forward rate: F(t, T) = S_t * exp((r + c - y) * (T - t))."""
        t_years = maturity_hours / 8760.0
        carry_cost, conv_yield = self.get_carry_costs()

        # SLA multiplier
        sla_multiplier = {
            SLATier.BEST_EFFORT: 0.85,
            SLATier.FAULT_TOLERANT: 1.00,
            SLATier.MISSION_CRITICAL: 1.35
        }.get(sla, 1.00)

        exponent = (self.risk_free_rate + carry_cost - conv_yield) * t_years
        forward_price = self.base_spot_rate * math.exp(exponent) * sla_multiplier
        basis = forward_price - self.base_spot_rate

        return ForwardCurvePoint(
            maturity_hours=maturity_hours,
            forward_rate_per_gpu_hr=round(forward_price, 4),
            cost_of_carry_pct=round(carry_cost * 100, 2),
            convenience_yield_pct=round(conv_yield * 100, 2),
            basis_spread_usd=round(basis, 4)
        )

    def generate_full_term_structure(self) -> List[ForwardCurvePoint]:
        """Generates the standard institutional term structure: 1h, 6h, 24h, 72h, 168h, 720h, 2160h."""
        maturities = [1.0, 6.0, 24.0, 72.0, 168.0, 720.0, 2160.0]
        return [self.price_forward(m) for m in maturities]
