"""Automated Energy-Compute Market Maker (A-ECMM) Engine.

Executes continuous multi-asset invariant bonding curve swaps:
Phi(E, C, f) = E^alpha * C^beta * f^gamma = K
Arbitrages synthetic inertia and fast-frequency reserve power against compute load.
"""

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AmmPoolState:
    """State of the physical/virtual A-ECMM pool."""
    stored_energy_mwh: float          # Fast response BESS/supercap/flywheel energy
    throttling_compute_mw: float      # Flexible shedable compute capacity (DVFS)
    grid_frequency_hz: float          # Baseline: 60.0 Hz
    invariant_k: float


@dataclass
class InertiaSwapQuote:
    """Instantaneous quote for power-frequency stabilization swap."""
    frequency_deviation_hz: float
    required_power_mw: float
    gpu_shed_fraction: float          # e.g., 0.40 (shed 40% clock rate)
    clearing_revenue_per_mwh: float
    synthesized_inertia_mw_s: float


class SyntheticInertiaAMM:
    """Multi-asset continuous invariant bonding curve AMM for power and compute."""

    def __init__(
        self,
        initial_energy_mwh: float = 25.0,
        initial_compute_mw: float = 500.0,
        nominal_frequency_hz: float = 60.0,
        alpha: float = 0.45,
        beta: float = 0.45,
        gamma: float = 0.10
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.nominal_f = nominal_frequency_hz

        # Calculate initial constant K
        self.state = AmmPoolState(
            stored_energy_mwh=initial_energy_mwh,
            throttling_compute_mw=initial_compute_mw,
            grid_frequency_hz=nominal_frequency_hz,
            invariant_k=self._calc_k(initial_energy_mwh, initial_compute_mw, nominal_frequency_hz)
        )

    def _calc_k(self, e: float, c: float, f: float) -> float:
        """Computes K = E^alpha * C^beta * f^gamma."""
        return (e ** self.alpha) * (c ** self.beta) * (f ** self.gamma)

    def quote_frequency_response(
        self,
        observed_frequency_hz: float,
        ancillary_market_clearing_price_usd_per_mwh: float = 4500.0
    ) -> InertiaSwapQuote:
        """Calculates compute shedding and synthetic inertia injection during grid trips."""
        delta_f = observed_frequency_hz - self.nominal_f

        if delta_f >= 0:
            # Grid normal or over-frequency; zero emergency shedding required
            return InertiaSwapQuote(
                frequency_deviation_hz=delta_f,
                required_power_mw=0.0,
                gpu_shed_fraction=0.0,
                clearing_revenue_per_mwh=0.0,
                synthesized_inertia_mw_s=0.0
            )

        # Under-frequency trip: calculate required compute shedding to inject synthetic inertia
        # Elasticity: dC / df = (gamma / beta) * (C_0 / f_0) * DroopMultiplier
        droop_gain = 20.0  # Standard 5% grid governor droop characteristic
        norm_delta_f = abs(delta_f) / self.nominal_f
        shed_fraction = min(0.75, (self.gamma / self.beta) * droop_gain * norm_delta_f)
        shed_mw = self.state.throttling_compute_mw * shed_fraction

        # 30-second energy injected in MWh
        energy_dispatched_mwh = (shed_mw * 30.0) / 3600.0
        revenue = energy_dispatched_mwh * ancillary_market_clearing_price_usd_per_mwh

        # Synthesized inertia: H = E_kin / S_rated (MW-seconds)
        inertia_mw_s = shed_mw * 30.0

        return InertiaSwapQuote(
            frequency_deviation_hz=round(delta_f, 3),
            required_power_mw=round(shed_mw, 2),
            gpu_shed_fraction=round(shed_fraction, 3),
            clearing_revenue_per_mwh=round(revenue, 2),
            synthesized_inertia_mw_s=round(inertia_mw_s, 2)
        )
