"""Triple-Barrier Labeling and Meta-Labeling Position Sizer.

Applies Marcos López de Prado's triple-barrier methodology to compute-energy
arbitrage and options forward spreads.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class BarrierEvent:
    """Records the outcome of a triple-barrier trajectory."""
    entry_index: int
    exit_index: int
    entry_rate: float
    exit_rate: float
    label: int                       # 1: Upper (Profit), -1: Lower (Loss), 0: Vertical (Timeout)
    realized_return: float
    holding_periods: int


class TripleBarrierLabeler:
    """Labels market observations using dynamic upper, lower, and vertical barriers."""

    def __init__(
        self,
        profit_multiplier: float = 2.0,
        stop_loss_multiplier: float = 1.0,
        max_holding_horizon: int = 24
    ):
        self.profit_multiplier = profit_multiplier
        self.stop_loss_multiplier = stop_loss_multiplier
        self.max_holding_horizon = max_holding_horizon

    def label_trajectory(
        self,
        rates: List[float],
        entry_idx: int,
        rolling_volatility: float
    ) -> BarrierEvent:
        """Determines which barrier is breached first along a price trajectory."""
        entry_p = rates[entry_idx]
        upper_barrier = entry_p * (1.0 + self.profit_multiplier * rolling_volatility)
        lower_barrier = entry_p * (1.0 - self.stop_loss_multiplier * rolling_volatility)
        max_idx = min(len(rates) - 1, entry_idx + self.max_holding_horizon)

        for curr_idx in range(entry_idx + 1, max_idx + 1):
            curr_p = rates[curr_idx]
            # Check upper
            if curr_p >= upper_barrier:
                ret = (curr_p - entry_p) / entry_p
                return BarrierEvent(entry_idx, curr_idx, entry_p, curr_p, 1, ret, curr_idx - entry_idx)
            # Check lower
            if curr_p <= lower_barrier:
                ret = (curr_p - entry_p) / entry_p
                return BarrierEvent(entry_idx, curr_idx, entry_p, curr_p, -1, ret, curr_idx - entry_idx)

        # Reached vertical barrier
        final_p = rates[max_idx]
        ret = (final_p - entry_p) / entry_p
        return BarrierEvent(entry_idx, max_idx, entry_p, final_p, 0, ret, max_idx - entry_idx)

    @staticmethod
    def compute_meta_allocation_size(
        probability_success: float,
        calibration_sigma: float = 0.25
    ) -> float:
        """Scales allocation fraction using calibrated Gaussian CDF around neutral 0.50.
        
        Size in [-1.0, 1.0]. Zero when prob = 0.50.
        """
        z = (probability_success - 0.50) / calibration_sigma
        # Error function based standard normal CDF approximation
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        size = 2.0 * cdf - 1.0
        return max(-1.0, min(1.0, size))
