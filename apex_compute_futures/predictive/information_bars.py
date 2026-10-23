"""Information-Driven Sampling Clock (CUSUM Energy/Compute Bars).

Replaces calendar time-bars with informational bars sampled upon cumulative
surplus deviation of normalized power-compute turnover.
"""

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class InformationBar:
    """Standardized information bar observation."""
    bar_index: int
    open_time: float
    close_time: float
    open_rate: float
    high_rate: float
    low_rate: float
    close_rate: float
    volume_units: float
    total_energy_mwh: float


class CusumInformationSampler:
    """Samples dynamic bars based on cumulative symmetric deviations."""

    def __init__(self, threshold_factor: float = 2.0, initial_threshold: float = 0.50):
        self.threshold_factor = threshold_factor
        self.threshold = initial_threshold
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.last_bar_close: Optional[float] = None
        self.current_ticks: List[float] = []
        self.current_volume = 0.0
        self.current_energy = 0.0
        self.bar_count = 0

    def process_tick(
        self,
        rate: float,
        timestamp: float,
        volume: float = 1.0,
        energy_mwh: float = 0.01
    ) -> Optional[InformationBar]:
        """Ingests a raw price/rate tick and emits an InformationBar when threshold breaches."""
        if self.last_bar_close is None:
            self.last_bar_close = rate
            self.current_ticks = [rate]
            self.current_volume = volume
            self.current_energy = energy_mwh
            self.open_time = timestamp
            return None

        delta = rate - self.last_bar_close
        self.s_pos = max(0.0, self.s_pos + delta)
        self.s_neg = min(0.0, self.s_neg + delta)

        self.current_ticks.append(rate)
        self.current_volume += volume
        self.current_energy += energy_mwh

        # Check breach
        if max(self.s_pos, -self.s_neg) >= self.threshold:
            bar = InformationBar(
                bar_index=self.bar_count,
                open_time=self.open_time,
                close_time=timestamp,
                open_rate=self.current_ticks[0],
                high_rate=max(self.current_ticks),
                low_rate=min(self.current_ticks),
                close_rate=rate,
                volume_units=self.current_volume,
                total_energy_mwh=self.current_energy
            )

            # Reset CUSUM state
            self.bar_count += 1
            self.s_pos = 0.0
            self.s_neg = 0.0
            self.last_bar_close = rate
            self.current_ticks = [rate]
            self.current_volume = 0.0
            self.current_energy = 0.0
            self.open_time = timestamp
            return bar

        return None
