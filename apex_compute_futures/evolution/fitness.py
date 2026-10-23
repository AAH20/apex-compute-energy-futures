"""Evolutionary Fitness Engine with Hard Barrier Constraints.

Implements the Deflated Sharpe Ratio (DSR) with hard-barrier disqualification
for SLA invalidations and drawdown breaches.
"""

import math
from typing import List, Tuple


class SystemFitnessCalculator:
    """Calculates statistically deflated risk-adjusted fitness with barrier constraints."""

    def __init__(
        self,
        max_sla_invalidation_rate: float = 0.0005,  # 0.05% max breach tolerance
        max_drawdown_limit: float = 0.15,          # 15% max acceptable drawdown
        risk_free_rate_annual: float = 0.045
    ):
        self.max_sla_rate = max_sla_invalidation_rate
        self.max_dd = max_drawdown_limit
        self.rf = risk_free_rate_annual

    @staticmethod
    def calculate_moments(returns: List[float]) -> Tuple[float, float, float, float]:
        """Calculates mean, variance, skewness, and kurtosis of returns."""
        n = len(returns)
        if n < 3:
            return 0.0, 1.0, 0.0, 3.0

        mean = sum(returns) / n
        var = sum((r - mean) ** 2 for r in returns) / (n - 1)
        std = math.sqrt(max(1e-9, var))

        skew = (sum((r - mean) ** 3 for r in returns) / n) / (std ** 3)
        kurt = (sum((r - mean) ** 4 for r in returns) / n) / (std ** 4)

        return mean, std, skew, kurt

    @classmethod
    def calculate_deflated_sharpe_ratio(
        cls,
        returns: List[float],
        num_trials: int = 100,
        annualization_factor: float = 8760.0  # Hourly periods
    ) -> float:
        """Calculates Deflated Sharpe Ratio (Bailey & López de Prado)."""
        if len(returns) < 5:
            return 0.0

        mean, std, skew, kurt = cls.calculate_moments(returns)
        if std == 0.0:
            return 0.0

        # Annualized Sharpe Ratio
        sr = (mean / std) * math.sqrt(annualization_factor)

        # Expected maximum Sharpe under null hypothesis (independent trials)
        euler_gamma = 0.5772156649
        z_n = (1.0 - euler_gamma) * math.sqrt(2.0 * math.log(max(2, num_trials)))
        expected_max_sr = max(0.0, z_n)

        # Variance of Sharpe ratio estimator
        var_sr = (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr ** 2)) / len(returns)
        std_sr = math.sqrt(max(1e-9, var_sr))

        # Test statistic
        dsr_stat = (sr - expected_max_sr) / std_sr
        # Standard normal CDF
        dsr_prob = 0.5 * (1.0 + math.erf(dsr_stat / math.sqrt(2.0)))
        return round(dsr_prob, 4)

    def evaluate_archetype_fitness(
        self,
        returns: List[float],
        sla_invalidation_rate: float,
        realized_max_drawdown: float,
        num_generations_tested: int = 50,
        contribution_margin_ratio: float = 4.8
    ) -> float:
        """Evaluates survival fitness score with hard barrier boundaries.
        
        Returns -infinity if hard safety boundaries are violated.
        """
        # 1. Hard Barrier Check: SLA Invalidation
        if sla_invalidation_rate > self.max_sla_rate:
            return float("-inf")

        # 2. Hard Barrier Check: Drawdown Breached
        if realized_max_drawdown > self.max_dd:
            return float("-inf")

        # 3. Deflated Sharpe Ratio
        dsr = self.calculate_deflated_sharpe_ratio(returns, num_generations_tested)
        if dsr <= 0.0:
            return 0.0

        # Non-linear scalar score: DSR * (1 - Drawdown / MaxDD) * log(ContributionMargin)
        dd_factor = max(0.0, 1.0 - (realized_max_drawdown / self.max_dd))
        margin_factor = math.log(max(1.0, contribution_margin_ratio))

        fitness = dsr * dd_factor * margin_factor
        return round(fitness, 4)
