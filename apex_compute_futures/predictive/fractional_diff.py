"""Fractional Differentiation Engine.

Transforms non-stationary commodity power and GPU spot prices into stationary
series while preserving long-term memory via fractional integration (1 - L)^d.
Pure Python standard library implementation.
"""

import math
from typing import List, Tuple


class FractionalDifferentiator:
    """Computes fractional differentiation weights and transforms time series."""

    @staticmethod
    def get_weights(d: float, size: int, threshold: float = 1e-4) -> List[float]:
        """Generates binomial expansion weights for (1 - L)^d.
        
        w_0 = 1
        w_k = -w_{k-1} * (d - k + 1) / k
        """
        weights = [1.0]
        k = 1
        while k < size:
            w_k = -weights[-1] * (d - k + 1) / k
            if abs(w_k) < threshold:
                break
            weights.append(w_k)
            k += 1
        return weights

    @classmethod
    def transform(cls, series: List[float], d: float, threshold: float = 1e-4) -> List[float]:
        """Applies fractional differentiation with order d to a 1D time series.
        
        Preserves autocorrelation and memory while removing stochastic drift.
        """
        n = len(series)
        if n == 0:
            return []
        
        weights = cls.get_weights(d, n, threshold)
        w_len = len(weights)
        
        transformed = []
        for i in range(n):
            val = 0.0
            for j in range(min(i + 1, w_len)):
                val += weights[j] * series[i - j]
            transformed.append(val)
        return transformed

    @staticmethod
    def memory_retention_ratio(original: List[float], differentiated: List[float]) -> float:
        """Calculates Pearson correlation between original and differenced series."""
        if len(original) != len(differentiated) or len(original) < 2:
            return 0.0

        n = len(original)
        mean_orig = sum(original) / n
        mean_diff = sum(differentiated) / n

        num = sum((original[i] - mean_orig) * (differentiated[i] - mean_diff) for i in range(n))
        den_orig = math.sqrt(sum((x - mean_orig) ** 2 for x in original))
        den_diff = math.sqrt(sum((y - mean_diff) ** 2 for y in differentiated))

        if den_orig == 0.0 or den_diff == 0.0:
            return 0.0
        return max(-1.0, min(1.0, num / (den_orig * den_diff)))

    @classmethod
    def find_optimal_d(cls, series: List[float], min_memory: float = 0.70) -> float:
        """Finds minimum fractional order d that achieves stationarity while retaining memory."""
        best_d = 0.50
        for test_d in [x * 0.05 for x in range(2, 19)]: # 0.10 to 0.90
            diff = cls.transform(series, test_d)
            corr = cls.memory_retention_ratio(series, diff)
            if corr >= min_memory:
                best_d = test_d
        return round(best_d, 2)
