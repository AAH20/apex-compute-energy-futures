r"""Hawkes Self-Exciting Point Process Engine.

Models temporal clustering of spot GPU preemption cascades and grid frequency
disturbance shocks via continuous self-exciting point processes:
    \lambda(t) = \mu_0 + \sum_{t_i < t} \alpha \cdot \exp(-\beta (t - t_i))

Features:
- Sub-critical branching ratio \eta = \alpha / \beta with stability validation (\eta < 1).
- O(N) recursive intensity evaluation.
- Analytical log-likelihood evaluation over observation intervals [0, T].
- Closed-form next-event conditional arrival probability within horizon \Delta t:
    P(T_{n+1} \le t + \Delta t | \mathcal{F}_t) = 1 - \exp\left( - \int_t^{t+\Delta t} \lambda(s) ds \right)
- Ogata thinning simulation algorithm.
- Maximum Likelihood Estimation (MLE) calibration via Nelder-Mead simplex optimization.
- Pure Python standard library implementation with zero third-party dependencies.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class HawkesParameters:
    r"""Calibrated parameters of the univariate self-exciting Hawkes process."""
    mu0: float                       # Baseline background arrival rate (\mu_0 > 0)
    alpha: float                     # Excitation magnitude (\alpha >= 0)
    beta: float                      # Exponential decay rate (\beta > 0)
    branching_ratio: float           # \eta = \alpha / \beta
    is_stable: bool                  # \eta < 1.0 (sub-critical stability)
    asymptotic_rate: float           # Asymptotic stationary intensity \mu_0 / (1 - \eta)
    log_likelihood: float = 0.0      # Sample log-likelihood
    aic: float = 0.0                 # Akaike Information Criterion


@dataclass
class PreemptionShockEvent:
    """Represents a discrete preemption or grid disturbance shock event."""
    timestamp: float
    intensity_before: float          # \lambda(t^-)
    intensity_after: float           # \lambda(t^+)
    cluster_generation: int = 0      # 0 for immigrant, >0 for offspring


class HawkesProcess:
    r"""Univariate exponential Hawkes self-exciting point process model."""

    def __init__(
        self,
        mu0: float = 0.1,
        alpha: float = 0.4,
        beta: float = 1.0
    ):
        if mu0 <= 0.0:
            raise ValueError(f"mu0 must be strictly positive, got {mu0}")
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if beta <= 0.0:
            raise ValueError(f"beta must be strictly positive, got {beta}")

        self.mu0 = mu0
        self.alpha = alpha
        self.beta = beta

    @property
    def branching_ratio(self) -> float:
        r"""Calculates branching ratio \eta = \alpha / \beta."""
        return self.alpha / self.beta

    @property
    def is_stable(self) -> bool:
        r"""Checks sub-critical stability condition: \eta < 1.0."""
        return self.branching_ratio < 1.0

    @property
    def asymptotic_rate(self) -> float:
        r"""Calculates long-term stationary event intensity: \bar{\lambda} = \mu_0 / (1 - \eta)."""
        eta = self.branching_ratio
        if eta >= 1.0:
            return float("inf")
        return self.mu0 / (1.0 - eta)

    def intensity_at(self, t: float, history: List[float]) -> float:
        r"""Evaluates exact conditional intensity \lambda(t) given past events t_i < t."""
        if t < 0.0:
            return self.mu0

        decay_sum = 0.0
        for ti in history:
            if ti >= t:
                break
            decay_sum += math.exp(-self.beta * (t - ti))

        return self.mu0 + self.alpha * decay_sum

    def compute_event_intensities(
        self,
        events: List[float]
    ) -> List[PreemptionShockEvent]:
        r"""Computes \lambda(t_i^-) and \lambda(t_i^+) in O(N) using recursive formulation.
        
        R_1 = 0
        R_i = exp(-\beta (t_i - t_{i-1})) * (R_{i-1} + 1)
        \lambda(t_i^-) = \mu_0 + \alpha R_i
        \lambda(t_i^+) = \lambda(t_i^-) + \alpha
        """
        if not events:
            return []

        n = len(events)
        result: List[PreemptionShockEvent] = []

        r = 0.0
        lambda_minus = self.mu0
        lambda_plus = lambda_minus + self.alpha
        result.append(
            PreemptionShockEvent(
                timestamp=events[0],
                intensity_before=lambda_minus,
                intensity_after=lambda_plus
            )
        )

        for i in range(1, n):
            dt = events[i] - events[i - 1]
            if dt < 0.0:
                raise ValueError(f"Event timestamps must be monotonically non-decreasing: t_{i-1}={events[i-1]}, t_{i}={events[i]}")
            r = math.exp(-self.beta * dt) * (r + 1.0)
            lambda_minus = self.mu0 + self.alpha * r
            lambda_plus = lambda_minus + self.alpha
            result.append(
                PreemptionShockEvent(
                    timestamp=events[i],
                    intensity_before=lambda_minus,
                    intensity_after=lambda_plus
                )
            )

        return result

    def log_likelihood(self, events: List[float], t_end: Optional[float] = None) -> float:
        r"""Calculates exact log-likelihood of events over observation window [0, T].
        
        \ln L = \sum_{i=1}^n \ln \lambda(t_i^-) - \int_0^T \lambda(s) ds
        where \int_0^T \lambda(s) ds = \mu_0 T + \frac{\alpha}{\beta} \sum_{i=1}^n (1 - \exp(-\beta(T - t_i)))
        """
        if not events:
            t = t_end if t_end is not None else 0.0
            return -self.mu0 * max(0.0, t)

        t_final = t_end if t_end is not None else events[-1]
        if t_final < events[-1]:
            raise ValueError(f"t_end ({t_final}) cannot be earlier than last event ({events[-1]})")

        n = len(events)
        # Sum of log intensities at event arrival times
        r = 0.0
        sum_log_lambda = math.log(max(1e-12, self.mu0))

        for i in range(1, n):
            dt = events[i] - events[i - 1]
            r = math.exp(-self.beta * dt) * (r + 1.0)
            lambda_minus = self.mu0 + self.alpha * r
            sum_log_lambda += math.log(max(1e-12, lambda_minus))

        # Compensator integral \int_0^T \lambda(s) ds
        integral_baseline = self.mu0 * t_final
        decay_factor = self.alpha / self.beta
        integral_excitation = decay_factor * sum(
            1.0 - math.exp(-self.beta * (t_final - ti)) for ti in events
        )

        log_lik = sum_log_lambda - (integral_baseline + integral_excitation)
        return log_lik

    def next_event_probability(
        self,
        current_time: float,
        horizon: float,
        history: List[float]
    ) -> float:
        r"""Calculates closed-form probability of at least one arrival in [current_time, current_time + horizon].
        
        P(T_{n+1} \le t + \Delta t | \mathcal{F}_t) = 1 - \exp\left( - \int_t^{t+\Delta t} \lambda(s) ds \right)
        For s >= t, assuming no arrivals occur between t and s:
            \lambda(s) = \mu_0 + (\lambda(t) - \mu_0) \exp(-\beta (s - t))
            \int_t^{t+\Delta t} \lambda(s) ds = \mu_0 \Delta t + \frac{\lambda(t) - \mu_0}{\beta} (1 - \exp(-\beta \Delta t))
        """
        if horizon <= 0.0:
            return 0.0

        current_intensity = self.intensity_at(current_time, history)
        excitation_component = max(0.0, current_intensity - self.mu0)

        integral = self.mu0 * horizon + (excitation_component / self.beta) * (
            1.0 - math.exp(-self.beta * horizon)
        )

        prob = 1.0 - math.exp(-max(0.0, integral))
        return min(1.0, max(0.0, prob))

    def simulate(
        self,
        t_end: float,
        seed: Optional[int] = None
    ) -> List[float]:
        r"""Simulates synthetic event arrival path using Ogata's modified thinning algorithm."""
        if seed is not None:
            random.seed(seed)
        if t_end <= 0.0:
            return []

        events: List[float] = []
        t = 0.0
        r = 0.0

        while t < t_end:
            # Current upper bound on intensity: immediately after last event (or at t=0)
            if not events:
                lambda_bar = self.mu0 + self.alpha
            else:
                lambda_bar = self.mu0 + self.alpha * (r + 1.0)

            # Generate candidate time step from homogeneous Poisson process with rate lambda_bar
            u = random.random()
            if u < 1e-12:
                u = 1e-12
            w = -math.log(u) / lambda_bar
            t += w

            if t >= t_end:
                break

            # Evaluate exact intensity at candidate time t
            if not events:
                lambda_t = self.mu0
            else:
                dt = t - events[-1]
                decayed_r = math.exp(-self.beta * dt) * (r + 1.0)
                lambda_t = self.mu0 + self.alpha * decayed_r

            # Acceptance-rejection step
            d = random.random()
            if d <= (lambda_t / lambda_bar):
                # Accepted
                if events:
                    dt = t - events[-1]
                    r = math.exp(-self.beta * dt) * (r + 1.0)
                else:
                    r = 0.0
                events.append(t)

        return events

    @classmethod
    def fit(
        cls,
        events: List[float],
        t_end: Optional[float] = None,
        max_iter: int = 120
    ) -> "HawkesProcess":
        r"""Calibrates (\mu_0, \alpha, \beta) via Maximum Likelihood Estimation (Nelder-Mead)."""
        if len(events) < 3:
            raise ValueError(f"At least 3 events required to calibrate Hawkes process, got {len(events)}")

        t_final = t_end if t_end is not None else events[-1]
        n = len(events)
        obs_rate = n / max(1e-6, t_final)

        # Initial parameter estimates
        init_mu = max(1e-3, obs_rate * 0.5)
        init_beta = 1.0
        init_alpha = 0.4 * init_beta

        def objective(params: List[float]) -> float:
            mu0, alpha, beta = params
            # Penalize invalid parameter configurations
            if mu0 <= 1e-5 or alpha < 0.0 or beta <= 1e-5:
                return 1e12
            if (alpha / beta) >= 0.999:  # Enforce sub-critical stability
                return 1e12 + 1e6 * ((alpha / beta) - 0.999)

            model = cls(mu0=mu0, alpha=alpha, beta=beta)
            return -model.log_likelihood(events, t_final)

        # Pure Python Nelder-Mead simplex optimizer
        x0 = [init_mu, init_alpha, init_beta]
        best_params = cls._nelder_mead(objective, x0, max_iter=max_iter)

        mu_opt, alpha_opt, beta_opt = best_params
        mu_opt = max(1e-4, mu_opt)
        beta_opt = max(1e-4, beta_opt)
        alpha_opt = max(0.0, min(0.99 * beta_opt, alpha_opt))

        return cls(mu0=mu_opt, alpha=alpha_opt, beta=beta_opt)

    def get_parameters(self, events: Optional[List[float]] = None, t_end: Optional[float] = None) -> HawkesParameters:
        """Returns structured HawkesParameters dataclass for the current model instance."""
        eta = self.branching_ratio
        stable = self.is_stable
        asympt = self.asymptotic_rate

        log_lik = 0.0
        aic = 0.0
        if events:
            log_lik = self.log_likelihood(events, t_end)
            aic = 2.0 * 3.0 - 2.0 * log_lik

        return HawkesParameters(
            mu0=self.mu0,
            alpha=self.alpha,
            beta=self.beta,
            branching_ratio=round(eta, 4),
            is_stable=stable,
            asymptotic_rate=round(asympt, 4) if stable else float("inf"),
            log_likelihood=round(log_lik, 4),
            aic=round(aic, 4)
        )

    @staticmethod
    def _nelder_mead(func, x0: List[float], max_iter: int = 100) -> List[float]:
        """Compact Nelder-Mead simplex optimization in pure Python standard library."""
        dim = len(x0)
        simplex = [list(x0)]

        for i in range(dim):
            point = list(x0)
            point[i] = point[i] * 1.25 if abs(point[i]) > 1e-6 else 0.1
            simplex.append(point)

        values = [func(p) for p in simplex]

        for _ in range(max_iter):
            # Sort simplex by function value
            indices = sorted(range(len(values)), key=lambda idx: values[idx])
            simplex = [simplex[i] for i in indices]
            values = [values[i] for i in indices]

            # Best, second-worst, and worst points
            best = simplex[0]
            worst = simplex[-1]
            second_worst = simplex[-2]

            # Centroid of all points except worst
            centroid = [
                sum(simplex[i][d] for i in range(dim)) / dim
                for d in range(dim)
            ]

            # 1. Reflection
            reflected = [centroid[d] + 1.0 * (centroid[d] - worst[d]) for d in range(dim)]
            f_reflected = func(reflected)

            if values[0] <= f_reflected < values[-2]:
                simplex[-1] = reflected
                values[-1] = f_reflected
                continue

            # 2. Expansion
            if f_reflected < values[0]:
                expanded = [centroid[d] + 2.0 * (reflected[d] - centroid[d]) for d in range(dim)]
                f_expanded = func(expanded)
                if f_expanded < f_reflected:
                    simplex[-1] = expanded
                    values[-1] = f_expanded
                else:
                    simplex[-1] = reflected
                    values[-1] = f_reflected
                continue

            # 3. Contraction
            contracted = [centroid[d] + 0.5 * (worst[d] - centroid[d]) for d in range(dim)]
            f_contracted = func(contracted)
            if f_contracted < values[-1]:
                simplex[-1] = contracted
                values[-1] = f_contracted
                continue

            # 4. Shrink
            for i in range(1, len(simplex)):
                for d in range(dim):
                    simplex[i][d] = best[d] + 0.5 * (simplex[i][d] - best[d])
                values[i] = func(simplex[i])

        return simplex[0]
