r"""Continuous-Time Hidden Markov Model (HMM) & Recursive Hamilton Filter.

Detects macro and micro market regimes in coupled compute-energy markets:
    0: Normal Contango (low volatility, stable positive forward basis, calm power grid)
    1: Power Grid Stress (LMP price spikes, extreme volatility, negative spark spread / backwardation)
    2: Compute Scarcity Crunch (preemption cascades, massive GPU spot premiums, capacity hoarding)

Features:
- Continuous-time Markov jump process governed by infinitesimal generator Q.
- Exact matrix exponential mapping P(t) = exp(Qt) via scaling and squaring Taylor expansion.
- Analytical stationary distribution solver satisfying \pi_\infty Q = 0 and \sum \pi_{\infty, k} = 1.
- Online recursive Hamilton (1989) filter updating state probabilities \pi_t in O(1) time per tick.
- Regime-conditional expectation and variance calculators (Law of Total Variance).
- Pure Python standard library implementation with zero third-party dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple, Union


class MarketRegime(IntEnum):
    """Discrete market regimes for compute-energy futures and spot markets."""

    NORMAL_CONTANGO = 0          # Low volatility, stable forward basis, normal power grid
    POWER_GRID_STRESS = 1        # LMP spikes, high power volatility, backwardation / curtailment
    COMPUTE_SCARCITY_CRUNCH = 2  # Spot GPU preemption cascades, severe compute premium


@dataclass(frozen=True)
class RegimeState:
    """Statistical parameters characterizing market observables under a specific regime."""

    regime: MarketRegime
    name: str
    basis_mean: float            # Expected compute basis (Futures - Spot, $/hr or $/PF-hr)
    basis_std: float             # Basis volatility (\sigma_{basis})
    lmp_mean: float              # Expected Locational Marginal Price ($/MWh)
    lmp_std: float               # Power LMP volatility (\sigma_{lmp})
    preemption_mean: float       # Expected spot GPU preemption rate [0, 1]
    preemption_std: float        # Preemption rate volatility


@dataclass(frozen=True)
class HamiltonFilterState:
    """Current state estimate from the online recursive Hamilton filter."""

    probabilities: Tuple[float, float, float]
    most_likely_regime: MarketRegime
    log_likelihood: float                  # Step marginal log-likelihood ln P(y_t | F_{t-1})
    cumulative_log_likelihood: float       # Accumulated log-likelihood across time series
    entropy: float                         # Shannon uncertainty entropy: -\sum \pi_k ln(\pi_k)
    conditional_basis_mean: float          # E[Basis_t | F_t]
    conditional_basis_variance: float      # Var(Basis_t | F_t) via Law of Total Variance
    timestamp: float                       # Elapsed time or tick index


# Default continuous-time transition rate generator matrix Q (rates per hour).
# Row sums are strictly 0.0. Off-diagonals q_{ij} >= 0 represent transition intensity i -> j.
DEFAULT_GENERATOR_MATRIX: List[List[float]] = [
    [-0.07,  0.05,  0.02],   # From NORMAL: 5% to Grid Stress, 2% to Compute Crunch
    [ 0.40, -0.45,  0.05],   # From GRID STRESS: 40% back to Normal, 5% to Crunch
    [ 0.25,  0.05, -0.30],   # From COMPUTE CRUNCH: 25% back to Normal, 5% to Grid Stress
]

# Canonical regime definitions for sovereign compute-energy arbitrage
DEFAULT_REGIME_STATES: List[RegimeState] = [
    RegimeState(
        regime=MarketRegime.NORMAL_CONTANGO,
        name="Normal Contango",
        basis_mean=2.50,         # Stable contango premium
        basis_std=1.20,
        lmp_mean=35.0,           # Baseline ERCOT/PJM wholesale power price
        lmp_std=8.0,
        preemption_mean=0.01,    # Minimal spot preemption (< 1%)
        preemption_std=0.005,
    ),
    RegimeState(
        regime=MarketRegime.POWER_GRID_STRESS,
        name="Power Grid Stress",
        basis_mean=-18.00,       # Severe backwardation as high power prices force curtailment
        basis_std=8.50,
        lmp_mean=450.0,          # LMP spikes to $450-$2000+/MWh
        lmp_std=150.0,
        preemption_mean=0.06,    # Secondary preemptions from emergency shutdowns
        preemption_std=0.03,
    ),
    RegimeState(
        regime=MarketRegime.COMPUTE_SCARCITY_CRUNCH,
        name="Compute Scarcity Crunch",
        basis_mean=48.00,        # Massive spot premium / basis spike due to cluster demand
        basis_std=14.00,
        lmp_mean=42.0,           # Normal grid condition
        lmp_std=12.0,
        preemption_mean=0.38,    # 38% spot instances preempted in cluster buyouts
        preemption_std=0.09,
    ),
]


def _mat3_mult(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> List[List[float]]:
    """Multiplies two 3x3 matrices in pure Python."""
    return [
        [
            A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j]
            for j in range(3)
        ]
        for i in range(3)
    ]


def _mat3_vec_mult(A: Sequence[Sequence[float]], v: Sequence[float]) -> Tuple[float, float, float]:
    """Computes matrix-vector product A * v for 3x3 matrix and 3x1 vector."""
    return (
        A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
        A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
        A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2],
    )


def _vec_mat3_mult(v: Sequence[float], A: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    """Computes vector-matrix product v * A for 1x3 vector and 3x3 matrix."""
    return (
        v[0] * A[0][0] + v[1] * A[1][0] + v[2] * A[2][0],
        v[0] * A[0][1] + v[1] * A[1][1] + v[2] * A[2][1],
        v[0] * A[0][2] + v[1] * A[1][2] + v[2] * A[2][2],
    )


class ContinuousTimeHMM:
    """Continuous-Time Hidden Markov Model with matrix exponential transition generator."""

    def __init__(
        self,
        generator_matrix: Optional[Sequence[Sequence[float]]] = None,
        regimes: Optional[Sequence[RegimeState]] = None,
    ) -> None:
        """Initializes the continuous-time HMM.

        Args:
            generator_matrix: 3x3 infinitesimal generator Q where off-diagonals >= 0
                and row sums equal 0.0. Defaults to compute-energy market rates.
            regimes: List of 3 RegimeState definitions. Defaults to standard market regimes.
        """
        raw_q = generator_matrix if generator_matrix is not None else DEFAULT_GENERATOR_MATRIX
        if len(raw_q) != 3 or any(len(row) != 3 for row in raw_q):
            raise ValueError(f"generator_matrix must be 3x3, got {len(raw_q)} rows")

        # Validate generator matrix row stochasticity (sum = 0)
        self._q: List[List[float]] = []
        for i in range(3):
            row = [float(val) for val in raw_q[i]]
            row_sum = sum(row)
            if abs(row_sum) > 1e-5:
                # Project diagonal so row sums strictly to 0
                row[i] -= row_sum
            self._q.append(row)

        raw_regimes = regimes if regimes is not None else DEFAULT_REGIME_STATES
        if len(raw_regimes) != 3:
            raise ValueError(f"regimes must contain exactly 3 states, got {len(raw_regimes)}")
        self._regimes = list(raw_regimes)

        # Precompute stationary distribution
        self._pi_inf = self._solve_stationary_distribution()

    @property
    def generator_matrix(self) -> List[List[float]]:
        """Returns the infinitesimal generator rate matrix Q."""
        return [list(row) for row in self._q]

    @property
    def regimes(self) -> List[RegimeState]:
        """Returns the configured regime states."""
        return list(self._regimes)

    def transition_matrix(self, dt: float) -> List[List[float]]:
        r"""Calculates finite-time transition probability matrix P(dt) = exp(Q * dt).

        Employs scaling and squaring with high-order Taylor series expansion:
            P(dt) = [exp( (Q * dt) / 2^s )]^(2^s)
        Guarantees non-negative probabilities and row-stochastic normalization (\sum_j P_{ij} = 1).

        Args:
            dt: Continuous elapsed time interval (\Delta t \ge 0).

        Returns:
            3x3 row-stochastic transition probability matrix.
        """
        if dt < 0.0:
            raise ValueError(f"dt must be non-negative, got {dt}")
        if dt == 0.0:
            return [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]

        # Form scaled matrix A = Q * dt
        A = [[self._q[i][j] * dt for j in range(3)] for i in range(3)]

        # Infinity norm (maximum absolute row sum)
        norm_inf = max(sum(abs(A[i][j]) for j in range(3)) for i in range(3))

        # Scaling power s such that ||A / 2^s|| <= 0.5
        s = 0
        if norm_inf > 0.5:
            s = max(0, int(math.ceil(math.log2(norm_inf / 0.5))))

        scale = 1.0 / (2.0 ** s)
        B = [[A[i][j] * scale for j in range(3)] for i in range(3)]

        # Taylor series: exp(B) = I + B + B^2/2! + ... + B^M/M! (M = 16)
        res = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
        term = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]

        for k in range(1, 17):
            term = _mat3_mult(term, B)
            inv_k = 1.0 / k
            for i in range(3):
                for j in range(3):
                    term[i][j] *= inv_k
                    res[i][j] += term[i][j]

        # Squaring step: res = res^(2^s)
        for _ in range(s):
            res = _mat3_mult(res, res)

        # Enforce non-negativity and exact row-stochasticity
        for i in range(3):
            # Clamp numerical precision artifacts < 0
            for j in range(3):
                if res[i][j] < 0.0:
                    res[i][j] = 0.0
            row_sum = sum(res[i])
            if row_sum > 0.0:
                inv_sum = 1.0 / row_sum
                for j in range(3):
                    res[i][j] *= inv_sum
            else:
                res[i] = [1.0 if i == j else 0.0 for j in range(3)]

        return res

    def _solve_stationary_distribution(self) -> Tuple[float, float, float]:
        r"""Analytically solves stationary distribution \pi_\infty Q = 0 with \sum \pi_{\infty, k} = 1.

        Uses the cross product of linearly independent rows of Q^T, which lies in the
        one-dimensional nullspace of Q^T:
            w = (row_0 of Q^T) x (row_1 of Q^T)
        """
        # Rows of Q^T are columns of Q
        v0 = (self._q[0][0], self._q[1][0], self._q[2][0])
        v1 = (self._q[0][1], self._q[1][1], self._q[2][1])

        # Cross product w = v0 x v1
        w0 = v0[1] * v1[2] - v0[2] * v1[1]
        w1 = v0[2] * v1[0] - v0[0] * v1[2]
        w2 = v0[0] * v1[1] - v0[1] * v1[0]

        total = w0 + w1 + w2
        if abs(total) < 1e-14 or (w0 * total < 0 or w1 * total < 0 or w2 * total < 0):
            # Fallback to alternate pair of columns or Gaussian elimination
            v2 = (self._q[0][2], self._q[1][2], self._q[2][2])
            w0 = v1[1] * v2[2] - v1[2] * v2[1]
            w1 = v1[2] * v2[0] - v1[0] * v2[2]
            w2 = v1[0] * v2[1] - v1[1] * v2[0]
            total = w0 + w1 + w2

        if abs(total) < 1e-14:
            # Degenerate case: uniform prior
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        p0 = max(0.0, w0 / total)
        p1 = max(0.0, w1 / total)
        p2 = max(0.0, w2 / total)
        norm = p0 + p1 + p2
        return (p0 / norm, p1 / norm, p2 / norm)

    def stationary_distribution(self) -> Tuple[float, float, float]:
        r"""Returns the ergodic stationary distribution \pi_\infty = [\pi_0, \pi_1, \pi_2]."""
        return self._pi_inf

    def emission_density(
        self,
        regime_idx: int,
        observation: Union[float, Dict[str, float]],
    ) -> float:
        r"""Computes the continuous emission probability density f(y_t | S_t = k).

        Supports both scalar basis observation and multi-observable dictionary
        (e.g., {'basis': -15.0, 'lmp': 380.0, 'preemption': 0.05}).

        Employs log-domain Gaussian likelihood for numerical robustness:
            ln f(y) = -0.5 ln(2\pi) - ln(\sigma) - (y - \mu)^2 / (2\sigma^2)
        """
        regime = self._regimes[regime_idx]
        log_density = 0.0

        if isinstance(observation, (int, float)):
            # Single observable: compute basis spread
            y = float(observation)
            mu = regime.basis_mean
            sigma = max(1e-4, regime.basis_std)
            diff = (y - mu) / sigma
            log_density = -0.5 * math.log(2.0 * math.pi) - math.log(sigma) - 0.5 * diff * diff
        elif isinstance(observation, dict):
            # Multi-feature emission
            if "basis" in observation:
                y = float(observation["basis"])
                mu, sigma = regime.basis_mean, max(1e-4, regime.basis_std)
                diff = (y - mu) / sigma
                log_density += -0.5 * math.log(2.0 * math.pi) - math.log(sigma) - 0.5 * diff * diff
            if "lmp" in observation:
                y = float(observation["lmp"])
                mu, sigma = regime.lmp_mean, max(1e-4, regime.lmp_std)
                diff = (y - mu) / sigma
                log_density += -0.5 * math.log(2.0 * math.pi) - math.log(sigma) - 0.5 * diff * diff
            if "preemption" in observation:
                y = float(observation["preemption"])
                mu, sigma = regime.preemption_mean, max(1e-4, regime.preemption_std)
                diff = (y - mu) / sigma
                log_density += -0.5 * math.log(2.0 * math.pi) - math.log(sigma) - 0.5 * diff * diff
        else:
            raise TypeError(f"observation must be float or dict, got {type(observation)}")

        # Safeguard against extreme floating point underflow
        log_density = max(-500.0, min(500.0, log_density))
        return math.exp(log_density)

    def conditional_expectation(
        self,
        probabilities: Sequence[float],
        feature: str = "basis",
    ) -> float:
        r"""Calculates regime-conditional expected value E[Y | \pi_t] = \sum_k \pi_k \mu_k."""
        if feature == "basis":
            means = [r.basis_mean for r in self._regimes]
        elif feature == "lmp":
            means = [r.lmp_mean for r in self._regimes]
        elif feature == "preemption":
            means = [r.preemption_mean for r in self._regimes]
        else:
            raise ValueError(f"Unknown feature '{feature}'")

        return sum(probabilities[k] * means[k] for k in range(3))

    def conditional_variance(
        self,
        probabilities: Sequence[float],
        feature: str = "basis",
    ) -> float:
        r"""Calculates conditional variance Var(Y | \pi_t) via Law of Total Variance.

        Var(Y | \pi_t) = E[Var(Y | S)] + Var(E[Y | S])
                       = \sum \pi_k \sigma_k^2 + \sum \pi_k (\mu_k - \bar{\mu})^2
        """
        if feature == "basis":
            means = [r.basis_mean for r in self._regimes]
            variances = [r.basis_std ** 2 for r in self._regimes]
        elif feature == "lmp":
            means = [r.lmp_mean for r in self._regimes]
            variances = [r.lmp_std ** 2 for r in self._regimes]
        elif feature == "preemption":
            means = [r.preemption_mean for r in self._regimes]
            variances = [r.preemption_std ** 2 for r in self._regimes]
        else:
            raise ValueError(f"Unknown feature '{feature}'")

        expected_mean = sum(probabilities[k] * means[k] for k in range(3))
        within_regime_var = sum(probabilities[k] * variances[k] for k in range(3))
        between_regime_var = sum(
            probabilities[k] * ((means[k] - expected_mean) ** 2)
            for k in range(3)
        )
        return within_regime_var + between_regime_var

    def create_filter(
        self,
        initial_probabilities: Optional[Tuple[float, float, float]] = None,
    ) -> HamiltonFilter:
        """Instantiates a dedicated recursive Hamilton filter attached to this HMM."""
        return HamiltonFilter(self, initial_probabilities=initial_probabilities)


class HamiltonFilter:
    r"""Online Recursive Hamilton (1989) Filter for market regime detection in O(1) time.

    Maintains the posterior state distribution \pi_t = P(S_t = k | \mathcal{F}_t)
    conditioned on filtration \mathcal{F}_t of past observations.

    Online Cycle:
        1. Time update (Continuous Jump Predict): \pi_{t|t-1} = \pi_{t-1} * exp(Q * dt)
        2. Measurement update (Bayesian Conditioning): \pi_{t, k} \propto \pi_{t|t-1, k} * f(y_t | S_t = k)
    """

    def __init__(
        self,
        hmm: Optional[ContinuousTimeHMM] = None,
        initial_probabilities: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Initializes the Hamilton Filter.

        Args:
            hmm: Underlying ContinuousTimeHMM model. If None, default model is used.
            initial_probabilities: Initial prior distribution. Defaults to stationary distribution.
        """
        self.hmm = hmm if hmm is not None else ContinuousTimeHMM()
        self._pi: Tuple[float, float, float] = (
            initial_probabilities
            if initial_probabilities is not None
            else self.hmm.stationary_distribution()
        )
        self._cumulative_log_likelihood: float = 0.0
        self._current_time: float = 0.0
        self._last_state: HamiltonFilterState = self._build_state(
            log_likelihood=0.0,
            timestamp=0.0,
        )

    @property
    def probabilities(self) -> Tuple[float, float, float]:
        r"""Current posterior regime probabilities \pi_t = [\pi_0, \pi_1, \pi_2]."""
        return self._pi

    @property
    def state(self) -> HamiltonFilterState:
        """Current complete HamiltonFilterState."""
        return self._last_state

    @property
    def most_likely_regime(self) -> MarketRegime:
        r"""Most probable regime argmax_k \pi_{t, k}."""
        if self._pi[0] >= self._pi[1] and self._pi[0] >= self._pi[2]:
            return MarketRegime.NORMAL_CONTANGO
        if self._pi[1] >= self._pi[2]:
            return MarketRegime.POWER_GRID_STRESS
        return MarketRegime.COMPUTE_SCARCITY_CRUNCH

    @property
    def cumulative_log_likelihood(self) -> float:
        """Cumulative log-likelihood across all processed observations."""
        return self._cumulative_log_likelihood

    def _build_state(self, log_likelihood: float, timestamp: float) -> HamiltonFilterState:
        """Helper to construct immutable HamiltonFilterState dataclass."""
        probs = self._pi
        entropy = -sum(
            p * math.log(max(1e-15, p))
            for p in probs
            if p > 0.0
        )
        cond_mean = self.hmm.conditional_expectation(probs, feature="basis")
        cond_var = self.hmm.conditional_variance(probs, feature="basis")

        return HamiltonFilterState(
            probabilities=probs,
            most_likely_regime=self.most_likely_regime,
            log_likelihood=log_likelihood,
            cumulative_log_likelihood=self._cumulative_log_likelihood,
            entropy=entropy,
            conditional_basis_mean=cond_mean,
            conditional_basis_variance=cond_var,
            timestamp=timestamp,
        )

    def predict(self, dt: float) -> Tuple[float, float, float]:
        r"""Prediction step: propagates state distribution across continuous time dt.

        \pi_{t|t-\Delta t} = \pi_{t-\Delta t} * P(\Delta t)
        """
        if dt < 0.0:
            raise ValueError(f"dt must be non-negative, got {dt}")
        if dt == 0.0:
            return self._pi

        P = self.hmm.transition_matrix(dt)
        pi_pred = _vec_mat3_mult(self._pi, P)

        # Normalize predicted probabilities
        total = sum(pi_pred)
        if total > 0.0:
            inv = 1.0 / total
            self._pi = (pi_pred[0] * inv, pi_pred[1] * inv, pi_pred[2] * inv)
        return self._pi

    def update(
        self,
        observation: Union[float, Dict[str, float]],
        dt: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> HamiltonFilterState:
        r"""Performs complete O(1) recursive Hamilton filter step (Predict + Measurement Update).

        Args:
            observation: Observed market basis float, or dictionary with 'basis', 'lmp', 'preemption'.
            dt: Time elapsed since prior tick (\Delta t \ge 0).
            timestamp: Explicit timestamp or tick counter.

        Returns:
            Updated HamiltonFilterState with posterior probabilities.
        """
        if timestamp is not None:
            t = timestamp
        else:
            self._current_time += dt
            t = self._current_time

        # 1. Prediction step via matrix exponential transition
        if dt > 0.0:
            self.predict(dt)

        # 2. Measurement update via emission likelihoods
        f0 = self.hmm.emission_density(0, observation)
        f1 = self.hmm.emission_density(1, observation)
        f2 = self.hmm.emission_density(2, observation)

        # Unnormalized posteriors \tilde{\pi}_k = \pi_k * f(y | S=k)
        u0 = self._pi[0] * f0
        u1 = self._pi[1] * f1
        u2 = self._pi[2] * f2

        marginal_evidence = u0 + u1 + u2

        if marginal_evidence > 1e-300:
            inv_ev = 1.0 / marginal_evidence
            self._pi = (u0 * inv_ev, u1 * inv_ev, u2 * inv_ev)
            step_log_lik = math.log(marginal_evidence)
        else:
            # Numerical underflow safeguard: retain predicted prior
            step_log_lik = -300.0

        self._cumulative_log_likelihood += step_log_lik
        self._last_state = self._build_state(step_log_lik, t)
        return self._last_state

    def step(
        self,
        observation: Union[float, Dict[str, float]],
        dt: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> HamiltonFilterState:
        """Alias for update(). Executes one recursive Hamilton step."""
        return self.update(observation, dt=dt, timestamp=timestamp)

    def filter_sequence(
        self,
        observations: Sequence[Union[float, Dict[str, float]]],
        dts: Optional[Sequence[float]] = None,
    ) -> List[HamiltonFilterState]:
        """Filters an entire sequence of market observations sequentially."""
        states: List[HamiltonFilterState] = []
        n = len(observations)
        for i in range(n):
            dt = dts[i] if dts is not None else 1.0
            st = self.update(observations[i], dt=dt, timestamp=float(i))
            states.append(st)
        return states

    def reset(
        self,
        initial_probabilities: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Resets the filter state back to stationary distribution or custom prior."""
        self._pi = (
            initial_probabilities
            if initial_probabilities is not None
            else self.hmm.stationary_distribution()
        )
        self._cumulative_log_likelihood = 0.0
        self._current_time = 0.0
        self._last_state = self._build_state(log_likelihood=0.0, timestamp=0.0)
