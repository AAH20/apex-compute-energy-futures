r"""Ornstein-Uhlenbeck (O-U) Mean-Reversion and Jump-Diffusion Engine.

Models compute basis spreads (Futures - Spot) and spark spreads (Power - Fuel)
via continuous-time mean-reverting jump-diffusion stochastic differential equations:
    dX_t = \theta (\mu - X_t) dt + \sigma dW_t + J_t dq_t

Features:
- Discrete Maximum Likelihood Estimation (MLE) of \theta, \mu, and \sigma.
- Exact continuous-discrete mapping via AR(1) calibration.
- Analytical half-life calculation: \tau_{1/2} = ln(2) / \theta.
- Stationary standard deviation: \sigma_\infty = \sigma / \sqrt{2\theta}.
- Clewlow-Strickland residual jump detection, intensity, and parameter estimation.
- Optimal z-score entry, exit, and stop-loss trading bands.
- Conditional multi-horizon forecasting (expectation and variance).
- Pure Python standard library implementation with zero third-party dependencies.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class OUParameters:
    r"""Calibrated parameters of the Ornstein-Uhlenbeck jump-diffusion process."""
    theta: float                     # Mean-reversion velocity (\theta > 0)
    mu: float                        # Long-term equilibrium level
    sigma: float                     # Continuous diffusion volatility (\sigma > 0)
    half_life: float                 # Reversion half-life: ln(2) / \theta
    stationary_volatility: float     # Asymptotic std: \sigma / \sqrt{2\theta}
    dt: float = 1.0                  # Observation interval
    jump_intensity: float = 0.0      # Jump arrival rate per unit time (\lambda_J)
    jump_mean: float = 0.0           # Mean jump amplitude (\mu_J)
    jump_std: float = 0.0            # Jump amplitude std (\sigma_J)
    num_jumps: int = 0               # Total identified jump occurrences
    log_likelihood: float = 0.0      # Maximum log-likelihood of calibrated model
    aic: float = 0.0                 # Akaike Information Criterion


@dataclass
class SpreadSignal:
    """Trading signal generated from O-U equilibrium z-score bands."""
    index: int
    spread_value: float
    z_score: float
    signal: int                      # +1: Long spread, -1: Short spread, 0: Neutral/Exit
    expected_mean: float
    is_jump: bool = False


class OUJumpDiffusionModel:
    """Calibrates and evaluates continuous Ornstein-Uhlenbeck Jump-Diffusion processes."""

    def __init__(
        self,
        theta: float = 1.0,
        mu: float = 0.0,
        sigma: float = 1.0,
        dt: float = 1.0,
        jump_intensity: float = 0.0,
        jump_mean: float = 0.0,
        jump_std: float = 0.0
    ):
        if theta <= 0.0:
            raise ValueError(f"theta must be strictly positive, got {theta}")
        if sigma <= 0.0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")
        if dt <= 0.0:
            raise ValueError(f"dt must be strictly positive, got {dt}")

        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    @property
    def half_life(self) -> float:
        r"""Calculates exact half-life of mean-reversion: \tau_{1/2} = ln(2) / \theta."""
        return math.log(2.0) / self.theta

    @property
    def stationary_volatility(self) -> float:
        r"""Calculates asymptotic stationary standard deviation: \sigma_\infty = \sigma / \sqrt{2\theta}."""
        return self.sigma / math.sqrt(2.0 * self.theta)

    def calculate_z_score(self, x: float) -> float:
        """Calculates standardized z-score deviation from long-term equilibrium."""
        stat_vol = self.stationary_volatility
        if stat_vol < 1e-12:
            return 0.0
        return (x - self.mu) / stat_vol

    def predict_conditional(self, x0: float, horizon: float) -> Tuple[float, float]:
        """Calculates conditional expectation and variance at t + horizon given X_t = x0.
        
        Returns:
            Tuple of (E[X_{t+h} | X_t], Var(X_{t+h} | X_t))
        """
        if horizon < 0.0:
            raise ValueError(f"horizon must be non-negative, got {horizon}")
        decay = math.exp(-self.theta * horizon)
        cond_mean = self.mu + (x0 - self.mu) * decay
        cond_var = (self.sigma ** 2 / (2.0 * self.theta)) * (1.0 - math.exp(-2.0 * self.theta * horizon))
        return cond_mean, cond_var

    @classmethod
    def fit(
        cls,
        series: List[float],
        dt: float = 1.0,
        jump_threshold_std: float = 3.0
    ) -> "OUJumpDiffusionModel":
        r"""Fits O-U jump-diffusion parameters via Discrete Maximum Likelihood (MLE).
        
        Uses an exact discrete AR(1) mapping:
            X_{i} = a X_{i-1} + b + \epsilon_i
        with jump identification and iterative filtering to isolate pure continuous diffusion.
        """
        n = len(series)
        if n < 4:
            raise ValueError(f"At least 4 observations required to calibrate O-U model, got {n}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        # Stage 1: Preliminary OLS AR(1) calibration
        x_lag = series[:-1]
        x_lead = series[1:]
        m = len(x_lag)

        sum_x = sum(x_lag)
        sum_y = sum(x_lead)
        sum_xx = sum(x * x for x in x_lag)
        sum_yy = sum(y * y for y in x_lead)
        sum_xy = sum(x * y for x, y in zip(x_lag, x_lead))

        denom = (m * sum_xx - sum_x * sum_x)
        if abs(denom) < 1e-14:
            a = 0.999
            b = sum_y / m * (1.0 - a)
        else:
            a = (m * sum_xy - sum_x * sum_y) / denom
            b = (sum_y - a * sum_x) / m

        # Compute initial residuals
        residuals = [x_lead[i] - (a * x_lag[i] + b) for i in range(m)]
        res_mean = sum(residuals) / m
        res_var = sum((r - res_mean) ** 2 for r in residuals) / max(1, m - 1)
        res_std = math.sqrt(max(1e-12, res_var))

        # Stage 2: Jump identification via outlier residual thresholding
        jump_threshold = jump_threshold_std * res_std
        clean_x_lag: List[float] = []
        clean_x_lead: List[float] = []
        jump_sizes: List[float] = []

        for i in range(m):
            if abs(residuals[i]) > jump_threshold:
                jump_sizes.append(residuals[i])
            else:
                clean_x_lag.append(x_lag[i])
                clean_x_lead.append(x_lead[i])

        # Stage 3: Refined AR(1) calibration on clean transitions
        m_clean = len(clean_x_lag)
        if m_clean >= 3:
            s_x = sum(clean_x_lag)
            s_y = sum(clean_x_lead)
            s_xx = sum(x * x for x in clean_x_lag)
            s_yy = sum(y * y for y in clean_x_lead)
            s_xy = sum(x * y for x, y in zip(clean_x_lag, clean_x_lead))
            d = (m_clean * s_xx - s_x * s_x)
            if abs(d) > 1e-14:
                a_clean = (m_clean * s_xy - s_x * s_y) / d
                b_clean = (s_y - a_clean * s_x) / m_clean
            else:
                a_clean = a
                b_clean = b
            clean_res = [clean_x_lead[i] - (a_clean * clean_x_lag[i] + b_clean) for i in range(m_clean)]
            sigma_eps_sq = sum(r * r for r in clean_res) / m_clean
        else:
            a_clean = a
            b_clean = b
            sigma_eps_sq = res_var

        # Regularize a to enforce mean-reverting stationarity: 0 < a < 1
        a_clamped = max(1e-6, min(0.999999, a_clean))

        # Stage 4: Map AR(1) to continuous O-U parameters
        theta = -math.log(a_clamped) / dt
        mu = b_clean / (1.0 - a_clamped)

        # Discrete variance \sigma_\epsilon^2 = \frac{\sigma^2}{2\theta}(1 - e^{-2\theta dt}) = \frac{\sigma^2}{2\theta}(1 - a^2)
        one_minus_a2 = max(1e-12, 1.0 - (a_clamped ** 2))
        sigma_sq = (sigma_eps_sq * 2.0 * theta) / one_minus_a2
        sigma = math.sqrt(max(1e-12, sigma_sq))

        # Jump statistics
        total_time = m * dt
        num_jumps = len(jump_sizes)
        jump_intensity = num_jumps / total_time if total_time > 0 else 0.0
        if num_jumps > 0:
            jump_mean = sum(jump_sizes) / num_jumps
            jump_var = sum((j - jump_mean) ** 2 for j in jump_sizes) / num_jumps
            jump_std = math.sqrt(max(1e-12, jump_var))
        else:
            jump_mean = 0.0
            jump_std = 0.0

        model = cls(
            theta=theta,
            mu=mu,
            sigma=sigma,
            dt=dt,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_std=jump_std
        )
        return model

    def get_parameters(self, series: Optional[List[float]] = None) -> OUParameters:
        """Returns structured OUParameters dataclass for the current model instance."""
        half_life = self.half_life
        stat_vol = self.stationary_volatility

        log_lik = 0.0
        aic = 0.0
        if series is not None and len(series) >= 2:
            m = len(series) - 1
            a = math.exp(-self.theta * self.dt)
            b = self.mu * (1.0 - a)
            sigma_eps_sq = (self.sigma ** 2 / (2.0 * self.theta)) * (1.0 - a * a)
            sigma_eps_sq = max(1e-12, sigma_eps_sq)

            sse = 0.0
            for i in range(m):
                cond_mean = a * series[i] + b
                sse += (series[i + 1] - cond_mean) ** 2

            log_lik = -0.5 * m * math.log(2.0 * math.pi * sigma_eps_sq) - (sse / (2.0 * sigma_eps_sq))
            k_params = 3 + (3 if self.jump_intensity > 0 else 0)
            aic = 2.0 * k_params - 2.0 * log_lik

        return OUParameters(
            theta=self.theta,
            mu=self.mu,
            sigma=self.sigma,
            half_life=half_life,
            stationary_volatility=stat_vol,
            dt=self.dt,
            jump_intensity=self.jump_intensity,
            jump_mean=self.jump_mean,
            jump_std=self.jump_std,
            num_jumps=int(round(self.jump_intensity * (len(series) - 1) * self.dt)) if series else 0,
            log_likelihood=round(log_lik, 4),
            aic=round(aic, 4)
        )

    def generate_signals(
        self,
        series: List[float],
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0
    ) -> List[SpreadSignal]:
        """Generates mean-reverting algorithmic spread signals with hysteresis.
        
        Strategy:
        - Flat and Z > entry_z: Short spread (expecting contraction down to mu)
        - Flat and Z < -entry_z: Long spread (expecting expansion up to mu)
        - Long position: Exit when Z >= -exit_z or stop-loss when Z <= -stop_z
        - Short position: Exit when Z <= exit_z or stop-loss when Z >= stop_z
        """
        signals: List[SpreadSignal] = []
        current_pos = 0  # 0: flat, 1: long, -1: short
        stat_vol = self.stationary_volatility

        for idx, val in enumerate(series):
            z = (val - self.mu) / stat_vol if stat_vol > 1e-12 else 0.0
            is_jump = abs(z) >= stop_z

            if current_pos == 0:
                if z <= -entry_z:
                    current_pos = 1
                elif z >= entry_z:
                    current_pos = -1
            elif current_pos == 1:
                # Check exit or stop
                if z >= -exit_z or z <= -stop_z:
                    current_pos = 0
            elif current_pos == -1:
                # Check exit or stop
                if z <= exit_z or z >= stop_z:
                    current_pos = 0

            signals.append(
                SpreadSignal(
                    index=idx,
                    spread_value=val,
                    z_score=round(z, 4),
                    signal=current_pos,
                    expected_mean=self.mu,
                    is_jump=is_jump
                )
            )

        return signals

    def simulate(
        self,
        x0: float,
        steps: int,
        seed: Optional[int] = None
    ) -> List[float]:
        """Simulates sample path of the O-U jump-diffusion process using exact discretization."""
        if seed is not None:
            random.seed(seed)

        a = math.exp(-self.theta * self.dt)
        b = self.mu * (1.0 - a)
        one_minus_a2 = max(0.0, 1.0 - a * a)
        sigma_eps = self.sigma * math.sqrt(one_minus_a2 / (2.0 * self.theta))

        path = [x0]
        curr = x0
        jump_prob = 1.0 - math.exp(-self.jump_intensity * self.dt)

        for _ in range(steps):
            # Diffusion innovation
            z = random.gauss(0.0, 1.0)
            next_val = a * curr + b + sigma_eps * z

            # Jump component
            if self.jump_intensity > 0.0 and random.random() < jump_prob:
                jump = random.gauss(self.jump_mean, max(1e-9, self.jump_std))
                next_val += jump

            path.append(next_val)
            curr = next_val

        return path
