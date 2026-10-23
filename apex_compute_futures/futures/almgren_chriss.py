"""Almgren-Chriss Optimal Execution Engine for Spot Liquidation and Forward Unwinding.

Implements the classical Almgren-Chriss (2000) optimal execution framework balancing
temporary/permanent market impact against inventory price risk and preemption:
    min_{x_k} E[x] + lambda * Var(x)

Yields the closed-form hyperbolic liquidation schedule:
    tau_j = sinh(kappa * (T - t_j)) / sinh(kappa * T)
    where kappa = sqrt(lambda * sigma^2 / eta)

Provides exact closed-form optimal trajectory, expected shortfall, execution variance,
half-life of liquidation, and efficient frontier surface generation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AlmgrenChrissConfig:
    """Execution parameters for compute contract unwinding / liquidation."""
    total_volume: float = 1000.0        # Initial inventory X_0 (contracts / GPU-hrs)
    total_time_horizon: float = 1.0     # Time horizon T (e.g., 1.0 hour or 3600 seconds)
    num_intervals: int = 20             # Number of discrete trading intervals N
    volatility: float = 0.25            # Price volatility sigma ($/unit / sqrt(time))
    temp_impact_eta: float = 2.5e-4     # Temporary market impact coefficient eta ($ / (unit/time))
    perm_impact_gamma: float = 1.0e-5   # Permanent market impact coefficient gamma ($ / unit^2)
    half_spread_epsilon: float = 0.02   # Bid-ask half-spread / transaction fee ($/unit)
    risk_aversion_lambda: float = 1e-4  # Trader risk aversion parameter lambda
    initial_price: float = 2.50         # Starting mark-to-market price S_0 ($/unit)


@dataclass(frozen=True)
class TrajectoryStep:
    """Snapshot at time step t_j along the optimal execution path."""
    step: int
    time_t: float
    inventory_remaining: float
    fraction_remaining: float
    trade_size: float
    trading_rate: float
    expected_price: float
    cumulative_cash_proceeds: float


@dataclass(frozen=True)
class OptimalExecutionSchedule:
    """Comprehensive optimal execution plan, risk metrics, and trajectory."""
    config: AlmgrenChrissConfig
    urgency_kappa: float
    half_life_time: float
    expected_shortfall: float
    execution_variance: float
    execution_std_dev: float
    utility_cost: float
    permanent_impact_cost: float
    temporary_impact_cost: float
    spread_cost: float
    trajectory: List[TrajectoryStep]

    @property
    def is_linear_twap(self) -> bool:
        """Returns True if schedule reduces to linear Time-Weighted Average Price (lambda=0)."""
        return self.urgency_kappa < 1e-8


class AlmgrenChrissEngine:
    """Almgren-Chriss Optimal Execution Engine."""

    def __init__(self, config: Optional[AlmgrenChrissConfig] = None) -> None:
        self.config = config or AlmgrenChrissConfig()

    @staticmethod
    def calculate_kappa(risk_aversion: float, volatility: float, temp_impact_eta: float) -> float:
        """Computes continuous urgency parameter:
        kappa = sqrt(lambda * sigma^2 / eta)
        """
        if risk_aversion <= 0.0 or volatility <= 0.0 or temp_impact_eta <= 0.0:
            return 0.0
        return math.sqrt((risk_aversion * (volatility ** 2)) / temp_impact_eta)

    def calculate_half_life(self, kappa: float, t_horizon: float) -> float:
        """Computes time t_half when inventory reaches 50% of X_0.
        sinh(kappa * (T - t_half)) / sinh(kappa * T) = 0.5
        """
        if kappa < 1e-8:
            return 0.5 * t_horizon
        sinh_term = 0.5 * math.sinh(kappa * t_horizon)
        # asinh(x) = ln(x + sqrt(x^2 + 1))
        arg = math.asinh(sinh_term)
        t_half = t_horizon - (arg / kappa)
        return max(0.0, min(t_horizon, t_half))

    def compute_trajectory(
        self,
        config: Optional[AlmgrenChrissConfig] = None
    ) -> OptimalExecutionSchedule:
        """Computes the closed-form optimal liquidation schedule.
        
        Inventory at interval j:
            x_j = X_0 * sinh(kappa * (T - t_j)) / sinh(kappa * T)
        
        Trade size in interval j:
            n_j = x_{j-1} - x_j
        
        Expected shortfall:
            E[x] = 0.5 * gamma * X_0^2 + epsilon * X_0 + eta * sum(n_j^2 / tau)
        
        Variance:
            Var(x) = sigma^2 * tau * sum(x_j^2)
        """
        cfg = config or self.config
        x_0 = cfg.total_volume
        t_hor = cfg.total_time_horizon
        n_steps = cfg.num_intervals
        tau = t_hor / n_steps
        
        kappa = self.calculate_kappa(cfg.risk_aversion_lambda, cfg.volatility, cfg.temp_impact_eta)
        half_life = self.calculate_half_life(kappa, t_hor)
        
        sinh_kt = math.sinh(kappa * t_hor) if kappa > 1e-8 else 0.0
        
        trajectory: List[TrajectoryStep] = []
        x_prev = x_0
        cum_cash = 0.0
        current_exp_price = cfg.initial_price
        
        # Initial point t_0 = 0.0
        trajectory.append(TrajectoryStep(
            step=0,
            time_t=0.0,
            inventory_remaining=round(x_0, 6),
            fraction_remaining=1.0,
            trade_size=0.0,
            trading_rate=0.0,
            expected_price=round(current_exp_price, 4),
            cumulative_cash_proceeds=0.0
        ))
        
        sum_nj_sq_over_tau = 0.0
        sum_xj_sq = 0.0
        
        for j in range(1, n_steps + 1):
            t_j = j * tau
            
            # Compute x_j
            if j == n_steps:
                x_j = 0.0
            elif kappa < 1e-8:
                # Linear TWAP limit as kappa -> 0
                x_j = x_0 * (1.0 - (t_j / t_hor))
            else:
                rem_time = max(0.0, t_hor - t_j)
                ratio = math.sinh(kappa * rem_time) / sinh_kt
                x_j = x_0 * ratio
                
            x_j = max(0.0, x_j)
            trade_size = max(0.0, x_prev - x_j)
            trading_rate = trade_size / tau
            
            # Impact on expected transaction price
            # Permanent impact accumulates: S_k = S_0 - gamma * (X_0 - x_k)
            # Temporary impact on this execution: - epsilon - eta * (trade_size / tau)
            perm_price_drop = cfg.perm_impact_gamma * (x_0 - x_j)
            temp_price_drop = cfg.half_spread_epsilon + (cfg.temp_impact_eta * trading_rate)
            exec_price = max(0.01, cfg.initial_price - perm_price_drop - temp_price_drop)
            
            cum_cash += trade_size * exec_price
            
            # Accumulate variance and temporary impact
            sum_nj_sq_over_tau += (trade_size ** 2) / tau
            sum_xj_sq += (x_j ** 2)
            
            trajectory.append(TrajectoryStep(
                step=j,
                time_t=round(t_j, 6),
                inventory_remaining=round(x_j, 6),
                fraction_remaining=round(x_j / x_0, 6) if x_0 > 0 else 0.0,
                trade_size=round(trade_size, 6),
                trading_rate=round(trading_rate, 6),
                expected_price=round(exec_price, 4),
                cumulative_cash_proceeds=round(cum_cash, 4)
            ))
            
            x_prev = x_j
            
        # Closed-form costs
        perm_cost = 0.5 * cfg.perm_impact_gamma * (x_0 ** 2)
        spread_cost = cfg.half_spread_epsilon * x_0
        temp_cost = cfg.temp_impact_eta * sum_nj_sq_over_tau
        expected_shortfall = perm_cost + spread_cost + temp_cost
        
        # Execution variance
        variance = (cfg.volatility ** 2) * tau * sum_xj_sq
        std_dev = math.sqrt(variance)
        utility = expected_shortfall + cfg.risk_aversion_lambda * variance
        
        return OptimalExecutionSchedule(
            config=cfg,
            urgency_kappa=round(kappa, 6),
            half_life_time=round(half_life, 6),
            expected_shortfall=round(expected_shortfall, 4),
            execution_variance=round(variance, 4),
            execution_std_dev=round(std_dev, 4),
            utility_cost=round(utility, 4),
            permanent_impact_cost=round(perm_cost, 4),
            temporary_impact_cost=round(temp_cost, 4),
            spread_cost=round(spread_cost, 4),
            trajectory=trajectory
        )

    def compute_efficient_frontier(
        self,
        num_points: int = 25,
        lambda_min: float = 1e-6,
        lambda_max: float = 1e-1
    ) -> List[Tuple[float, float, float, float]]:
        """Computes the Almgren-Chriss efficient frontier (E[x] vs Var(x) tradeoff).
        
        Sweeps risk aversion lambda logarithmically from lambda_min to lambda_max.
        
        Returns:
            List of Tuples: (expected_shortfall, variance, std_dev, lambda_val)
        """
        frontier: List[Tuple[float, float, float, float]] = []
        log_min = math.log10(lambda_min)
        log_max = math.log10(lambda_max)
        step = (log_max - log_min) / max(1, num_points - 1)
        
        for i in range(num_points):
            cur_lambda = 10.0 ** (log_min + i * step)
            # Create subconfig with new lambda
            cfg = AlmgrenChrissConfig(
                total_volume=self.config.total_volume,
                total_time_horizon=self.config.total_time_horizon,
                num_intervals=self.config.num_intervals,
                volatility=self.config.volatility,
                temp_impact_eta=self.config.temp_impact_eta,
                perm_impact_gamma=self.config.perm_impact_gamma,
                half_spread_epsilon=self.config.half_spread_epsilon,
                risk_aversion_lambda=cur_lambda,
                initial_price=self.config.initial_price
            )
            sched = self.compute_trajectory(cfg)
            frontier.append((
                sched.expected_shortfall,
                sched.execution_variance,
                sched.execution_std_dev,
                cur_lambda
            ))
            
        return frontier

    def simulate_realized_execution(
        self,
        schedule: OptimalExecutionSchedule,
        random_seed: Optional[int] = None
    ) -> Dict[str, float]:
        """Simulates realized price path and total cash capture under Geometric/Arithmetic Brownian motion.
        
        S_k = S_{k-1} + sigma * sqrt(tau) * Z_k - gamma * n_k
        """
        if random_seed is not None:
            random.seed(random_seed)
            
        cfg = schedule.config
        tau = cfg.total_time_horizon / cfg.num_intervals
        sigma = cfg.volatility
        gamma = cfg.perm_impact_gamma
        eta = cfg.temp_impact_eta
        eps = cfg.half_spread_epsilon
        
        current_spot = cfg.initial_price
        total_cash = 0.0
        
        for step in schedule.trajectory[1:]:
            n_k = step.trade_size
            z_k = random.gauss(0.0, 1.0)
            
            # Browninan motion step + permanent impact
            current_spot += sigma * math.sqrt(tau) * z_k - gamma * n_k
            
            # Execution price with temporary impact and spread
            exec_price = current_spot - eps - eta * (n_k / tau)
            total_cash += n_k * exec_price
            
        book_value = cfg.total_volume * cfg.initial_price
        realized_shortfall = book_value - total_cash
        realized_vwap = total_cash / cfg.total_volume if cfg.total_volume > 0 else cfg.initial_price
        
        return {
            "initial_book_value": round(book_value, 2),
            "realized_cash_capture": round(total_cash, 2),
            "realized_shortfall": round(realized_shortfall, 2),
            "expected_shortfall": schedule.expected_shortfall,
            "realized_vwap": round(realized_vwap, 4),
            "final_spot_price": round(current_spot, 4)
        }
