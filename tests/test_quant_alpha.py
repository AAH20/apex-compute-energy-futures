"""High-Performance Unit Test Suite for Quantitative Alpha Models.

Validates:
1. Ornstein-Uhlenbeck Mean-Reversion and Jump-Diffusion parameter estimation,
   half-life correctness, jump filtering, and z-score trading bands.
2. Self-Exciting Hawkes Point Process branching ratio, closed-form intensity,
   log-likelihood evaluation, and next-event conditional arrival probability.
3. NSGA-II Multi-Objective Pareto Evolutionary Engine non-dominated sorting,
   crowding distance computation, SBX crossover, and polynomial mutation.

Designed to execute in < 5ms total using pure Python standard library unittest.
"""

import math
import random
import unittest
from typing import List

from apex_compute_futures.evolution.pareto_tournament import (
    ArchetypeIndividual,
    MultiObjectiveFitness,
    ParetoEvolutionEngine,
)
from apex_compute_futures.predictive.hawkes_process import (
    HawkesParameters,
    HawkesProcess,
    PreemptionShockEvent,
)
from apex_compute_futures.predictive.ou_jump_diffusion import (
    OUJumpDiffusionModel,
    OUParameters,
    SpreadSignal,
)


class TestOUJumpDiffusion(unittest.TestCase):
    """Test suite for Ornstein-Uhlenbeck Jump-Diffusion model."""

    def test_ou_exact_half_life(self) -> None:
        """Verifies exact half-life formula: tau_1/2 = ln(2) / theta."""
        for theta in [0.25, 0.5, 1.0, 2.0, 4.0, 10.0]:
            model = OUJumpDiffusionModel(theta=theta, mu=0.0, sigma=1.0)
            expected_hl = math.log(2.0) / theta
            self.assertAlmostEqual(model.half_life, expected_hl, places=9)
            # Inverse verification: exp(-theta * tau) == 0.5
            self.assertAlmostEqual(math.exp(-theta * model.half_life), 0.5, places=9)

    def test_ou_stationary_volatility_and_z_score(self) -> None:
        """Verifies stationary variance: sigma_inf = sigma / sqrt(2 * theta)."""
        theta = 2.0
        sigma = 2.0
        mu = 5.0
        model = OUJumpDiffusionModel(theta=theta, mu=mu, sigma=sigma)

        # Expected stationary std = 2.0 / sqrt(2 * 2) = 2.0 / 2.0 = 1.0
        expected_stat_vol = 1.0
        self.assertAlmostEqual(model.stationary_volatility, expected_stat_vol, places=7)

        # Z-score verification
        self.assertAlmostEqual(model.calculate_z_score(5.0), 0.0, places=7)
        self.assertAlmostEqual(model.calculate_z_score(7.0), 2.0, places=7)
        self.assertAlmostEqual(model.calculate_z_score(2.0), -3.0, places=7)

    def test_ou_discrete_mle_parameter_estimation(self) -> None:
        """Verifies AR(1) discrete MLE parameter recovery on synthetic trajectory."""
        true_theta = 1.5
        true_mu = 10.0
        true_sigma = 0.5
        dt = 0.2

        # Deterministic simulation with known AR(1) parameters
        # X_i = a * X_{i-1} + b + eps
        a = math.exp(-true_theta * dt)
        b = true_mu * (1.0 - a)
        var_eps = (true_sigma ** 2 / (2.0 * true_theta)) * (1.0 - a * a)
        std_eps = math.sqrt(var_eps)

        random.seed(1)
        x = 10.0
        series = [x]
        for _ in range(150):
            eps = random.gauss(0.0, std_eps)
            x = a * x + b + eps
            series.append(x)

        fitted_model = OUJumpDiffusionModel.fit(series, dt=dt, jump_threshold_std=3.5)
        params = fitted_model.get_parameters(series)

        # Ensure parameters are within tight statistical convergence bounds
        self.assertAlmostEqual(params.mu, true_mu, delta=0.2)
        self.assertAlmostEqual(params.theta, true_theta, delta=0.3)
        self.assertAlmostEqual(params.sigma, true_sigma, delta=0.1)
        self.assertAlmostEqual(params.half_life, math.log(2.0) / params.theta, places=7)

    def test_ou_jump_detection_and_filtering(self) -> None:
        """Verifies outlier residual filtering detects jumps and protects diffusion parameters."""
        theta = 2.0
        mu = 0.0
        sigma = 0.4
        dt = 0.1

        a = math.exp(-theta * dt)
        b = mu * (1.0 - a)
        std_eps = sigma * math.sqrt((1.0 - a * a) / (2.0 * theta))

        random.seed(123)
        series = [0.0]
        for _ in range(80):
            val = a * series[-1] + b + random.gauss(0.0, std_eps)
            series.append(val)

        # Inject 2 massive jumps (e.g., GPU cluster outage and grid trip)
        series[30] += 5.0
        series[60] -= 5.0

        model = OUJumpDiffusionModel.fit(series, dt=dt, jump_threshold_std=3.0)
        params = model.get_parameters(series)

        self.assertGreaterEqual(model.jump_intensity, 0.1)
        self.assertGreater(model.jump_std, 1.0)
        # Equilibrium should remain near 0.0 despite symmetrical jumps
        self.assertAlmostEqual(model.mu, 0.0, delta=0.5)

    def test_ou_trading_signals_and_hysteresis(self) -> None:
        """Verifies optimal entry/exit/stop z-score trading band signals."""
        model = OUJumpDiffusionModel(theta=1.0, mu=0.0, sigma=1.0)
        # stationary volatility = 1 / sqrt(2) ~= 0.7071

        # Synthetic spread trajectory with clear overbought and oversold excursions
        spread_values = [0.0, -1.8, -1.9, -0.2, 0.0, 1.8, 1.9, 0.2, 0.0]
        signals = model.generate_signals(spread_values, entry_z=2.0, exit_z=0.5, stop_z=4.0)

        self.assertEqual(len(signals), len(spread_values))
        # When spread drops below -2.0 z-score, signal becomes Long (+1)
        # -1.8 / 0.7071 = -2.54 -> Long
        self.assertEqual(signals[1].signal, 1)
        self.assertEqual(signals[2].signal, 1)
        # Exits when reverting back above -0.5 z-score
        self.assertEqual(signals[3].signal, 0)
        # Short excursion
        # 1.8 / 0.7071 = 2.54 -> Short (-1)
        self.assertEqual(signals[5].signal, -1)
        self.assertEqual(signals[6].signal, -1)
        # Exits when reverting back below +0.5 z-score
        self.assertEqual(signals[7].signal, 0)

    def test_ou_conditional_prediction(self) -> None:
        """Verifies conditional expectation and variance asymptotic convergence."""
        model = OUJumpDiffusionModel(theta=2.0, mu=5.0, sigma=1.0)
        x0 = 10.0

        # At horizon = 0: E[X_0] = x0, Var[X_0] = 0
        mean_0, var_0 = model.predict_conditional(x0, horizon=0.0)
        self.assertAlmostEqual(mean_0, x0, places=7)
        self.assertAlmostEqual(var_0, 0.0, places=7)

        # At long horizon h -> inf: E[X_inf] -> mu, Var[X_inf] -> stationary_vol^2
        mean_inf, var_inf = model.predict_conditional(x0, horizon=50.0)
        self.assertAlmostEqual(mean_inf, 5.0, places=5)
        self.assertAlmostEqual(var_inf, model.stationary_volatility ** 2, places=5)


class TestHawkesProcess(unittest.TestCase):
    """Test suite for Self-Exciting Hawkes Point Process."""

    def test_hawkes_branching_ratio_and_stability(self) -> None:
        """Verifies branching ratio eta = alpha / beta and sub-critical stability."""
        stable_hawkes = HawkesProcess(mu0=0.2, alpha=0.6, beta=1.0)
        self.assertAlmostEqual(stable_hawkes.branching_ratio, 0.6, places=7)
        self.assertTrue(stable_hawkes.is_stable)
        # Asymptotic stationary rate = mu0 / (1 - eta) = 0.2 / 0.4 = 0.5
        self.assertAlmostEqual(stable_hawkes.asymptotic_rate, 0.5, places=7)

        unstable_hawkes = HawkesProcess(mu0=0.2, alpha=1.2, beta=1.0)
        self.assertAlmostEqual(unstable_hawkes.branching_ratio, 1.2, places=7)
        self.assertFalse(unstable_hawkes.is_stable)
        self.assertEqual(unstable_hawkes.asymptotic_rate, float("inf"))

    def test_hawkes_intensity_evaluation_and_decay(self) -> None:
        """Verifies recursive intensity evaluation and exact exponential decay."""
        mu0 = 0.1
        alpha = 0.5
        beta = 2.0
        hawkes = HawkesProcess(mu0=mu0, alpha=alpha, beta=beta)

        events = [1.0, 1.5, 3.0]
        shock_events = hawkes.compute_event_intensities(events)

        self.assertEqual(len(shock_events), 3)

        # Event 0 at t=1.0:
        # Before event: lambda(1.0^-) = mu0 = 0.1
        # After event: lambda(1.0^+) = mu0 + alpha = 0.6
        self.assertAlmostEqual(shock_events[0].intensity_before, 0.1, places=7)
        self.assertAlmostEqual(shock_events[0].intensity_after, 0.6, places=7)

        # Event 1 at t=1.5 (dt = 0.5):
        # Decayed excitation: alpha * exp(-beta * 0.5) = 0.5 * exp(-1.0) = 0.1839397
        # Before event: mu0 + 0.1839397 = 0.2839397
        expected_lambda_1 = mu0 + alpha * math.exp(-beta * 0.5)
        self.assertAlmostEqual(shock_events[1].intensity_before, expected_lambda_1, places=6)
        self.assertAlmostEqual(shock_events[1].intensity_after, expected_lambda_1 + alpha, places=6)

        # Intensity at arbitrary time t = 2.0 (no new events between 1.5 and 2.0)
        # Intensity must match exact summation
        intensity_t2 = hawkes.intensity_at(2.0, events)
        expected_t2 = mu0 + alpha * math.exp(-beta * (2.0 - 1.0)) + alpha * math.exp(-beta * (2.0 - 1.5))
        self.assertAlmostEqual(intensity_t2, expected_t2, places=6)

    def test_hawkes_log_likelihood_calculation(self) -> None:
        """Verifies closed-form log-likelihood matches point process compensator."""
        hawkes = HawkesProcess(mu0=0.2, alpha=0.3, beta=1.0)
        events = [0.5, 1.2, 2.0, 2.8]
        t_end = 3.5

        log_lik = hawkes.log_likelihood(events, t_end=t_end)

        # Calculate manually to independently verify
        # Compensator = mu0 * T + (alpha / beta) * sum(1 - exp(-beta * (T - ti)))
        compensator = 0.2 * t_end + (0.3 / 1.0) * sum(1.0 - math.exp(-1.0 * (t_end - ti)) for ti in events)

        # Sum of log intensities
        intensities = [
            hawkes.intensity_at(ti, [e for e in events if e < ti])
            for ti in events
        ]
        sum_log_lambda = sum(math.log(lam) for lam in intensities)
        expected_log_lik = sum_log_lambda - compensator

        self.assertAlmostEqual(log_lik, expected_log_lik, places=5)

    def test_hawkes_next_event_conditional_probability(self) -> None:
        """Verifies conditional preemption probability bounds and monotonicity."""
        hawkes = HawkesProcess(mu0=0.1, alpha=0.8, beta=2.0)
        events = [1.0]

        # Probability at horizon 0 is 0
        p_0 = hawkes.next_event_probability(current_time=1.0, horizon=0.0, history=events)
        self.assertEqual(p_0, 0.0)

        # Monotonically increasing in forecast horizon
        p_short = hawkes.next_event_probability(current_time=1.0, horizon=0.2, history=events)
        p_med = hawkes.next_event_probability(current_time=1.0, horizon=1.0, history=events)
        p_long = hawkes.next_event_probability(current_time=1.0, horizon=10.0, history=events)

        self.assertGreater(p_short, 0.0)
        self.assertGreater(p_med, p_short)
        self.assertGreater(p_long, p_med)
        self.assertLess(p_long, 1.0)

        # Immediately after event (t=1.001) probability is significantly higher than before event
        p_before = hawkes.next_event_probability(current_time=0.99, horizon=0.1, history=[])
        p_after = hawkes.next_event_probability(current_time=1.001, horizon=0.1, history=events)
        self.assertGreater(p_after, p_before)

    def test_hawkes_simulation_monotonicity(self) -> None:
        """Verifies Ogata thinning simulation yields strictly increasing events."""
        hawkes = HawkesProcess(mu0=0.5, alpha=0.3, beta=1.5)
        events = hawkes.simulate(t_end=15.0, seed=99)

        self.assertGreater(len(events), 0)
        for i in range(1, len(events)):
            self.assertGreater(events[i], events[i - 1])
            self.assertLessEqual(events[i], 15.0)


class TestParetoEvolutionaryEngine(unittest.TestCase):
    """Test suite for NSGA-II Multi-Objective Pareto Evolutionary Engine."""

    def setUp(self) -> None:
        self.engine = ParetoEvolutionEngine()

    def test_multi_objective_fitness_pareto_dominance(self) -> None:
        """Verifies 4D Pareto dominance relations: DSR, IR, -MaxDD, -SLARisk."""
        # Higher is strictly better for all 4 objectives
        ind_superior = MultiObjectiveFitness(
            deflated_sharpe_ratio=0.85,
            information_ratio=2.20,
            neg_max_drawdown=-0.05,
            neg_sla_breach_risk=-0.0001
        )
        ind_inferior = MultiObjectiveFitness(
            deflated_sharpe_ratio=0.70,
            information_ratio=1.50,
            neg_max_drawdown=-0.12,
            neg_sla_breach_risk=-0.0010
        )
        ind_tradeoff = MultiObjectiveFitness(
            deflated_sharpe_ratio=0.90,  # Superior DSR
            information_ratio=1.20,      # Inferior IR
            neg_max_drawdown=-0.08,      # Inferior Drawdown
            neg_sla_breach_risk=-0.0001  # Equal SLA
        )

        self.assertTrue(ind_superior.dominates(ind_inferior))
        self.assertFalse(ind_inferior.dominates(ind_superior))

        # Tradeoff neither dominates nor is dominated by superior
        self.assertFalse(ind_superior.dominates(ind_tradeoff))
        self.assertFalse(ind_tradeoff.dominates(ind_superior))

    def test_fast_non_dominated_sort(self) -> None:
        """Verifies assignment of rank 1 to Pareto boundary and rank 2 to dominated layer."""
        p1 = ArchetypeIndividual(
            genes={"ou_entry_z": 2.0},
            fitness=MultiObjectiveFitness(0.9, 2.0, -0.05, -0.001)
        )
        p2 = ArchetypeIndividual(
            genes={"ou_entry_z": 2.2},
            fitness=MultiObjectiveFitness(0.7, 2.5, -0.04, -0.001)  # Tradeoff with p1
        )
        p3 = ArchetypeIndividual(
            genes={"ou_entry_z": 1.5},
            fitness=MultiObjectiveFitness(0.5, 1.0, -0.15, -0.005)  # Dominated by both
        )

        population = [p1, p2, p3]
        fronts = self.engine.fast_non_dominated_sort(population)

        self.assertEqual(len(fronts), 2)
        # Front 1 contains non-dominated tradeoff pair [p1, p2]
        self.assertEqual(len(fronts[0]), 2)
        self.assertEqual(p1.rank, 1)
        self.assertEqual(p2.rank, 1)

        # Front 2 contains dominated individual p3
        self.assertEqual(len(fronts[1]), 1)
        self.assertEqual(p3.rank, 2)

    def test_crowding_distance_computation(self) -> None:
        """Verifies boundary points receive inf and interior receives normalized spacing."""
        p1 = ArchetypeIndividual(
            genes={"d": 0.2},
            fitness=MultiObjectiveFitness(0.2, 1.0, -0.20, -0.01)
        )
        p2 = ArchetypeIndividual(
            genes={"d": 0.4},
            fitness=MultiObjectiveFitness(0.5, 1.5, -0.10, -0.005)
        )
        p3 = ArchetypeIndividual(
            genes={"d": 0.6},
            fitness=MultiObjectiveFitness(0.8, 2.0, -0.05, -0.001)
        )

        front = [p1, p2, p3]
        self.engine.compute_crowding_distance(front)

        # Boundaries must have infinite crowding distance
        self.assertEqual(front[0].crowding_distance, float("inf"))
        self.assertEqual(front[-1].crowding_distance, float("inf"))
        # Interior point receives positive finite distance
        self.assertGreater(front[1].crowding_distance, 0.0)
        self.assertLess(front[1].crowding_distance, float("inf"))

    def test_crowded_tournament_selection(self) -> None:
        """Verifies rank priority and crowding distance tie-breaking."""
        p_rank1 = ArchetypeIndividual(genes={}, rank=1, crowding_distance=1.0)
        p_rank2 = ArchetypeIndividual(genes={}, rank=2, crowding_distance=5.0)

        # Rank wins over crowding distance
        winner_rank = self.engine.tournament_select(p_rank1, p_rank2)
        self.assertEqual(winner_rank, p_rank1)

        # Same rank: higher crowding distance wins
        p_rank1_sparse = ArchetypeIndividual(genes={}, rank=1, crowding_distance=8.0)
        winner_crowd = self.engine.tournament_select(p_rank1, p_rank1_sparse)
        self.assertEqual(winner_crowd, p_rank1_sparse)

    def test_simulated_binary_crossover_and_mutation_bounds(self) -> None:
        """Verifies SBX crossover and polynomial mutation strictly respect gene bounds."""
        parent_a = ArchetypeIndividual(genes={"ou_entry_z": 1.2, "d_fractional": 0.20})
        parent_b = ArchetypeIndividual(genes={"ou_entry_z": 3.0, "d_fractional": 0.70})

        random.seed(777)
        for _ in range(25):
            c1, c2 = self.engine.simulated_binary_crossover(parent_a, parent_b)
            c1 = self.engine.polynomial_mutation(c1)
            c2 = self.engine.polynomial_mutation(c2)

            for child in [c1, c2]:
                for gene, val in child.genes.items():
                    low, high = self.engine.bounds[gene]
                    self.assertGreaterEqual(val, low)
                    self.assertLessEqual(val, high)

    def test_nsga2_step_generation_elitism(self) -> None:
        """Verifies generational step preserves population size and retains Pareto front."""
        pop: List[ArchetypeIndividual] = [
            ArchetypeIndividual(
                genes={"ou_entry_z": 1.0 + 0.2 * i, "d_fractional": 0.2 + 0.05 * i}
            )
            for i in range(8)
        ]

        def dummy_evaluator(ind: ArchetypeIndividual) -> MultiObjectiveFitness:
            z = ind.genes.get("ou_entry_z", 2.0)
            d = ind.genes.get("d_fractional", 0.4)
            return MultiObjectiveFitness(
                deflated_sharpe_ratio=min(1.0, max(0.0, z * 0.25)),
                information_ratio=z * 0.5 + d,
                neg_max_drawdown=-(0.20 - min(0.15, d * 0.2)),
                neg_sla_breach_risk=-0.001
            )

        new_pop = self.engine.step_generation(pop, dummy_evaluator)
        self.assertEqual(len(new_pop), len(pop))

        # Pareto front must exist and be rank 1
        pareto_front = ParetoEvolutionEngine.get_pareto_front(new_pop)
        self.assertGreater(len(pareto_front), 0)
        for sol in pareto_front:
            self.assertEqual(sol.rank, 1)


if __name__ == "__main__":
    unittest.main()
