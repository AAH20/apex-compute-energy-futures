import unittest
from apex_compute_futures.evolution.fitness import SystemFitnessCalculator


class TestEvolutionFitness(unittest.TestCase):
    def setUp(self):
        self.calc = SystemFitnessCalculator(max_sla_invalidation_rate=0.001, max_drawdown_limit=0.10)

    def test_fitness_hard_barrier_sla_rejection(self):
        returns = [0.01, 0.02, -0.005, 0.015, 0.01] * 4
        # Breach SLA rate: 0.005 > 0.001
        score = self.calc.evaluate_archetype_fitness(returns, sla_invalidation_rate=0.005, realized_max_drawdown=0.02)
        self.assertEqual(score, float("-inf"))

    def test_fitness_hard_barrier_drawdown_rejection(self):
        returns = [0.01, 0.02, -0.005, 0.015, 0.01] * 4
        # Breach drawdown: 0.15 > 0.10
        score = self.calc.evaluate_archetype_fitness(returns, sla_invalidation_rate=0.0001, realized_max_drawdown=0.15)
        self.assertEqual(score, float("-inf"))

    def test_fitness_positive_for_healthy_archetype(self):
        returns = [0.01, 0.02, -0.005, 0.015, 0.01] * 4
        score = self.calc.evaluate_archetype_fitness(returns, sla_invalidation_rate=0.0001, realized_max_drawdown=0.03)
        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
