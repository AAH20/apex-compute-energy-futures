r"""Multi-Objective Pareto Evolutionary Engine.

Implements an NSGA-II inspired evolutionary algorithm using pure Python standard library.
Optimizes multi-dimensional fitness vectors for compute-energy arbitrage alpha archetypes:
    \mathbf{f}(\mathbf{x}) = \langle \text{DSR}(\mathbf{x}), \text{InformationRatio}(\mathbf{x}), -\text{MaxDrawdown}(\mathbf{x}), -\text{SLABreachRisk}(\mathbf{x}) \rangle

Features:
- Fast Non-Dominated Sorting (O(M * N^2) complexity).
- Crowding Distance Assignment for objective-space diversity preservation.
- Crowded Tournament Selection operator.
- Simulated Binary Crossover (SBX) with polynomial distribution index.
- Polynomial Mutation operator with bounded range enforcement.
- Elitist generational survival replacement.
- Pure Python standard library implementation with zero third-party dependencies.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class MultiObjectiveFitness:
    r"""4-Dimensional fitness vector for compute-energy arbitrage alpha archetypes.
    
    All objective dimensions are formulated such that LARGER is strictly BETTER:
    1. deflated_sharpe_ratio: Bailey & López de Prado DSR probability in [0, 1]
    2. information_ratio: Annualized excess return / tracking error
    3. neg_max_drawdown: -MaxDrawdown in [-1, 0] (e.g., -0.05 is superior to -0.25)
    4. neg_sla_breach_risk: -SLABreachRisk in [-1, 0] (e.g., -0.0001 is superior to -0.01)
    """
    deflated_sharpe_ratio: float
    information_ratio: float
    neg_max_drawdown: float
    neg_sla_breach_risk: float

    def to_vector(self) -> Tuple[float, float, float, float]:
        """Returns the objective values as an ordered numeric tuple."""
        return (
            self.deflated_sharpe_ratio,
            self.information_ratio,
            self.neg_max_drawdown,
            self.neg_sla_breach_risk
        )

    def dominates(self, other: "MultiObjectiveFitness") -> bool:
        r"""Evaluates Pareto dominance: self \succ other.
        
        True iff self is at least as good as other in all objectives,
        and strictly better in at least one objective.
        """
        vec_a = self.to_vector()
        vec_b = other.to_vector()

        greater_or_equal = True
        strictly_greater = False

        for a, b in zip(vec_a, vec_b):
            if a < b:
                return False
            if a > b:
                strictly_greater = True

        return greater_or_equal and strictly_greater


@dataclass
class ArchetypeIndividual:
    """Evolutionary individual representing a compute-energy trading archetype."""
    genes: Dict[str, float]
    fitness: Optional[MultiObjectiveFitness] = None
    rank: int = 0
    crowding_distance: float = 0.0

    def dominates(self, other: "ArchetypeIndividual") -> bool:
        """Pareto dominance comparison between two individuals."""
        if self.fitness is None or other.fitness is None:
            return False
        return self.fitness.dominates(other.fitness)


class ParetoEvolutionEngine:
    """NSGA-II multi-objective evolutionary engine for algorithmic archetypes."""

    DEFAULT_GENE_BOUNDS: Dict[str, Tuple[float, float]] = {
        "d_fractional": (0.10, 0.85),
        "profit_multiplier": (1.00, 6.00),
        "stop_loss_multiplier": (0.50, 3.50),
        "max_holding_horizon": (4.0, 72.0),
        "synthetic_inertia_alpha": (0.20, 0.60),
        "frequency_trip_threshold_hz": (59.50, 59.95),
        "ou_entry_z": (1.00, 3.50),
        "ou_exit_z": (0.10, 1.00),
        "hawkes_shock_threshold": (0.05, 0.80)
    }

    def __init__(
        self,
        gene_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        crossover_rate: float = 0.90,
        mutation_rate: Optional[float] = None,
        eta_crossover: float = 20.0,
        eta_mutation: float = 20.0
    ):
        self.bounds = gene_bounds if gene_bounds is not None else dict(self.DEFAULT_GENE_BOUNDS)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate if mutation_rate is not None else (1.0 / max(1, len(self.bounds)))
        self.eta_c = eta_crossover
        self.eta_m = eta_mutation

    @staticmethod
    def fast_non_dominated_sort(
        population: List[ArchetypeIndividual]
    ) -> List[List[ArchetypeIndividual]]:
        r"""Performs Deb's fast non-dominated sort in O(M * N^2).
        
        Assigns Pareto rank 1 to the boundary front F_1, rank 2 to F_2, etc.
        """
        n = len(population)
        if n == 0:
            return []

        # S_p: list of individuals that individual p dominates
        # n_p: number of individuals that dominate individual p
        dominated_sets: Dict[int, List[int]] = {i: [] for i in range(n)}
        domination_counts: Dict[int, int] = {i: 0 for i in range(n)}
        fronts: List[List[ArchetypeIndividual]] = [[]]

        for p in range(n):
            for q in range(p + 1, n):
                if population[p].dominates(population[q]):
                    dominated_sets[p].append(q)
                    domination_counts[q] += 1
                elif population[q].dominates(population[p]):
                    dominated_sets[q].append(p)
                    domination_counts[p] += 1

            if domination_counts[p] == 0:
                population[p].rank = 1
                fronts[0].append(population[p])

        curr_front_idx = 0
        while curr_front_idx < len(fronts) and len(fronts[curr_front_idx]) > 0:
            next_front: List[ArchetypeIndividual] = []
            for ind_p in fronts[curr_front_idx]:
                p_idx = population.index(ind_p)
                for q_idx in dominated_sets[p_idx]:
                    domination_counts[q_idx] -= 1
                    if domination_counts[q_idx] == 0:
                        population[q_idx].rank = curr_front_idx + 2
                        next_front.append(population[q_idx])

            curr_front_idx += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return [f for f in fronts if f]

    @staticmethod
    def compute_crowding_distance(front: List[ArchetypeIndividual]) -> None:
        """Assigns NSGA-II crowding distances to individuals within a non-dominated front."""
        size = len(front)
        if size == 0:
            return

        for ind in front:
            ind.crowding_distance = 0.0

        if size <= 2:
            for ind in front:
                ind.crowding_distance = float("inf")
            return

        num_objectives = 4
        for m in range(num_objectives):
            front.sort(key=lambda ind: ind.fitness.to_vector()[m] if ind.fitness else 0.0)

            # Boundary points receive infinite crowding distance
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            f_min = front[0].fitness.to_vector()[m] if front[0].fitness else 0.0
            f_max = front[-1].fitness.to_vector()[m] if front[-1].fitness else 0.0
            f_range = f_max - f_min

            if f_range > 1e-12:
                for i in range(1, size - 1):
                    prev_f = front[i - 1].fitness.to_vector()[m] if front[i - 1].fitness else 0.0
                    next_f = front[i + 1].fitness.to_vector()[m] if front[i + 1].fitness else 0.0
                    front[i].crowding_distance += (next_f - prev_f) / f_range

    @staticmethod
    def tournament_select(
        candidate_a: ArchetypeIndividual,
        candidate_b: ArchetypeIndividual
    ) -> ArchetypeIndividual:
        """Executes binary crowded tournament selection."""
        # Rank comparison (lower rank is superior)
        if candidate_a.rank < candidate_b.rank:
            return candidate_a
        if candidate_b.rank < candidate_a.rank:
            return candidate_b

        # Tie-breaker: Crowding distance (larger distance is superior)
        if candidate_a.crowding_distance > candidate_b.crowding_distance:
            return candidate_a
        return candidate_b

    def simulated_binary_crossover(
        self,
        parent_a: ArchetypeIndividual,
        parent_b: ArchetypeIndividual
    ) -> Tuple[ArchetypeIndividual, ArchetypeIndividual]:
        """Performs Simulated Binary Crossover (SBX) with parameter bounds checking."""
        child_a_genes: Dict[str, float] = {}
        child_b_genes: Dict[str, float] = {}

        for gene, (low, high) in self.bounds.items():
            p1 = parent_a.genes.get(gene, low)
            p2 = parent_b.genes.get(gene, high)

            if random.random() <= self.crossover_rate and abs(p1 - p2) > 1e-12:
                y1 = min(p1, p2)
                y2 = max(p1, p2)

                u = random.random()
                if u <= 0.5:
                    beta_q = (2.0 * u) ** (1.0 / (self.eta_c + 1.0))
                else:
                    beta_q = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta_c + 1.0))

                c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

                # Clamp to bounds
                child_a_genes[gene] = max(low, min(high, c1))
                child_b_genes[gene] = max(low, min(high, c2))
            else:
                child_a_genes[gene] = p1
                child_b_genes[gene] = p2

        return (
            ArchetypeIndividual(genes=child_a_genes),
            ArchetypeIndividual(genes=child_b_genes)
        )

    def polynomial_mutation(
        self,
        individual: ArchetypeIndividual
    ) -> ArchetypeIndividual:
        """Applies Deb's bounded Polynomial Mutation to individual genes."""
        mutated_genes: Dict[str, float] = {}

        for gene, (low, high) in self.bounds.items():
            val = individual.genes.get(gene, low)

            if random.random() <= self.mutation_rate:
                delta_range = high - low
                if delta_range > 1e-12:
                    delta_1 = (val - low) / delta_range
                    delta_2 = (high - val) / delta_range
                    u = random.random()

                    if u <= 0.5:
                        inner = 2.0 * u + (1.0 - 2.0 * u) * ((1.0 - delta_1) ** (self.eta_m + 1.0))
                        delta_q = (inner ** (1.0 / (self.eta_m + 1.0))) - 1.0
                    else:
                        inner = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * ((1.0 - delta_2) ** (self.eta_m + 1.0))
                        delta_q = 1.0 - (inner ** (1.0 / (self.eta_m + 1.0)))

                    mut_val = val + delta_q * delta_range
                    val = max(low, min(high, mut_val))

            mutated_genes[gene] = val

        return ArchetypeIndividual(genes=mutated_genes)

    def step_generation(
        self,
        population: List[ArchetypeIndividual],
        evaluator: Callable[[ArchetypeIndividual], MultiObjectiveFitness]
    ) -> List[ArchetypeIndividual]:
        """Executes a single NSGA-II generational step with elitist replacement."""
        pop_size = len(population)
        if pop_size == 0:
            return []

        # 1. Evaluate fitness if missing
        for ind in population:
            if ind.fitness is None:
                ind.fitness = evaluator(ind)

        # 2. Fast non-dominated sort & crowding distance on current population
        fronts = self.fast_non_dominated_sort(population)
        for f in fronts:
            self.compute_crowding_distance(f)

        # 3. Offspring generation via tournament selection, SBX crossover, and mutation
        offspring: List[ArchetypeIndividual] = []
        while len(offspring) < pop_size:
            p1 = self.tournament_select(random.choice(population), random.choice(population))
            p2 = self.tournament_select(random.choice(population), random.choice(population))
            c1, c2 = self.simulated_binary_crossover(p1, p2)
            c1 = self.polynomial_mutation(c1)
            c2 = self.polynomial_mutation(c2)
            c1.fitness = evaluator(c1)
            c2.fitness = evaluator(c2)
            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        # 4. Form combined population R_t of size 2N
        combined = population + offspring

        # 5. Non-dominated sort on combined population
        combined_fronts = self.fast_non_dominated_sort(combined)

        # 6. Fill new population front by front
        next_population: List[ArchetypeIndividual] = []
        for front in combined_fronts:
            self.compute_crowding_distance(front)
            if len(next_population) + len(front) <= pop_size:
                next_population.extend(front)
            else:
                # Crowding distance sort descending for tie-breaking
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                needed = pop_size - len(next_population)
                next_population.extend(front[:needed])
                break

        return next_population

    def evolve(
        self,
        initial_population: List[ArchetypeIndividual],
        evaluator: Callable[[ArchetypeIndividual], MultiObjectiveFitness],
        generations: int = 20
    ) -> List[ArchetypeIndividual]:
        """Runs the multi-objective evolutionary loop across specified generations."""
        pop = list(initial_population)
        for _ in range(generations):
            pop = self.step_generation(pop, evaluator)

        # Final sort to assign rank and crowding distance
        fronts = self.fast_non_dominated_sort(pop)
        for f in fronts:
            self.compute_crowding_distance(f)

        return pop

    @classmethod
    def get_pareto_front(
        cls,
        population: List[ArchetypeIndividual]
    ) -> List[ArchetypeIndividual]:
        """Extracts Rank 1 (non-dominated Pareto front) individuals from a population."""
        fronts = cls.fast_non_dominated_sort(population)
        if not fronts:
            return []
        cls.compute_crowding_distance(fronts[0])
        return fronts[0]
