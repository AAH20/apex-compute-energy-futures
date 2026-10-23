"""Evolutionary Chromosome and Genetic Operators for Compute-Energy Swarm Agents."""

import random
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class AgentChromosome:
    """Bounded hyperparameter genes governing agent predictive and execution behavior."""
    d_fractional: float = 0.45             # Fractional differencing order
    profit_multiplier: float = 2.50         # Triple barrier profit threshold
    stop_loss_multiplier: float = 1.20      # Triple barrier stop threshold
    max_holding_horizon: int = 24           # Vertical barrier hours
    synthetic_inertia_alpha: float = 0.45   # AMM energy exponent
    frequency_trip_threshold_hz: float = 59.85  # Grid trip shedding trigger


class ChromosomeEngine:
    """Manages genetic crossover, bounded mutation, and archetype breeding."""

    GENE_BOUNDS: Dict[str, Tuple[float, float]] = {
        "d_fractional": (0.10, 0.85),
        "profit_multiplier": (1.00, 6.00),
        "stop_loss_multiplier": (0.50, 3.50),
        "max_holding_horizon": (4.0, 72.0),
        "synthetic_inertia_alpha": (0.20, 0.60),
        "frequency_trip_threshold_hz": (59.50, 59.95)
    }

    @classmethod
    def mutate(cls, chrom: AgentChromosome, mutation_rate: float = 0.15) -> AgentChromosome:
        """Applies bounded Gaussian perturbations to active genes."""
        data = {
            "d_fractional": chrom.d_fractional,
            "profit_multiplier": chrom.profit_multiplier,
            "stop_loss_multiplier": chrom.stop_loss_multiplier,
            "max_holding_horizon": float(chrom.max_holding_horizon),
            "synthetic_inertia_alpha": chrom.synthetic_inertia_alpha,
            "frequency_trip_threshold_hz": chrom.frequency_trip_threshold_hz
        }

        for gene, (low, high) in cls.GENE_BOUNDS.items():
            if random.random() < mutation_rate:
                sigma = (high - low) * 0.05
                val = data[gene] + random.gauss(0.0, sigma)
                data[gene] = max(low, min(high, val))

        return AgentChromosome(
            d_fractional=round(data["d_fractional"], 3),
            profit_multiplier=round(data["profit_multiplier"], 2),
            stop_loss_multiplier=round(data["stop_loss_multiplier"], 2),
            max_holding_horizon=int(round(data["max_holding_horizon"])),
            synthetic_inertia_alpha=round(data["synthetic_inertia_alpha"], 3),
            frequency_trip_threshold_hz=round(data["frequency_trip_threshold_hz"], 3)
        )

    @classmethod
    def crossover(cls, parent_a: AgentChromosome, parent_b: AgentChromosome) -> AgentChromosome:
        """Executes uniform crossover between two fit parent archetypes."""
        return AgentChromosome(
            d_fractional=random.choice([parent_a.d_fractional, parent_b.d_fractional]),
            profit_multiplier=random.choice([parent_a.profit_multiplier, parent_b.profit_multiplier]),
            stop_loss_multiplier=random.choice([parent_a.stop_loss_multiplier, parent_b.stop_loss_multiplier]),
            max_holding_horizon=random.choice([parent_a.max_holding_horizon, parent_b.max_holding_horizon]),
            synthetic_inertia_alpha=random.choice([parent_a.synthetic_inertia_alpha, parent_b.synthetic_inertia_alpha]),
            frequency_trip_threshold_hz=random.choice([parent_a.frequency_trip_threshold_hz, parent_b.frequency_trip_threshold_hz])
        )
