# Module Deep-Dive: `evolution`

The `evolution` module implements genetic archetype reproduction and closed-form Deflated Sharpe Ratio (DSR) evaluation with hard barrier safety constraints.

---

## Key Classes & Functions

### 1. `SystemFitnessCalculator` (`evolution/fitness.py`)
Computes statistically rigorous, non-linear fitness scores that penalize selection bias and immediately disqualify unsafe archetypes:

```python
from apex_compute_futures.evolution.fitness import SystemFitnessCalculator

calc = SystemFitnessCalculator(max_sla_invalidation_rate=0.0005, max_drawdown_limit=0.15)
fitness = calc.evaluate_archetype_fitness(
    returns=sim_returns,
    sla_invalidation_rate=0.0001,
    realized_max_drawdown=0.042,
    num_generations_tested=50
)
```

#### The Deflated Sharpe Ratio (DSR):
Standard Sharpe ratios overstate performance when evaluating multiple trials ($N$). The DSR adjusts for the expected maximum Sharpe under the null hypothesis of uninformative strategies:

$$\text{DSR} = \text{SR} \cdot \sqrt{1 - \gamma_3 \cdot \text{SR} + \frac{\gamma_4 - 1}{4} \cdot \text{SR}^2} \cdot \left(1 - \frac{\gamma_{\text{Euler}}}{\sqrt{2 \ln N}}\right)$$

#### Hard Barrier Invariants:
* **SLA Invalidation Barrier:** If $\text{SLA Invalidation Rate} > 0.05\%$, fitness $= -\infty$.
* **Max Drawdown Barrier:** If $\text{Drawdown} > 15\%$, fitness $= -\infty$.

---

### 2. `AgentChromosome` & `ChromosomeEngine` (`evolution/chromosome.py`)
Encodes bounded, operational hyperparameters governing agent prediction and execution:
* `d_fractional`: Fractional differencing order $\in [0.10, 0.85]$.
* `profit_multiplier`: Take-profit barrier multiplier $\in [1.00, 6.00]$.
* `stop_loss_multiplier`: Stop-loss barrier multiplier $\in [0.50, 3.50]$.
* `max_holding_horizon`: Maximum holding window in hours $\in [4, 72]$.
* `synthetic_inertia_alpha`: AMM energy bonding exponent $\in [0.20, 0.60]$.
* `frequency_trip_threshold_hz`: Grid trip trigger $\in [59.50, 59.95]\text{ Hz}$.

Provides bounded Gaussian mutation and uniform crossover operators.
