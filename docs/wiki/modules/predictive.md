# Module Deep-Dive: `predictive`

The `predictive` module transforms non-stationary commodity power and spot compute prices into stationary, memory-preserving representations, samples them via informational clocks, and applies triple-barrier meta-labeling.

---

## Key Classes & Functions

### 1. `FractionalDifferentiator` (`predictive/fractional_diff.py`)
Computes binomial series weights for the fractional difference operator $(1 - L)^d$:

$$w_0 = 1, \quad w_k = -w_{k-1} \frac{d - k + 1}{k}$$

#### Usage:
```python
from apex_compute_futures.predictive.fractional_diff import FractionalDifferentiator

raw_prices = [2.40, 2.45, 2.42, 2.50, 2.55, 2.60]

# Find optimal order d* retaining at least 75% autocorrelation memory
optimal_d = FractionalDifferentiator.find_optimal_d(raw_prices, min_memory=0.75)

# Transform raw series
diff_series = FractionalDifferentiator.transform(raw_prices, d=optimal_d)
```

---

### 2. `CusumInformationSampler` (`predictive/information_bars.py`)
Replaces calendar time-bars with dynamic information bars sampled upon cumulative symmetric deviation of normalized power-compute turnover:

$$S_t^+ = \max(0, S_{t-1}^+ + \Delta P_t), \quad S_t^- = \min(0, S_{t-1}^- + \Delta P_t)$$
$$\text{Sample Bar if } \max(S_t^+, -S_t^-) \ge h$$

Filters out intra-day liquidity churn and samples densely only when structural price/power shifts occur.

---

### 3. `TripleBarrierLabeler` (`predictive/triple_barrier.py`)
Labels trading trajectories across three concurrent boundaries:
1. **Upper Horizontal Barrier:** Profit target $\theta_{\text{profit}} = P_0 (1 + \beta_{\text{up}} \sigma)$.
2. **Lower Horizontal Barrier:** Stop-loss $\theta_{\text{loss}} = P_0 (1 - \beta_{\text{down}} \sigma)$.
3. **Vertical Barrier:** Maximum temporal holding window $T_{\text{hold}}$.

* **`compute_meta_allocation_size(probability_success, calibration_sigma)`:** Uses a calibrated Gaussian CDF to scale capital allocation smoothly between $0\%$ and $100\%$, preventing binary all-or-nothing allocation errors.
