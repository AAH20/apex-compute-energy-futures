# Module Deep-Dive: `futures`

The `futures` module models forward compute yield curves, executes synthetic inertia automated market-maker swaps, and scans for cross-venue spot arbitrage.

---

## Key Classes & Functions

### 1. `ComputeForwardCurve` (`futures/forward_curve.py`)
Computes the forward price of GPU compute using continuous cost-of-carry principles:

$$F(t, T) = S_t \cdot \exp\left((r + c - y)(T - t)\right) \cdot \text{Multiplier}_{\text{SLA}}$$

Where:
* $S_t$: Spot rate ($/GPU-hour).
* $r$: Risk-free rate of capital.
* $c$: Physical carrying cost (GPU power draw $\times$ facility PUE $\times$ electricity tariff $/kWh).
* $y$: Convenience yield of holding uncommitted immediate capacity.
* $\text{Multiplier}_{\text{SLA}}$: 0.85 (Best-effort), 1.00 (Fault-tolerant), 1.35 (Mission-critical).

```python
from apex_compute_futures.futures.forward_curve import ComputeForwardCurve

curve = ComputeForwardCurve(region="US_EAST_VA", base_spot_rate=2.45, facility_pue=1.18)
term_structure = curve.generate_full_term_structure()
```

---

### 2. `SyntheticInertiaAMM` (`futures/synthetic_inertia_amm.py`)
Multi-asset continuous invariant bonding curve uniting battery storage $E$, shedable compute capacity $C$, and grid frequency $f$:

$$\Phi(E, C, f) = E^\alpha \cdot C^\beta \cdot f^\gamma = K$$

#### Grid Frequency Droop Response:
When the regional power grid experiences an under-frequency event ($\Delta f < 0$), the AMM calculates the required compute shedding via microsecond DVFS clock drops:

$$\Delta C = \frac{\gamma}{\beta} \cdot \frac{C_0}{f_0} \cdot \Delta f \cdot G_{\text{droop}}$$

Monetizes datacenter energy flexibility as synthetic inertia during ERCOT/PJM emergency ancillary service price spikes ($12,500/MWh).

---

### 3. `ComputeArbitrageEngine` (`futures/arbitrage_engine.py`)
Scans instantaneous spot order books (vast.ai, RunPod) against the forward clearance threshold.
* Calculates non-linear order book market impact:
  $$\text{Slippage}(V) = \kappa \sqrt{\frac{V}{\text{Depth}}}$$
* Formulates verified `ArbitrageQuote`s including gross spread, estimated slippage, taker fees, and net edge.
