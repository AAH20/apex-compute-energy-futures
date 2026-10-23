# Module Deep-Dive: `core`

The `core` module provides the foundational risk gating, hardware node representations, and formal mathematical safety invariants for all compute transactions.

---

## Key Classes & Functions

### 1. `ActionGate` (`core/action_gate.py`)
Deterministic fast-path execution boundary. Designed to run in **$<10\text{ µs}$** without heap allocations or I/O waits.

```python
from apex_compute_futures.core.action_gate import ActionGate

gate = ActionGate(
    max_single_trade_usd=2500.0,
    max_daily_budget_usd=25000.0,
    max_packet_loss_pct=1.0,
    max_ping_latency_ms=45.0
)
```

#### Evaluation Pipeline:
1. **Single Trade & Daily Quota Guard:** Rejects trades immediately if budget thresholds are exceeded.
2. **Pre-Vetted Host Telemetry Filter:** Validates that the target host exists in the local eBPF cache and satisfies packet loss ($\le 1.0\%$) and latency constraints.
3. **Forward Clearance Ceiling:** Rejects spot bids that fail to clear the regional forward curve.
4. **Formal SMT Invariant Verification:** Verifies capital floor conservation and physical thermal breaker boundaries.

---

### 2. `FormalInvariantVerifier` (`core/invariants.py`)
Pure-Python representation of first-principles physical and financial constraints.

* **`verify_capital_conservation(balance, outflow)`:** Proves that post-trade capital remains above the regulatory minimum floor:
  $$\text{Balance} - \text{Outflow} \ge \text{Floor}_{\text{Basel}}$$
* **`verify_thermal_power_boundary(current_kw, add_kw)`:** Enforces an 80% maximum safety headroom on rack power breakers:
  $$\text{RackDraw}_{\text{total}} \le \text{BreakerLimit}_{\text{rack}} \times 0.80$$
* **`verify_no_double_allocation(instance_id)`:** Guarantees that a GPU node cannot be leased twice simultaneously across competing swarm agents.
* **`generate_cryptographic_audit_hash(...)`:** Produces a tamper-evident SHA-256 evidence record for the immutable audit trail.
