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

---

### 3. `FormalSettlementProver` (`core/settlement_prover.py`)
Executable Formal Settlement Prover, First-Order Logic SMT constraint engine, and cryptographic audit ledger.

```python
from apex_compute_futures.core.settlement_prover import FormalSettlementProver, ProofStatus

prover = FormalSettlementProver(reserve_floor_usd=1_000_000.0, max_rack_kw=40.0, t_crit_c=85.0)

# Formally prove trade settlement invariants
ok, cert, block = prover.prove_trade_settlement(
    trade_id="TR_1001",
    instance_ids=["h100_node_alpha", "h100_node_beta"],
    cash_delta_usd=-120_000.0,
    lease_value_usd=145_000.0,
    vault_balance_usd=5_000_000.0
)

assert ok is True
assert cert.status == ProofStatus.PROVED
print(f"Audit Block Hash: {block.block_hash}, Merkle Root: {block.merkle_root}")

# Verify complete ledger cryptographic integrity
ledger_valid, msg = prover.verify_audit_ledger()
assert ledger_valid is True
```

#### Core Components:
1. **`SMTConstraintSolver`**: First-Order Logic AST solver evaluating linear arithmetic inequalities ($\le, \ge, ==, \&\&, !$).
2. **`MerkleTree`**: RFC 6962 SHA-256 binary state trees with $O(\log N)$ inclusion proof generation and verification.
3. **`Ed25519Signer`**: RFC 8032 standard Curve25519 digital signature generator and verifier in pure Python with table acceleration.
4. **`CryptographicAuditLedger`**: Append-only tamper-evident blockchain ledger committing signed state blocks.
