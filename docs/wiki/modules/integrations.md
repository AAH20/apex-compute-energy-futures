# Module Deep-Dive: `integrations`

The `integrations` module provides an on-demand plugin and adapter registry that binds `apex-compute-energy-futures` to external ecosystem repositories.

---

## 🔌 The `EcosystemIntegrationRegistry` (`integrations/registry.py`)

Maintains metadata and dynamic import hooks for peer kernels across the **Ahmed Hassan (@AAH20)** ecosystem. The core monorepo runs completely standalone with zero dependencies; external adapters are loaded lazily on demand.

### Usage Example:
```python
from apex_compute_futures.integrations.registry import EcosystemIntegrationRegistry

# 1. Discover all registered ecosystem targets
available = EcosystemIntegrationRegistry.list_available_integrations()
print("Supported integrations:", list(available.keys()))

# 2. Lazy-load specific adapter
dcc_adapter = EcosystemIntegrationRegistry.load_adapter("datacenter_np_hard_kernel")
telemetry = dcc_adapter.get_facility_telemetry()
print("Facility PUE:", telemetry["pue_ratio"])
```

---

## Available On-Demand Adapters

| Adapter Module | Target Ecosystem Kernel | Primary Production Interface |
| :--- | :--- | :--- |
| `dcc_adapter.py` | [`AAH20/datacenter-np-hard-kernel`](https://github.com/AAH20/datacenter-np-hard-kernel) | `get_facility_telemetry()`: Ingests read-only rack power & PUE. |
| `gigawatt_adapter.py` | [`AAH20/gigawatt-ride-through-amm-kernel`](https://github.com/AAH20/gigawatt-ride-through-amm-kernel) | `trigger_dvfs_shedding()`: Dispatches sub-2ms GPU clock throttling. |
| `hft_adapter.py` | [`AAH20/hft-microstructure-kernel`](https://github.com/AAH20/hft-microstructure-kernel) | `execute_optimal_liquidation()`: Almgren-Chriss hyperbolic execution. |
| `formal_guard_adapter.py` | [`AAH20/agentic-fintech-kernel`](https://github.com/AAH20/agentic-fintech-kernel) | `prove_settlement_invariants()`: Formal SMT/Z3 satisfiability solver. |
| `fde_matrix_adapter.py` | [`AAH20/Apex_FDE_Matrix`](https://github.com/AAH20/Apex_FDE_Matrix) | `calculate_mutation_blast_radius()`: 0.12ms Causal Digital Twin check. |
| `liquidity_sweeper_adapter.py` | [`AAH20/intraday-liquidity-orchestrator`](https://github.com/AAH20/intraday-liquidity-orchestrator) | `trigger_camt050_liquidity_sweep()`: Automated ISO 20022 sweeps. |
