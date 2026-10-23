# Apex Compute-Energy Futures: Engineering Code Wiki

Reference documentation and system topography for the `apex-compute-energy-futures` monorepo.

---

## 🗺️ Module Architecture Map

| Module | Core Responsibility | Primary Classes | Source Path |
| :--- | :--- | :--- | :--- |
| **`core`** | Sub-microsecond risk boundaries, hardware telemetry, & invariant proofs. | `ActionGate`, `FormalInvariantVerifier`, `FormalSettlementProver`, `CryptographicAuditLedger` | [`apex_compute_futures/core`](file:///Users/ahmedhassan/Downloads/2000%20workflows/projects/apex-compute-energy-futures/apex_compute_futures/core) |
| **`simulation`** | Institutional multi-scenario crisis stress testing & 10,000-path Monte Carlo risk engine. | `InstitutionalStressTestingEngine`, `StressScenario`, `StressTestResult` | [`apex_compute_futures/simulation`](file:///Users/ahmedhassan/Downloads/2000%20workflows/projects/apex-compute-energy-futures/apex_compute_futures/simulation) |
| **`predictive`** | Fractional differentiation, CUSUM information sampling, & triple-barrier meta-labeling. | `FractionalDifferentiator`, `CusumInformationSampler`, `TripleBarrierLabeler` | [`apex_compute_futures/predictive`](file:///Users/ahmedhassan/Downloads/2000%20workflows/projects/apex-compute-energy-futures/apex_compute_futures/predictive) |
| **`futures`** | Forward yield curves, cost-of-carry, A-ECMM synthetic inertia bonding, & arbitrage scanner. | `ComputeForwardCurve`, `SyntheticInertiaAMM`, `ComputeArbitrageEngine` | [`apex_compute_futures/futures`](file:///Users/ahmedhassan/Downloads/2000%20workflows/projects/apex-compute-energy-futures/apex_compute_futures/futures) |
| **`evolution`** | Closed-form Deflated Sharpe Ratio (DSR), hard barrier constraints, & genetic breeding. | `SystemFitnessCalculator`, `AgentChromosome`, `ChromosomeEngine` | [`apex_compute_futures/evolution`](file:///Users/ahmedhassan/Downloads/2000%20workflows/projects/apex-compute-energy-futures/apex_compute_futures/evolution) |
| **`quant_alpha`**| O-U jump-diffusion MLE, Hawkes processes, Pareto DSR tournament, thermal enthalpy, & Almgren-Chriss. | `OrnsteinUhlenbeckJumpEstimator`, `HawkesPointProcessEngine`, `ParetoTournamentEngine`, `AlmgrenChrissExecutionEngine`, `LockFreeSPSCQueue` | [`docs/QUANT_ALPHA_SPEC.md`](file:///Users/ahmedhassan/Downloads/2000%20workflows/projects/apex-compute-energy-futures/docs/QUANT_ALPHA_SPEC.md) |
| **`integrations`**| Dynamic on-demand adapter registry for ecosystem kernels (`dcc`, `gigawatt`, `hft`, etc.).| `EcosystemIntegrationRegistry` | [`apex_compute_futures/integrations`](file:///Users/ahmedhassan/Downloads/2000%20workflows/projects/apex-compute-energy-futures/apex_compute_futures/integrations) |

---

## 🧭 Navigation Index

* [System Architecture & Dataflow](architecture.md)
* [UML Class Diagram](diagrams/class-diagram.md)
* [Execution Sequence Diagrams](diagrams/sequences.md)
* [Getting Started & Local Testing](getting-started.md)
* **Module Deep Dives:**
  * [Core Gating & Formal Settlement Prover](modules/core.md)
  * [Institutional Crisis Simulation & Stress Testing](modules/simulation.md)
  * [Predictive Modeling & Sampling](modules/predictive.md)
  * [Futures & Synthetic Inertia AMM](modules/futures.md)
  * [Evolution & Deflated Sharpe Engine](modules/evolution.md)
  * [Quantitative Alpha & Mathematical Modules](modules/quant_alpha.md)
  * [Ecosystem Integrations Adapter](modules/integrations.md)
