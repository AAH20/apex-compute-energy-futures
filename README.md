# Apex Compute-Energy Futures, Predictive ML & Microsecond Arbitrage Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests Passing](https://img.shields.io/badge/tests-14%2F14%20passing%20(1ms)-brightgreen.svg)]()
[![Zero Dependencies](https://img.shields.io/badge/dependencies-pure%20standard%20library-brightgreen.svg)]()
[![Demonstration Verified](https://img.shields.io/badge/demo-verified%20(1.92ms)-purple.svg)]()
[![NVIDIA Inception](https://img.shields.io/badge/NVIDIA%20Inception-Member%20Portfolio-green.svg)](https://www.nvidia.com/en-us/startups/)
[![AWS Marketplace](https://img.shields.io/badge/AWS%20Marketplace-Verified%20Seller%20Offering-orange.svg)](https://aws.amazon.com/marketplace)

> **A sovereign, zero-dependency algorithmic operating system and predictive machine learning platform for distributed AI data centers, power-grid energy futures, synthetic inertia bonding curves, and sub-millisecond compute arbitrage.**

---

## ⚡ 5-Second Zero-Friction Inspection Quickstart

Run the master interactive demonstration harness in pure standard Python with zero pip dependencies:

```bash
git clone https://github.com/AAH20/apex-compute-energy-futures.git
cd apex-compute-energy-futures
python3 demo.py
```

*Or run the test and benchmark suite:*
```bash
make test   # Executes 14/14 unit tests in ~1 millisecond
make all    # Runs test suite + master interactive demo
```

---

## 🏛️ System Architecture: The Multi-Scale Temporal Hierarchy

Traditional monolithic machine learning models fail when trying to couple semiconductor supply-chain lead times (months) with high-frequency spot GPU price fluctuations (seconds). 

`apex-compute-energy-futures` decouples execution into **three asynchronous temporal horizons**:

```mermaid
flowchart TD
    subgraph Tier3 ["Tier 3: Macro-Horizon Predictive Engine (30 - 90 Days)"]
        T3_DATA["Observable Proxy Data<br/>FERC Interconnection Queues • Semi Book-to-Bill<br/>Hyperscaler CapEx Amortization • TSMC 10-K Filings"]
        T3_MODEL["Bayesian Structural Vector Error Correction (SVEC)<br/>Horseshoe Shrinkage Priors"]
        T3_OUTPUT["Macro Structural Regime Prior P(Regime | Theta)"]
        T3_DATA --> T3_MODEL --> T3_OUTPUT
    end

    subgraph Tier2 ["Tier 2: Meso-Horizon Yield & Forward Curve Engine (1 - 24 Hours)"]
        T2_DATA["Locational Marginal Electricity Pricing (ERCOT/PJM LMP)<br/>Natural Gas Futures (Henry Hub) • Facility PUE & Heat Plumes"]
        T2_MODEL["Continuous Cost-of-Carry Forward Curve Engine<br/>F(t, T) = S_t * exp((r + c - y) * (T - t))<br/>A-ECMM Invariant Solver: Phi(E, C, f) = K"]
        T2_OUTPUT["Forward Compute Arbitrage Clearance Matrix P_clear(r, t)"]
        T3_OUTPUT -->|"Bi-Weekly Bayesian Update"| T2_MODEL
        T2_DATA --> T2_MODEL --> T2_OUTPUT
    end

    subgraph Tier1 ["Tier 1: Micro-Horizon Real-Time Execution Core (<10 µs)"]
        T1_DATA["L3 Order Books & Spot Ticks (vast.ai, RunPod)<br/>eBPF Host Telemetry (Ping, Packet Loss, Jitter Cache)"]
        T1_GATE["Deterministic Fast-Path ActionGate<br/>P_spot < P_clear AND Host in HealthySet"]
        T1_EXEC["High-Throughput Matching & Liquidation Engine<br/>Almgren-Chriss Hyperbolic Trajectory • DVFS Shedding"]
        T2_OUTPUT -->|"Pushed Hourly to Shared Memory"| T1_GATE
        T1_DATA --> T1_GATE -->|"Approved (<10 µs)"| T1_EXEC
    end

    subgraph OutOfBand ["Asynchronous Verification & Audit Ledger"]
        Z3["FormalSettlementGuard (SMT/Z3 Invariants)"]
        AUDIT["Immutable SHA-256 Evidence Vault"]
        T1_EXEC -.->|"Post-Trade Stream"| Z3 --> AUDIT
    end
```

---

## ⚡ Core Algorithmic Engines

### 1. Memory-Preserving Fractional Differencing (`predictive/fractional_diff.py`)
Applies the binomial series expansion $(1 - L)^d = \sum_{k=0}^{\infty} (-1)^k \binom{d}{k} L^k$ to transform non-stationary commodity power and spot prices into stationary inputs without erasing long-range memory.

### 2. CUSUM Information Sampling Clock (`predictive/information_bars.py`)
Replaces calendar time-bars with informational bars sampled upon cumulative symmetric deviation of normalized power-compute turnover, filtering out high-frequency market noise.

### 3. Triple-Barrier Labeling & Meta-Labeling Sizer (`predictive/triple_barrier.py`)
Dynamic upper (profit), lower (stop-loss), and vertical (holding timeout) boundaries with Gaussian CDF probability-calibrated capital allocation sizing.

### 4. Continuous Cost-of-Carry Forward Curve (`futures/forward_curve.py`)
Calculates the structural forward yield curve $F(t, T) = S_t e^{(r + c - y)(T - t)}$ across maturities from 1 hour to 90 days, integrating facility PUE, locational electricity tariffs, and convenience yields.

### 5. Automated Energy-Compute Market Maker (`futures/synthetic_inertia_amm.py`)
Implements the multi-asset invariant bonding curve $\Phi(E, C, f) = E^\alpha C^\beta f^\gamma = K$, enabling datacenters to monetize fast battery storage and microsecond DVFS GPU clock throttling ($P \propto f^{2.8}$) as **synthetic inertia** during ERCOT/PJM under-frequency trips.

### 6. Deterministic Fast-Path ActionGate (`core/action_gate.py`)
Sub-10 microsecond prove-or-deny risk boundary evaluating pre-vetted eBPF host telemetry, price ceilings, capital floor invariants, and thermal breaker limits before dispatching trades.

### 7. Closed-Form Deflated Sharpe Ratio Evolution (`evolution/fitness.py`)
Corrects backtest Sharpe ratios for multiple-testing selection bias (Bailey & López de Prado) and enforces hard barrier disqualification ($-\infty$) for SLA invalidations and drawdown breaches.

---

## 📊 The 6 Production System Evaluation Parameters (KPIs)

| Metric | Name | Mathematical Scope | Acceptance Threshold |
| :--- | :--- | :--- | :--- |
| **1. INR** | Information-to-Noise Ratio | $\text{Var}(\mu_{\text{structural}}) / \text{Var}(\epsilon_{\text{noise}})$ | $\ge 3.2\text{ dB}$ |
| **2. PDG** | Predictive Divergence Gap | $\frac{1}{M} \sum \|\mathbf{P}_{\text{forecast}} - \mathbf{P}_{\text{settled}}\| / \mathbf{P}_{\text{settled}}$ | $\le 4.8\%\text{ (30-day window)}$ |
| **3. SPRL** | Stateless Preemption Recovery | $t_{\text{detect}} + t_{\text{route}} + t_{\text{init}}$ | $\le 850\text{ ms}$ |
| **4. SCM** | System Contribution Margin | $(\text{Net Arbitrage Revenue}) / \sum \text{Agent Compute COGS}$ | $\ge 4.5\times$ |
| **5. SIP** | SLA Invalidation Probability | $N_{\text{breach}} / N_{\text{total leases}}$ | $< 0.01\%\text{ (Zero-Tolerance)}$ |
| **6. ESD** | Execution Slippage Deficit | $\text{Realized Slippage} - \text{Almgren-Chriss Optimal Bound}$ | $\le 2.8\text{ bps}$ |

---

## 🌐 Sovereign Ecosystem On-Demand Integrations

`apex-compute-energy-futures` is architected with a decoupled `EcosystemIntegrationRegistry` allowing on-demand binding to specialized peer kernels across the **Ahmed Hassan (@AAH20)** ecosystem:

```python
from apex_compute_futures.integrations.registry import EcosystemIntegrationRegistry

# List available ecosystem adapters
integrations = EcosystemIntegrationRegistry.list_available_integrations()

# Dynamically load the Data Center Commander / NP-Hard kernel adapter
dcc = EcosystemIntegrationRegistry.load_adapter("datacenter_np_hard_kernel")
facility_stats = dcc.get_facility_telemetry()
```

| Integration Target | Repository | Core Role in Production Deployment |
| :--- | :--- | :--- |
| **Data Center Commander** | [`AAH20/data-center-commander`](https://github.com/AAH20/data-center-commander) | Read-only facility telemetry, PUE, and 6 NP-Hard combinatorial solvers. |
| **Gigawatt Ride-Through AMM** | [`AAH20/gigawatt-ride-through-amm-kernel`](https://github.com/AAH20/gigawatt-ride-through-amm-kernel) | Multi-tier Ragone knapsack & sub-2ms DVFS GPU power shedding. |
| **HFT Microstructure Kernel** | [`AAH20/hft-microstructure-kernel`](https://github.com/AAH20/hft-microstructure-kernel) | 471k orders/s LOB & Almgren-Chriss hyperbolic liquidation trajectories. |
| **Apex Quant Whale Kernel** | [`AAH20/apex-quant-whale-kernel`](https://github.com/AAH20/apex-quant-whale-kernel) | Hawkes point process toxic flow classifier & relativistic skywave routing. |
| **Agentic FinTech Kernel** | [`AAH20/agentic-fintech-kernel`](https://github.com/AAH20/agentic-fintech-kernel) | SMT/Z3 formal invariant proving engine & multi-rail token settlement. |
| **Apex FDE Matrix** | [`AAH20/Apex_FDE_Matrix`](https://github.com/AAH20/Apex_FDE_Matrix) | 0.12ms Causal Digital Twin & bounded blast-radius SRE control plane. |
| **Intraday Liquidity Orchestrator** | [`AAH20/intraday-liquidity-orchestrator`](https://github.com/AAH20/intraday-liquidity-orchestrator) | 24/7 ISO 20022 `camt.050` liquidity sweeps across FedNow/RTP/SEPA. |

*See [`docs/INTEGRATIONS.md`](docs/INTEGRATIONS.md) for full interface specifications.*

---

## 🏛️ Open-Core vs. Enterprise Platform Demarcation

To preserve the commercial envelope, this project enforces strict separation between the public Apache-2.0 open-core and the proprietary enterprise platform:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ OPEN-SOURCE COMMUNITY CORE (APACHE-2.0)   │ ENTERPRISE PLATFORM (APEX GROWTH SYSTEMS)  │
├───────────────────────────────────────────┼────────────────────────────────────────────┤
│ • Pure Python Standard Library ML Solvers │ • Multi-MW Live DCIM / Modbus Ingestion    │
│ • CUSUM & Fractional Differencing Engines │ • Sub-2ms DVFS Hardware Shedding Daemon    │
│ • A-ECMM Synthetic Inertia Bonding Curve  │ • AWS Marketplace 100% EDP Drawdown Listing│
│ • Deterministic Fast ActionGate (<10 µs)  │ • ISO 20022 camt.050 Clearinghouse Sweeps  │
│ • SMT/Z3 Proof Primitives & Audit Hashing │ • 24/7 Enterprise SLA & Dedicated Support  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

*See [`docs/COMMERCIAL_SPEC.md`](docs/COMMERCIAL_SPEC.md) for commercial tiers, pricing, and AWS private offers.*

---

## 💼 AWS Marketplace Offering & Enterprise Procurement

The Enterprise Platform is distributed via the **AWS Marketplace** under **Apex Growth Systems LLC**:
* **100% AWS EDP Drawdown:** Enterprise buyers burn mandatory committed cloud spend with zero new procurement cycles.
* **NVIDIA Inception Member:** Qualified startup portfolio architecture.
* **US Entity:** Apex Growth Systems LLC (Arizona, USA).
* **Enterprise Licensing Inquiries:** `sales@a2zsoc.com` | [https://a2zsoc.com](https://a2zsoc.com)

---

## 📜 License
Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) for terms.
Copyright 2026 Apex Growth Systems LLC.
