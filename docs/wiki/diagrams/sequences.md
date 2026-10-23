# Execution Sequence Diagrams

This document details the microsecond execution sequences across critical runtime workflows.

---

## 1. Fast-Path Arbitrage Evaluation (Sub-10us)

```mermaid
sequenceDiagram
    autonumber
    actor SpotBook as Spot Market Feed
    participant Gate as ActionGate (Sub-10us)
    participant Cache as Host Health Cache
    participant Invariant as Invariant Verifier
    participant Node as Target GPU Node
    participant Vault as Evidence Vault

    SpotBook->>Gate: ArbitrageQuote ($1.55 vs $2.46 clearance)
    Gate->>Cache: Lookup Host IP Telemetry
    Cache-->>Gate: Host Health OK (0.0 pct loss, 18ms ping)
    Gate->>Invariant: verify_capital_conservation(balance, cost)
    Invariant-->>Gate: Capital Floor Proved (Balance above $1,000)
    Gate->>Invariant: verify_thermal_power_boundary(rack_kw, node_kw)
    Invariant-->>Gate: Thermal Headroom Proved (Total within 32 kW)
    Gate->>Node: Dispatch Container Lease (Sub-20us)
    Gate-->>Vault: Async Emit Signed Audit Hash (SHA-256)
```

---

## 2. Emergency Grid Under-Frequency Response (A-ECMM)

```mermaid
sequenceDiagram
    autonumber
    actor Grid as Substation PLL Sensor
    participant AMM as Synthetic Inertia AMM
    participant DVFS as DVFS Cluster Daemon
    participant GPU as H100 Cluster Racks
    participant RTO as ERCOT / PJM Market

    Grid->>AMM: Under-Frequency Event (59.82 Hz, -0.18 Hz trip)
    AMM->>AMM: Solve Invariant dC/df = (gamma/beta) * (C0/f0) * Droop
    AMM->>DVFS: Dispatch Clock Throttle (6.67 MW shedding)
    DVFS->>GPU: Microsecond DVFS Clock Drop (P proportional to f^2.8)
    GPU-->>DVFS: Shedding Verified in sub-2ms (Zero weight loss)
    AMM-->>RTO: Settle 200.00 MW-s Synthetic Inertia ($250/30s event)
```

---

## 3. Asynchronous Settlement & Liquidity Sweep

```mermaid
sequenceDiagram
    autonumber
    participant Gate as ActionGate
    participant Z3 as SMT/Z3 Settlement Guard
    participant Sweeper as Intraday Liquidity Sweeper
    participant FedNow as FedNow Clearing Enclave
    participant Audit as a2zsoc.com Vault

    Gate-)Z3: Post-Trade Event (Amount, HostID, Timestamp)
    Z3->>Z3: Formally Prove Balance Conservation & Zero Double-Spend
    Z3->>Sweeper: Trigger Reserve Floor Check
    Sweeper->>FedNow: Dispatch ISO 20022 camt.050.001.05 Sweep
    FedNow-->>Sweeper: Clearing Account Restocked (24/7/365)
    Z3-)Audit: Seal Cryptographic Proof in a2zsoc.com Evidence Vault
```
