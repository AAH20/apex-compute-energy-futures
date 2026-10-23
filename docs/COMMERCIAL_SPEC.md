# Apex Compute-Energy Futures: Commercial & AWS Marketplace Specification

This document defines the commercial licensing envelope and go-to-market structure for `apex-compute-energy-futures`.

---

## 1. Open-Core Demarcation Matrix

The repository strictly separates the open-source community foundation from the enterprise commercial platform:

| Capability / Module | Community Open-Core (Apache-2.0) | Enterprise Platform (Apex Growth Systems LLC) |
| :--- | :--- | :--- |
| **Combinatorial Forward Curve Engine** | Included (Pure Stdlib Python) | Included + Real-Time Bloomberg/Refinitiv Feeds |
| **Fractional Differentiation Engine** | Included (Memory-Preserving) | Included + Distributed GPU Accelerated Kernels |
| **CUSUM Information-Bar Sampling** | Included | Included |
| **A-ECMM Synthetic Inertia AMM** | Included (Mathematical Model) | Included + Real-Time Substation SCADA Gateway |
| **Deterministic ActionGate** | Included (<10 µs Local Boundary) | Included + Multi-Tenant EKS Cluster Enclave |
| **Live Hardware Actuation** | Read-Only (Non-Actuating) | Sub-2ms DVFS Clock Locking (IPMI/BMC/NVML) |
| **Facility Ingestion** | Synthetic & File Ingestors | Direct BACnet, Modbus, SNMP DCIM Connectors |
| **SLA & Compliance Attestation** | Standard Hash Ledger | a2zsoc.com Tamper-Evident Evidence Vault |
| **Deployment Footprint** | Local Python CLI / Script | Single-Tenant Customer VPC via PrivateLink |
| **AWS Marketplace Listing** | Community Open Source | **100% AWS EDP Drawdown Eligible** |

---

## 2. AWS Marketplace Commercial Mechanics

* **Seller Entity:** **Apex Growth Systems LLC** (Arizona LLC, ACC 25121151, EIN 61-2391219).
* **Tax Status:** Foreign-owned disregarded entity, verified W-8BEN-E (0% withholding treaty rate).
* **Banking:** Mercury Bank (US) with direct USD wire/ACH routing.
* **AWS Marketplace Seller Status:** Verified Seller.
* **NVIDIA Inception Status:** Member Company (October 2026).

### 100% EDP Drawdown (The Enterprise Procurement Advantage)
Enterprise customers burn pre-committed AWS Enterprise Discount Program (EDP) balances. 
1. The transaction is billed directly to the customer's unified AWS consolidated bill.
2. 100% of the contract value draws down against the customer's mandatory minimum spend commitment.
3. No legal procurement friction: standard AWS Standard Contract for Marketplace (SCMP) with a "Read-Only Safety Invariant" rider.

---

## 3. Commercial SKU Schedule

```
========================================================================================================================
SKU CODE              TIER NAME                   PRICE               BILLING CADENCE       METERING DIMENSION
========================================================================================================================
[SKU-FUT-01]          Community Core              $0                  Open Source           Apache-2.0 Public
[SKU-FUT-02]          Foundation Enclave          $3,500 / month      Monthly SaaS          Up to 64 GPUs / 250 kW
[SKU-FUT-03]          Autonomous Cluster Pro      $12,500 / month     Annual Contract       Up to 512 GPUs / 2.5 MW
[SKU-FUT-04]          Sovereign Megawatt          $350,000 / year     Custom Private Offer  Multi-MW Campus (10 MW+)
========================================================================================================================
```

### Consumption Dimensions (Metered via `BatchMeterUsage`):
* `Managed_GPU_Hours`: **\$0.08 / GPU-hr** (overage beyond contracted node baseline).
* `Arbitraged_Yield_Share`: **\$15.00 / MWh** (metered when DVFS power shedding captures emergency grid ancillary spikes).
* `Formal_Proof_Executions`: **\$5.00 / 10k proofs** (metered by SMT/Z3 invariant engine).

---

## 4. Contact & Inquiries
* **Enterprise Licensing & Custom Private Offers:** `sales@a2zsoc.com`
* **Technical Inquiries:** `aah@a2zsoc.com`
* **Entity:** Apex Growth Systems LLC, Arizona, USA.
* **Web:** [https://a2zsoc.com](https://a2zsoc.com)
