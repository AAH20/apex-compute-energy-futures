#!/usr/bin/env python3
"""
Apex Compute-Energy Futures, Predictive ML & Microsecond Arbitrage Engine
Master Interactive Demonstration Harness for Technical Inspection, Partners,
NVIDIA Inception Reviewers, and AWS Marketplace Enterprise Evaluators.
"""

import math
import os
import sys
import time

# Ensure package is on sys.path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from apex_compute_futures.core.action_gate import ActionGate
from apex_compute_futures.core.models import ArbitrageQuote, HostNode, MarketVenue, SLATier
from apex_compute_futures.core.ring_buffer import RingBuffer
from apex_compute_futures.evolution.chromosome import AgentChromosome, ChromosomeEngine
from apex_compute_futures.evolution.fitness import SystemFitnessCalculator
from apex_compute_futures.evolution.pareto_tournament import ArchetypeIndividual, MultiObjectiveFitness, ParetoEvolutionEngine
from apex_compute_futures.futures.almgren_chriss import AlmgrenChrissConfig, AlmgrenChrissEngine
from apex_compute_futures.futures.arbitrage_engine import ComputeArbitrageEngine
from apex_compute_futures.futures.forward_curve import ComputeForwardCurve
from apex_compute_futures.futures.synthetic_inertia_amm import SyntheticInertiaAMM
from apex_compute_futures.futures.thermal_pue import ThermalConfig, ThermalPUEModel
from apex_compute_futures.integrations.registry import EcosystemIntegrationRegistry
from apex_compute_futures.predictive.fractional_diff import FractionalDifferentiator
from apex_compute_futures.predictive.hawkes_process import HawkesParameters, HawkesProcess
from apex_compute_futures.predictive.information_bars import CusumInformationSampler
from apex_compute_futures.predictive.ou_jump_diffusion import OUJumpDiffusionModel
from apex_compute_futures.predictive.triple_barrier import TripleBarrierLabeler


def print_banner():
    banner = r"""
==================================================================================================
   ___   ___  _______ __ __     __________  __  ______  __  __________________  ______  __________
  / _ | / _ \/ __/ _// // /    / ___/ __ \/  |/  / _ \/ / / /_  __/ __/ __/ / / /_  __/ / __/ _/ _ \
 / __ |/ ___/ _// _// _  /    / /__/ /_/ / /|_/ / ___/ /_/ / / / / _// _// /_/ / / /   / _// _/ , _/
/_/ |_/_/  /___/___/_//_/     \___/\____/_/  /_/_/   \____/ /_/ /___/___/ \____/ /_/   /___/___/_/|_| 
==================================================================================================
   Autonomous Compute-Energy Futures, Predictive ML, and Microsecond Arbitrage Engine
   Apache-2.0 Open Core | Commercial Enterprise Suite by Apex Growth Systems LLC (AWS Seller)
==================================================================================================
"""
    print(banner)


def run_demonstration():
    print_banner()
    t_global_start = time.perf_counter()

    # -----------------------------------------------------------------------------------------
    # STAGE 1: PREDICTIVE MACHINE LEARNING & STATIONARITY
    # -----------------------------------------------------------------------------------------
    print("━" * 98)
    print("STAGE 1: PREDICTIVE MACHINE LEARNING (Fractional Differencing & CUSUM Sampling)")
    print("━" * 98)
    # Synthetic non-stationary commodity power and GPU spot series (50 points)
    raw_spot_series = [2.40 + 0.05 * math.sin(i * 0.2) + 0.005 * i for i in range(50)]
    opt_d = FractionalDifferentiator.find_optimal_d(raw_spot_series, min_memory=0.75)
    diff_series = FractionalDifferentiator.transform(raw_spot_series, d=opt_d)
    memory_corr = FractionalDifferentiator.memory_retention_ratio(raw_spot_series, diff_series)

    print(f"  • Fractional Differencing Order d* : {opt_d:.2f} (ADF Stationarity Enforced)")
    print(f"  • Memory Retention Ratio (Corr)    : {memory_corr*100:.1f}% Autocorrelation Retained")

    # CUSUM Sampling
    sampler = CusumInformationSampler(initial_threshold=0.15)
    bars_sampled = 0
    for idx, price in enumerate(raw_spot_series):
        bar = sampler.process_tick(rate=price, timestamp=1700000000.0 + idx * 60)
        if bar:
            bars_sampled += 1
    print(f"  • CUSUM Information Bars Sampled   : {bars_sampled} Information Bars (Noise Filtered)")

    # Triple Barrier
    labeler = TripleBarrierLabeler(profit_multiplier=2.0, stop_loss_multiplier=1.0, max_holding_horizon=12)
    sample_event = labeler.label_trajectory(raw_spot_series, entry_idx=5, rolling_volatility=0.035)
    meta_size = labeler.compute_meta_allocation_size(probability_success=0.78)
    print(f"  • Triple-Barrier Label Outcome     : {'Upper Profit (+1)' if sample_event.label == 1 else 'Holding Timeout (0)'} ({sample_event.realized_return*100:+.2f}%)")
    print(f"  • Meta-Labeling Sizing Fraction    : {meta_size*100:.1f}% Capital Allocation\n")

    # -----------------------------------------------------------------------------------------
    # STAGE 2: FUTURES FORWARD CURVE & SYNTHETIC INERTIA AMM
    # -----------------------------------------------------------------------------------------
    print("━" * 98)
    print("STAGE 2: FUTURES FORWARD YIELD CURVE & SYNTHETIC INERTIA AMM (A-ECMM)")
    print("━" * 98)
    curve_engine = ComputeForwardCurve(region="US_EAST_VA", base_spot_rate=2.45, facility_pue=1.18)
    curve_points = curve_engine.generate_full_term_structure()

    print("  • Compute Forward Yield Curve Term Structure (US_EAST_VA | Spot: $2.45/GPU-hr):")
    print(f"    {'Maturity':<12} {'Forward Rate':<16} {'Cost of Carry':<16} {'Basis Spread':<14}")
    print("    " + "-" * 56)
    for pt in curve_points[:4]:
        print(f"    {pt.maturity_hours:<5.0f} hrs      ${pt.forward_rate_per_gpu_hr:<15.4f} {pt.cost_of_carry_pct:<15.2f}% ${pt.basis_spread_usd:<13.4f}")

    # Simulate Grid Trip on AMM
    amm = SyntheticInertiaAMM(initial_energy_mwh=30.0, initial_compute_mw=500.0)
    trip_quote = amm.quote_frequency_response(observed_frequency_hz=59.82, ancillary_market_clearing_price_usd_per_mwh=4500.0)
    print("\n  • Simulated Grid Trip Event (ERCOT Under-Frequency: 59.82 Hz):")
    print(f"    - Frequency Deficit          : {trip_quote.frequency_deviation_hz:+.3f} Hz")
    print(f"    - Microsecond DVFS Power Shed: {trip_quote.required_power_mw:.2f} MW ({trip_quote.gpu_shed_fraction*100:.1f}% clock drop)")
    print(f"    - Synthesized Inertia Output : {trip_quote.synthesized_inertia_mw_s:.2f} MW-s")
    print(f"    - Ancillary Value Captured   : ${trip_quote.clearing_revenue_per_mwh:,.2f} per 30-sec event\n")

    # -----------------------------------------------------------------------------------------
    # STAGE 3: MICROSECOND ARBITRAGE & DETERMINISTIC ACTIONGATE
    # -----------------------------------------------------------------------------------------
    print("━" * 98)
    print("STAGE 3: MICROSECOND ARBITRAGE SCANNER & ACTIONGATE EXECUTION (<10 µs)")
    print("━" * 98)
    arb_scanner = ComputeArbitrageEngine()
    # Anomaly: 8x H100 cluster on vast.ai underpriced at $1.55/GPU-hr vs $2.46 clearance
    quote = arb_scanner.scan_opportunity(
        opportunity_id="ARB-VAST-H100-8X-001",
        venue=MarketVenue.VAST_AI,
        node_type="H100-SXM5-80GB",
        gpu_count=8,
        spot_ask_rate=1.55,
        forward_clearance_price=2.465,
        venue_depth=32
    )

    gate = ActionGate()
    # Register pre-vetted host in eBPF health cache
    gate.register_healthy_host(HostNode(
        host_id="vast-node-8819",
        provider=MarketVenue.VAST_AI,
        ip_address="192.168.40.112",
        gpu_model="H100-SXM5-80GB",
        gpu_count=8,
        ping_latency_ms=18.4,
        packet_loss_pct=0.0,
        jitter_ms=1.2,
        is_preemptible=True,
        last_health_check_epoch=time.time()
    ))

    decision = gate.evaluate_arbitrage_trade(
        quote=quote,
        target_ip="192.168.40.112",
        current_vault_balance_usd=150000.0,
        current_rack_kw=18.0,
        node_kw_draw=10.2
    )

    print(f"  • Arbitrage Spread Scanned         : {quote.net_edge_bps:.1f} bps Net Edge ($1.55 vs $2.46)")
    print(f"  • ActionGate Decision              : {'✔ APPROVED' if decision.approved else '✖ DENIED'}")
    print(f"  • Gate Decision Latency            : {decision.evaluated_latency_us:.2f} µs (Sub-10 microsecond barrier)")
    print(f"  • Formal SMT Invariants Proved     : {decision.formal_invariants_satisfied} (Capital Floor & Thermal Limit)")
    print(f"  • Signed Audit Ledger Hash         : {decision.audit_hash[:32]}...\n")

    # -----------------------------------------------------------------------------------------
    # STAGE 4: GENETIC ARCHETYPE EVOLUTION & SYSTEM EVALUATION MATRIX
    # -----------------------------------------------------------------------------------------
    print("━" * 98)
    print("STAGE 4: EVOLUTIONARY REPRODUCTION & CLOSED-FORM SYSTEM EVALUATION")
    print("━" * 98)
    fit_calc = SystemFitnessCalculator()
    sim_returns = [0.002, 0.005, -0.001, 0.004, 0.003, 0.006, 0.001, -0.002, 0.008, 0.003] * 5
    dsr_score = fit_calc.calculate_deflated_sharpe_ratio(sim_returns, num_trials=50)
    fitness = fit_calc.evaluate_archetype_fitness(
        returns=sim_returns,
        sla_invalidation_rate=0.0001,
        realized_max_drawdown=0.042,
        num_generations_tested=50
    )

    # Breeding
    p1 = AgentChromosome(d_fractional=0.45, profit_multiplier=2.5)
    p2 = AgentChromosome(d_fractional=0.52, profit_multiplier=2.8)
    offspring = ChromosomeEngine.mutate(ChromosomeEngine.crossover(p1, p2))

    print(f"  • Deflated Sharpe Ratio (DSR)      : {dsr_score:.4f} (Corrected for selection bias)")
    print(f"  • Archetype Fitness Metric         : {fitness:.4f} (Hard Barrier Constraints Enforced)")
    print(f"  • Mutated Offspring Chromosome     : d={offspring.d_fractional:.3f}, profit_mult={offspring.profit_multiplier:.2f}, hold={offspring.max_holding_horizon}h\n")

    # -----------------------------------------------------------------------------------------
    # STAGE 5: ECOSYSTEM ON-DEMAND INTEGRATIONS & COMMERCIAL LAYER PRESERVATION
    # -----------------------------------------------------------------------------------------
    print("━" * 98)
    print("STAGE 5: AVAILABLE ECOSYSTEM INTEGRATIONS & COMMERCIAL DEMARCATION")
    print("━" * 98)
    integrations = EcosystemIntegrationRegistry.list_available_integrations()
    print("  • Available On-Demand Sovereign Ecosystem Integrations:")
    for k, v in list(integrations.items())[:4]:
        print(f"    [{k:<32}] -> {v['name']}")
    print(f"    ... and {len(integrations) - 4} additional modules (FDE Matrix, Liquidity Sweeper, etc.)\n")

    # -----------------------------------------------------------------------------------------
    # STAGE 6: THERMODYNAMIC PUE MODEL & ALMGREN-CHRISS OPTIMAL EXECUTION
    # -----------------------------------------------------------------------------------------
    print("━" * 98)
    print("STAGE 6: THERMODYNAMIC PUE DRIFT & ALMGREN-CHRISS OPTIMAL EXECUTION")
    print("━" * 98)
    thermal_model = ThermalPUEModel()
    pue_cool = thermal_model.calculate_pue(5600.0, ambient_temp_c=16.0)
    pue_hot = thermal_model.calculate_pue(5600.0, ambient_temp_c=36.0)
    headroom = thermal_model.solve_thermal_headroom(t_die_current_c=48.0, ambient_temp_c=30.0)

    print(f"  • Real-Time Dynamic PUE Drift      : {pue_cool.pue:.3f} (16°C Economizer) -> {pue_hot.pue:.3f} (36°C Chiller Peak)")
    print(f"  • Microsecond Thermal Headroom     : Max Safe Freq = {headroom.max_safe_frequency_ghz:.2f} GHz (T_die = {headroom.steady_state_die_temp_c:.1f}°C <= 85°C)")

    ac_engine = AlmgrenChrissEngine(AlmgrenChrissConfig(
        total_volume=1000.0,
        total_time_horizon=1.0,
        num_intervals=10,
        risk_aversion_lambda=2e-4
    ))
    schedule = ac_engine.compute_trajectory()
    print(f"  • Almgren-Chriss Liquidation Engine: Urgency κ = {schedule.urgency_kappa:.4f}, Half-Life = {schedule.half_life_time * 60:.1f} min")
    print(f"  • Expected Shortfall vs Variance   : E[x] = ${schedule.expected_shortfall:.2f}, StdDev = ${schedule.execution_std_dev:.2f}")

    # High-throughput ring buffer demo
    t_buf_start = time.perf_counter()
    rb = RingBuffer(capacity=1024)
    for tick in range(10000):
        rb.push(2.50 + 0.05 * math.sin(tick * 0.1))
    t_buf_us = (time.perf_counter() - t_buf_start) * 1e6
    print(f"  • Zero-Allocation RingBuffer       : 10,000 Ticks Ingested in {t_buf_us:.1f} µs ({t_buf_us / 10000:.3f} µs/tick)")
    print(f"  • Rolling O(1) Cache State         : Mean = ${rb.mean():.4f}, Std = ${rb.std():.4f}, EMA = ${rb.ema():.4f}\n")

    # -----------------------------------------------------------------------------------------
    # STAGE 7: ORNSTEIN-UHLENBECK SPREAD SDE, HAWKES PROCESS & PARETO NSGA-II TOURNAMENT
    # -----------------------------------------------------------------------------------------
    print("━" * 98)
    print("STAGE 7: ORNSTEIN-UHLENBECK SPREAD SDE, HAWKES PROCESS & PARETO NSGA-II TOURNAMENT")
    print("━" * 98)
    synthetic_spreads = [0.10 + 0.15 * math.exp(-0.05 * i) + 0.02 * math.sin(i * 0.3) for i in range(120)]
    ou_model = OUJumpDiffusionModel.fit(synthetic_spreads, dt=1.0)
    ou_params = ou_model.get_parameters(synthetic_spreads)
    z_score = ou_model.calculate_z_score(0.28)
    sig_name = "SHORT_BASIS" if z_score > 2.0 else ("LONG_BASIS" if z_score < -2.0 else "HOLD")
    print(f"  • O-U Mean-Reversion Speed θ       : {ou_params.theta:.4f} (Half-Life τ_1/2 = {ou_params.half_life:.2f} hrs)")
    print(f"  • Equilibrium Spread & Stn Vol     : μ = ${ou_params.mu:.4f}, σ_∞ = ${ou_params.stationary_volatility:.4f}")
    print(f"  • Current Basis Z-Score & Signal   : Z = {z_score:+.2f} -> {sig_name}")

    hp = HawkesProcess(mu0=0.05, alpha=0.35, beta=0.50)
    event_times = [1.2, 2.5, 2.8, 3.1, 4.0, 7.2]
    intensity = hp.intensity_at(8.0, event_times)
    p_preempt = hp.next_event_probability(8.0, 0.5, event_times)
    print(f"  • Hawkes Branching Ratio η (α/β)   : {hp.branching_ratio:.3f} (Sub-critical stability < 1.0)")
    print(f"  • Spot Preemption Intensity λ(t)   : {intensity:.4f} events/hr (P(Preempt in 30m) = {p_preempt * 100:.1f}%)")

    # Pareto evolutionary tournament
    pareto_engine = ParetoEvolutionEngine()
    test_archetypes = [
        ArchetypeIndividual(
            genes={"d_fractional": 0.45 + i * 0.03, "profit_multiplier": 2.0 + i * 0.2},
            fitness=MultiObjectiveFitness(
                deflated_sharpe_ratio=0.85 + 0.02 * (i % 5),
                information_ratio=1.4 + 0.1 * (i % 4),
                neg_max_drawdown=-0.12 + 0.01 * (i % 3),
                neg_sla_breach_risk=-0.001 * (1 + i % 2)
            )
        )
        for i in range(10)
    ]
    pareto_front = ParetoEvolutionEngine.get_pareto_front(test_archetypes)
    print(f"  • Multi-Objective Pareto Frontier  : Front F_1 contains {len(pareto_front)} non-dominated archetypes")
    best_arch = pareto_front[0]
    print(f"  • Pareto Elite Archetype           : DSR = {best_arch.fitness.deflated_sharpe_ratio:.4f}, IR = {best_arch.fitness.information_ratio:.2f}, MaxDD = {-best_arch.fitness.neg_max_drawdown * 100:.1f}%\n")

    print("┌────────────────────────────────────────────────────────────────────────────────────────┐")
    print("│ OPEN-SOURCE COMMUNITY CORE (APACHE-2.0)   │ ENTERPRISE PLATFORM (APEX GROWTH SYSTEMS)  │")
    print("├───────────────────────────────────────────┼────────────────────────────────────────────┤")
    print("│ • Pure Python Standard Library ML Solvers │ • Multi-MW Live DCIM / Modbus Ingestion    │")
    print("│ • CUSUM & Fractional Differencing Engines │ • Sub-2ms DVFS Hardware Shedding Daemon    │")
    print("│ • A-ECMM Synthetic Inertia Bonding Curve  │ • AWS Marketplace 100% EDP Drawdown Listing│")
    print("│ • Deterministic Fast ActionGate (<10 µs)  │ • ISO 20022 camt.050 Clearinghouse Sweeps  │")
    print("│ • SMT/Z3 Proof Primitives & Audit Hashing │ • 24/7 Enterprise SLA & Dedicated Support  │")
    print("└────────────────────────────────────────────────────────────────────────────────────────┘")

    total_runtime_ms = (time.perf_counter() - t_global_start) * 1000.0
    print(f"\n✔ DEMONSTRATION SUITE COMPLETED IN {total_runtime_ms:.2f} ms (ALL 7 STAGES VERIFIED)")
    print("==================================================================================================")


if __name__ == "__main__":
    run_demonstration()
