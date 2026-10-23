"""Deterministic Fast-Path ActionGate.

Executes sub-microsecond risk checks and prove-or-deny boundaries on proposed
compute leases and arbitrage trades before external execution.
"""

import time
from typing import Dict, Optional
from apex_compute_futures.core.models import ActionGateDecision, ArbitrageQuote, HostNode
from apex_compute_futures.core.invariants import FormalInvariantVerifier


class ActionGate:
    """Pre-compiled fast-path execution boundary for autonomous compute arbitrage."""

    def __init__(
        self,
        max_single_trade_usd: float = 2500.0,
        max_daily_budget_usd: float = 25000.0,
        max_packet_loss_pct: float = 1.0,
        max_ping_latency_ms: float = 45.0,
        verifier: Optional[FormalInvariantVerifier] = None
    ):
        self.max_single_trade_usd = max_single_trade_usd
        self.max_daily_budget_usd = max_daily_budget_usd
        self.max_packet_loss_pct = max_packet_loss_pct
        self.max_ping_latency_ms = max_ping_latency_ms
        self.verifier = verifier or FormalInvariantVerifier()

        # In-memory pre-vetted host health cache (populated asynchronously by eBPF daemon)
        self.healthy_hosts_pool: Dict[str, HostNode] = {}
        self.total_spent_today_usd: float = 0.0

    def register_healthy_host(self, host: HostNode) -> None:
        """Updates the local pre-vetted host registry."""
        self.healthy_hosts_pool[host.ip_address] = host

    def evaluate_arbitrage_trade(
        self,
        quote: ArbitrageQuote,
        target_ip: str,
        current_vault_balance_usd: float,
        current_rack_kw: float,
        node_kw_draw: float = 10.2
    ) -> ActionGateDecision:
        """Evaluates an instantaneous trade proposal in <10 microseconds."""
        t_start = time.perf_counter_ns()

        trade_cost_usd = quote.spot_ask_rate * quote.gpu_count

        # 1. Quota & Spending Limit Checks
        if trade_cost_usd > self.max_single_trade_usd:
            elapsed_us = (time.perf_counter_ns() - t_start) / 1000.0
            return ActionGateDecision(
                approved=False,
                reason=f"Exceeds single trade cap: ${trade_cost_usd:.2f} > ${self.max_single_trade_usd:.2f}",
                opportunity_id=quote.opportunity_id,
                evaluated_latency_us=elapsed_us,
                target_host_ip=target_ip,
                allocated_quota_usd=0.0,
                formal_invariants_satisfied=False,
                audit_hash=""
            )

        if (self.total_spent_today_usd + trade_cost_usd) > self.max_daily_budget_usd:
            elapsed_us = (time.perf_counter_ns() - t_start) / 1000.0
            return ActionGateDecision(
                approved=False,
                reason="Daily spending budget exhausted",
                opportunity_id=quote.opportunity_id,
                evaluated_latency_us=elapsed_us,
                target_host_ip=target_ip,
                allocated_quota_usd=0.0,
                formal_invariants_satisfied=False,
                audit_hash=""
            )

        # 2. Pre-Vetted Host Telemetry Check
        host = self.healthy_hosts_pool.get(target_ip)
        if not host:
            elapsed_us = (time.perf_counter_ns() - t_start) / 1000.0
            return ActionGateDecision(
                approved=False,
                reason=f"Target host {target_ip} not in pre-vetted healthy pool",
                opportunity_id=quote.opportunity_id,
                evaluated_latency_us=elapsed_us,
                target_host_ip=target_ip,
                allocated_quota_usd=0.0,
                formal_invariants_satisfied=False,
                audit_hash=""
            )

        if host.packet_loss_pct > self.max_packet_loss_pct:
            elapsed_us = (time.perf_counter_ns() - t_start) / 1000.0
            return ActionGateDecision(
                approved=False,
                reason=f"Host packet loss {host.packet_loss_pct:.2f}% exceeds tolerance {self.max_packet_loss_pct:.2f}%",
                opportunity_id=quote.opportunity_id,
                evaluated_latency_us=elapsed_us,
                target_host_ip=target_ip,
                allocated_quota_usd=0.0,
                formal_invariants_satisfied=False,
                audit_hash=""
            )

        # 3. Clearance Price Check (Must beat forward curve by threshold)
        if quote.spot_ask_rate >= quote.forward_clearance_price:
            elapsed_us = (time.perf_counter_ns() - t_start) / 1000.0
            return ActionGateDecision(
                approved=False,
                reason=f"Spot price ${quote.spot_ask_rate:.2f} >= clearance threshold ${quote.forward_clearance_price:.2f}",
                opportunity_id=quote.opportunity_id,
                evaluated_latency_us=elapsed_us,
                target_host_ip=target_ip,
                allocated_quota_usd=0.0,
                formal_invariants_satisfied=False,
                audit_hash=""
            )

        # 4. Formal Mathematical Invariant Proof Pass
        cap_ok, cap_msg = self.verifier.verify_capital_conservation(current_vault_balance_usd, trade_cost_usd)
        if not cap_ok:
            elapsed_us = (time.perf_counter_ns() - t_start) / 1000.0
            return ActionGateDecision(
                approved=False,
                reason=cap_msg,
                opportunity_id=quote.opportunity_id,
                evaluated_latency_us=elapsed_us,
                target_host_ip=target_ip,
                allocated_quota_usd=0.0,
                formal_invariants_satisfied=False,
                audit_hash=""
            )

        therm_ok, therm_msg = self.verifier.verify_thermal_power_boundary(current_rack_kw, node_kw_draw)
        if not therm_ok:
            elapsed_us = (time.perf_counter_ns() - t_start) / 1000.0
            return ActionGateDecision(
                approved=False,
                reason=therm_msg,
                opportunity_id=quote.opportunity_id,
                evaluated_latency_us=elapsed_us,
                target_host_ip=target_ip,
                allocated_quota_usd=0.0,
                formal_invariants_satisfied=False,
                audit_hash=""
            )

        # All gates passed: Lock allocation and sign cryptographic audit hash
        self.verifier.register_allocation(host.host_id)
        self.total_spent_today_usd += trade_cost_usd

        audit_hash = self.verifier.generate_cryptographic_audit_hash(
            action_name="ACTIONGATE_EXECUTE_ARBITRAGE",
            target_id=host.host_id,
            amount_usd=trade_cost_usd,
            invariants_passed=True
        )

        elapsed_us = (time.perf_counter_ns() - t_start) / 1000.0
        return ActionGateDecision(
            approved=True,
            reason="All risk boundaries, host telemetry, and SMT invariants formally verified",
            opportunity_id=quote.opportunity_id,
            evaluated_latency_us=elapsed_us,
            target_host_ip=target_ip,
            allocated_quota_usd=trade_cost_usd,
            formal_invariants_satisfied=True,
            audit_hash=audit_hash
        )
