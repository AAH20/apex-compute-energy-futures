import unittest
import time
from apex_compute_futures.core.action_gate import ActionGate
from apex_compute_futures.core.models import ArbitrageQuote, HostNode, MarketVenue


class TestActionGate(unittest.TestCase):
    def setUp(self):
        self.gate = ActionGate(max_single_trade_usd=1000.0, max_packet_loss_pct=1.0)
        self.host = HostNode(
            host_id="test-host-1",
            provider=MarketVenue.VAST_AI,
            ip_address="10.0.0.1",
            gpu_model="H100",
            gpu_count=4,
            ping_latency_ms=15.0,
            packet_loss_pct=0.0,
            jitter_ms=1.0,
            is_preemptible=True,
            last_health_check_epoch=time.time()
        )
        self.gate.register_healthy_host(self.host)

    def test_approve_valid_arbitrage(self):
        quote = ArbitrageQuote(
            opportunity_id="OPP-01",
            venue=MarketVenue.VAST_AI,
            node_type="H100",
            gpu_count=4,
            spot_ask_rate=1.80,
            forward_clearance_price=2.40,
            gross_margin_bps=2500.0,
            estimated_slippage_bps=20.0,
            net_edge_bps=2480.0,
            latency_microseconds=5.0
        )
        decision = self.gate.evaluate_arbitrage_trade(
            quote=quote,
            target_ip="10.0.0.1",
            current_vault_balance_usd=50000.0,
            current_rack_kw=15.0,
            node_kw_draw=4.0
        )
        self.assertTrue(decision.approved)
        self.assertTrue(decision.formal_invariants_satisfied)
        self.assertGreater(len(decision.audit_hash), 10)

    def test_deny_on_thermal_breaker_limit(self):
        quote = ArbitrageQuote(
            opportunity_id="OPP-02",
            venue=MarketVenue.VAST_AI,
            node_type="H100",
            gpu_count=4,
            spot_ask_rate=1.80,
            forward_clearance_price=2.40,
            gross_margin_bps=2500.0,
            estimated_slippage_bps=20.0,
            net_edge_bps=2480.0,
            latency_microseconds=5.0
        )
        decision = self.gate.evaluate_arbitrage_trade(
            quote=quote,
            target_ip="10.0.0.1",
            current_vault_balance_usd=50000.0,
            current_rack_kw=30.0,  # 30 + 10 = 40 kW > 32 kW safe limit
            node_kw_draw=10.0
        )
        self.assertFalse(decision.approved)
        self.assertIn("Thermal breaker", decision.reason)


if __name__ == "__main__":
    unittest.main()
