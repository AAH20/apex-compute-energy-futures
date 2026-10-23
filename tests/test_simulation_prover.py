"""Unit tests for institutional stress testing, formal settlement prover, and audit ledger."""

import hashlib
import unittest

from apex_compute_futures.core.settlement_prover import (
    AuditBlock,
    Constant,
    CryptographicAuditLedger,
    Ed25519Signer,
    FormalSettlementProver,
    Ge,
    Le,
    MerkleTree,
    ProofCertificate,
    ProofStatus,
    SMTConstraintSolver,
    Symbol,
)
from apex_compute_futures.simulation.stress_tester import (
    InstitutionalStressTestingEngine,
    StressScenarioType,
)


class TestInstitutionalStressTester(unittest.TestCase):
    """Verifies Black-Swan crisis simulation, VaR/CVaR, and survival calculations."""

    @classmethod
    def setUpClass(cls):
        cls.engine = InstitutionalStressTestingEngine(default_seed=1337)

    def test_winter_storm_uri_scenario(self):
        scenario = self.engine.create_winter_storm_uri_scenario(initial_capital=10_000_000.0)
        self.assertEqual(scenario.scenario_type, StressScenarioType.WINTER_STORM_URI_2)
        self.assertEqual(scenario.grid_frequency_hz, 59.35)
        self.assertEqual(scenario.power_lmp_per_mwh, 9000.0)
        self.assertEqual(scenario.datacenter_curtailment_pct, 45.0)

        # Microsecond benchmark: 10 paths, 8 time steps
        result = self.engine.run_stress_test(scenario, num_paths=10, time_steps=8, seed=42)
        self.assertGreater(result.max_drawdown_pct, 5.0)
        self.assertGreater(result.historical_var_999, 0.0)
        self.assertGreater(result.cvar_999, 0.0)
        self.assertGreaterEqual(result.cvar_999, result.historical_var_999 * 0.95)
        self.assertGreater(len(result.fan_chart_percentiles["p50"]), 0)

    def test_frontier_model_launch_shock(self):
        scenario = self.engine.create_frontier_model_shock_scenario()
        self.assertEqual(scenario.gpu_supply_withdrawal_count, 60_000)
        self.assertEqual(scenario.retail_preemption_rate, 0.85)
        self.assertEqual(scenario.forward_basis_shock_bps, 600.0)

        result = self.engine.run_stress_test(scenario, num_paths=10, time_steps=8, seed=42)
        self.assertGreater(result.parametric_var_999, 0.0)
        self.assertGreater(result.max_drawdown_usd, 0.0)
        self.assertGreaterEqual(result.system_survival_probability, 0.0)

    def test_subsea_terabit_partition(self):
        scenario = self.engine.create_subsea_partition_scenario()
        self.assertEqual(scenario.network_latency_spike_ms, 350.0)
        self.assertEqual(scenario.packet_loss_pct, 8.5)

        result = self.engine.run_stress_test(scenario, num_paths=10, time_steps=8, seed=42)
        self.assertGreater(result.worst_case_loss_usd, 0.0)
        self.assertEqual(len(result.sample_trajectories), 5)


class TestFormalSettlementProver(unittest.TestCase):
    """Verifies formal SMT invariants, double-allocation immunity, and thermal boundaries."""

    def setUp(self):
        # High-performance microsecond verification mode
        self.prover = FormalSettlementProver(
            reserve_floor_usd=1_000_000.0,
            max_rack_kw=40.0,
            t_crit_c=85.0,
            use_curve_signatures=False,
        )

    def test_trade_settlement_success_and_ledger_append(self):
        ok, cert, block = self.prover.prove_trade_settlement(
            trade_id="TR_101",
            instance_ids=["h100_node_01", "h100_node_02"],
            cash_delta_usd=-100_000.0,
            lease_value_usd=120_000.0,
            vault_balance_usd=5_000_000.0,
        )
        self.assertTrue(ok)
        self.assertEqual(cert.status, ProofStatus.PROVED)
        self.assertIsNotNone(block)
        self.assertEqual(block.block_index, 0)
        self.assertEqual(block.action_type, "TRADE_SETTLEMENT")

    def test_catches_double_allocation(self):
        ok1, _, _ = self.prover.prove_trade_settlement(
            trade_id="TR_A",
            instance_ids=["gpu_slot_alpha"],
            cash_delta_usd=-10_000.0,
            lease_value_usd=15_000.0,
            vault_balance_usd=2_000_000.0,
        )
        self.assertTrue(ok1)

        ok2, cert2, block2 = self.prover.prove_trade_settlement(
            trade_id="TR_B",
            instance_ids=["gpu_slot_alpha"],
            cash_delta_usd=-10_000.0,
            lease_value_usd=15_000.0,
            vault_balance_usd=2_000_000.0,
        )
        self.assertFalse(ok2)
        self.assertEqual(cert2.status, ProofStatus.VIOLATED)
        self.assertIn("Double-allocation violated", cert2.failure_reason)
        self.assertIsNone(block2)

        self.prover.release_allocations(["gpu_slot_alpha"])
        ok3, cert3, _ = self.prover.prove_trade_settlement(
            trade_id="TR_C",
            instance_ids=["gpu_slot_alpha"],
            cash_delta_usd=-10_000.0,
            lease_value_usd=15_000.0,
            vault_balance_usd=2_000_000.0,
        )
        self.assertTrue(ok3)

    def test_catches_capital_conservation_violation(self):
        ok, cert, block = self.prover.prove_trade_settlement(
            trade_id="TR_FAIL_CAP",
            instance_ids=["gpu_slot_beta"],
            cash_delta_usd=-400_000.0,
            lease_value_usd=500_000.0,
            vault_balance_usd=1_200_000.0,
        )
        self.assertFalse(ok)
        self.assertEqual(cert.status, ProofStatus.VIOLATED)
        self.assertIn("reserve_floor", cert.failure_reason)
        self.assertIsNone(block)

    def test_battery_dispatch_and_boundary_checks(self):
        ok, cert, block = self.prover.prove_battery_dispatch(
            dispatch_id="BAT_NORM",
            current_soc_mwh=60.0,
            dispatch_power_mw=25.0,
            duration_hours=1.0,
        )
        self.assertTrue(ok)
        self.assertEqual(cert.status, ProofStatus.PROVED)

        ok_fail, cert_fail, _ = self.prover.prove_battery_dispatch(
            dispatch_id="BAT_OVER_DISCHARGE",
            current_soc_mwh=60.0,
            dispatch_power_mw=60.0,
            duration_hours=1.0,
        )
        self.assertFalse(ok_fail)
        self.assertEqual(cert_fail.status, ProofStatus.VIOLATED)

    def test_dvfs_thermal_and_power_invariants(self):
        ok, cert, _ = self.prover.prove_dvfs_throttling(
            rack_id="RACK_01",
            target_frequency_mhz=1500.0,
            f_min_mhz=1000.0,
            f_max_mhz=2100.0,
            current_rack_power_kw=30.0,
            max_rack_kw=40.0,
            current_rack_temp_c=70.0,
            t_crit_c=85.0,
        )
        self.assertTrue(ok)

        ok_therm, cert_therm, _ = self.prover.prove_dvfs_throttling(
            rack_id="RACK_02",
            target_frequency_mhz=2400.0,
            f_min_mhz=1000.0,
            f_max_mhz=2100.0,
            current_rack_power_kw=38.0,
            max_rack_kw=40.0,
            current_rack_temp_c=82.0,
            t_crit_c=85.0,
        )
        self.assertFalse(ok_therm)
        self.assertEqual(cert_therm.status, ProofStatus.VIOLATED)


class TestMerkleTreeAndEd25519Ledger(unittest.TestCase):
    """Verifies RFC 6962 Merkle tree proofs, Ed25519 signing, and ledger tampering defense."""

    def test_merkle_tree_proof_and_verification(self):
        items = [b"tx_in_0", b"tx_in_1", b"tx_in_2", b"tx_in_3"]
        tree = MerkleTree(items)
        root = tree.get_root_hash()
        self.assertEqual(len(root), 64)

        proof = tree.generate_inclusion_proof(2)
        valid = MerkleTree.verify_inclusion_proof(items[2], proof, root)
        self.assertTrue(valid)

        invalid = MerkleTree.verify_inclusion_proof(b"tx_in_tampered", proof, root)
        self.assertFalse(invalid)

    def test_ed25519_signature_hashes_and_verification(self):
        priv, pub = Ed25519Signer.get_default_keypair()
        msg = b"APEX_INSTITUTIONAL_SETTLEMENT_PAYLOAD"
        sig_hash = Ed25519Signer.compute_signature_hash(msg, priv)
        self.assertEqual(len(sig_hash), 128)

        # Valid signature hash
        self.assertTrue(Ed25519Signer.verify_signature_hash(msg, sig_hash, priv))

        # Tampered message
        self.assertFalse(Ed25519Signer.verify_signature_hash(b"TAMPERED_PAYLOAD", sig_hash, priv))

    def test_ledger_integrity_and_tampering_detection(self):
        ledger = CryptographicAuditLedger(use_curve_signatures=False)
        cert = ProofCertificate(
            status=ProofStatus.PROVED,
            assertions=["True"],
            counterexample=None,
            failure_reason=None,
            proof_hash=hashlib.sha256(b"dummy_proof").hexdigest(),
            latency_us=2.0,
        )

        b0 = ledger.append_block("root_0", "ACTION_0", cert)
        b1 = ledger.append_block("root_1", "ACTION_1", cert)
        b2 = ledger.append_block("root_2", "ACTION_2", cert)

        intact, msg = ledger.verify_ledger_integrity()
        self.assertTrue(intact)

        # Tamper with block 1's prev_block_hash
        original_prev = b1.prev_block_hash
        b1.prev_block_hash = "00" * 32
        tampered, t_msg = ledger.verify_ledger_integrity()
        self.assertFalse(tampered)
        self.assertIn("Broken hash chain", t_msg)

        # Restore
        b1.prev_block_hash = original_prev
        intact_again, _ = ledger.verify_ledger_integrity()
        self.assertTrue(intact_again)


if __name__ == "__main__":
    unittest.main()
