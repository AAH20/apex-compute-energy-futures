import unittest
from apex_compute_futures.integrations.registry import EcosystemIntegrationRegistry


class TestIntegrationsRegistry(unittest.TestCase):
    def test_list_integrations_contains_all_core_repos(self):
        integrations = EcosystemIntegrationRegistry.list_available_integrations()
        self.assertIn("datacenter_np_hard_kernel", integrations)
        self.assertIn("gigawatt_ride_through_amm_kernel", integrations)
        self.assertIn("hft_microstructure_kernel", integrations)
        self.assertIn("apex_quant_whale_kernel", integrations)
        self.assertIn("agentic_fintech_kernel", integrations)
        self.assertIn("Apex_FDE_Matrix", integrations)
        self.assertIn("intraday_liquidity_orchestrator", integrations)

    def test_load_adapter_dynamic(self):
        dcc_adapter = EcosystemIntegrationRegistry.load_adapter("datacenter_np_hard_kernel")
        self.assertIsNotNone(dcc_adapter)
        telemetry = dcc_adapter.get_facility_telemetry()
        self.assertTrue(telemetry["read_only_boundary_enforced"])


if __name__ == "__main__":
    unittest.main()
