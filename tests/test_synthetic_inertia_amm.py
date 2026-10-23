import unittest
from apex_compute_futures.futures.synthetic_inertia_amm import SyntheticInertiaAMM


class TestSyntheticInertiaAMM(unittest.TestCase):
    def setUp(self):
        self.amm = SyntheticInertiaAMM(initial_energy_mwh=25.0, initial_compute_mw=500.0)

    def test_invariant_k_positive(self):
        self.assertGreater(self.amm.state.invariant_k, 0.0)

    def test_under_frequency_shedding(self):
        quote = self.amm.quote_frequency_response(59.80)
        self.assertEqual(quote.frequency_deviation_hz, -0.20)
        self.assertGreater(quote.required_power_mw, 0.0)
        self.assertGreater(quote.gpu_shed_fraction, 0.0)
        self.assertGreater(quote.clearing_revenue_per_mwh, 0.0)

    def test_normal_frequency_zero_shedding(self):
        quote = self.amm.quote_frequency_response(60.00)
        self.assertEqual(quote.required_power_mw, 0.0)
        self.assertEqual(quote.gpu_shed_fraction, 0.0)


if __name__ == "__main__":
    unittest.main()
