import unittest
from apex_compute_futures.futures.forward_curve import ComputeForwardCurve
from apex_compute_futures.core.models import SLATier


class TestForwardCurve(unittest.TestCase):
    def setUp(self):
        self.curve = ComputeForwardCurve(region="US_EAST", base_spot_rate=2.50)

    def test_term_structure_generation(self):
        points = self.curve.generate_full_term_structure()
        self.assertEqual(len(points), 7)
        self.assertEqual(points[0].maturity_hours, 1.0)
        self.assertEqual(points[-1].maturity_hours, 2160.0)

    def test_sla_premium(self):
        p_normal = self.curve.price_forward(24.0, SLATier.FAULT_TOLERANT)
        p_mission = self.curve.price_forward(24.0, SLATier.MISSION_CRITICAL)
        self.assertGreater(p_mission.forward_rate_per_gpu_hr, p_normal.forward_rate_per_gpu_hr)


if __name__ == "__main__":
    unittest.main()
