import unittest
from apex_compute_futures.predictive.fractional_diff import FractionalDifferentiator


class TestFractionalDiff(unittest.TestCase):
    def test_weights_decay(self):
        weights = FractionalDifferentiator.get_weights(d=0.45, size=20)
        self.assertEqual(len(weights), 20)
        self.assertEqual(weights[0], 1.0)
        self.assertLess(abs(weights[-1]), abs(weights[1]))

    def test_transform_preserves_length(self):
        data = [10.0 + i * 0.5 for i in range(30)]
        diff = FractionalDifferentiator.transform(data, d=0.40)
        self.assertEqual(len(diff), len(data))


if __name__ == "__main__":
    unittest.main()
