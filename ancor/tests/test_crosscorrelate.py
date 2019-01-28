import unittest
import numpy as np
import matplotlib.pyplot as plt

precision = 1e-6
class Correlate:

    def tau_shift(self, series1, series2):
        n = max(series1.shape)
        m = max(series2.shape)
        correlation=np.correlate(series1, series2,mode='full')
        tau_shift = np.argmax(correlation)+1
        return tau_shift

    def distributive_identity_data(self, source, target):
        pass

class TestSyntheticData(unittest.TestCase):

    def setUp(self):
        self.corr = Correlate()

    def test_tau_shift_max(self):
        """
         Tests simple tau shifts from a synthetic dataset

         currently not implemented

        """
        source_tau_shift = 0.0
        target_tau_shift = 5.0
        self.assertAlmostEqual(source_tau_shift,target_tau_shift,5,"tau shift not implemented")

    def test_correlation_identity(self):
        """
         Tests (g*h)*(g*h) = (g*g)*(h*h) identity

         currently not implemented

        """
        source =1
        target=0
        self.assertAlmostEqual(source,target, 5, "corr identity not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        source = 1
        target = 0
        self.assertAlmostEqual(source,target,5,'not implemented')


if __name__ == '__main__':
    unittest.main()