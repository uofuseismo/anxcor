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
    simple_synth1 = np.zeros((1000))
    simple_synth1[0]=1
    simple_synth2 = np.zeros((1000))
    simple_synth2[500]=1

    noisy_synth1= np.random.rand(1000)
    noisy_synth2= np.concatenate((np.random.rand(500),noisy_synth1[500::]))

    def setUp(self):
        self.corr = Correlate()

    def test_tau_shift_max(self):
        """
         Tests simple tau shifts from a synthetic dataset

         currently not implemented

        """
        source_tau_shift = 0.0
        target_tau_shift = float(self.corr.tau_shift(self.simple_synth1,self.simple_synth2))
        self.assertAlmostEqual(source_tau_shift,target_tau_shift,5,"tau shift not implemented")

    def test_correlation_identity(self):
        """
         Tests (g*h)*(g*h) = (g*g)*(h*h) identity

         currently not implemented

        """
        source =1
        target=0
        self.assertAlmostEqual(source,target, 5, "corr identity not implemented")







if __name__ == '__main__':
    unittest.main()