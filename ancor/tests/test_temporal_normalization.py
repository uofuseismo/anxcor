import unittest
class TestTemporalNormalization(unittest.TestCase):

    def test_temporal_normalization_spectrum(self):
        """
         ensure the temporal normalized spectrum is what we expect

         currently not implemented

        """
        source_tau_shift = 0.0
        target_tau_shift = 5.0
        self.assertAlmostEqual(source_tau_shift,target_tau_shift,5,"normalization spectrum test not implemented")

    def test_temporal_normalization_norm(self):
        """
         check to see the temporal normalization norm is what is expected

        """
        source =1
        target=0
        self.assertAlmostEqual(source,target, 5, "normalization norm not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        source = 1
        target = 0
        self.assertAlmostEqual(source,target,5,'not implemented')

if __name__ == '__main__':
    unittest.main()