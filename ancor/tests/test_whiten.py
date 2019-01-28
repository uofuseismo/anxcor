import unittest
class TestWhitening(unittest.TestCase):

    def test_whitening_spectrum(self):
        """
         ensure the whitened spectrum is what we expect

         currently not implemented

        """
        source_tau_shift = 0.0
        target_tau_shift = 5.0
        self.assertAlmostEqual(source_tau_shift,target_tau_shift,5,"whitening spectrum test not implemented")

    def test_whitening_norm(self):
        """
         check to see the whitening norm difference is small

        """
        source =1
        target=0
        self.assertAlmostEqual(source,target, 5, "whitening norm test not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        source = 1
        target = 0
        self.assertAlmostEqual(source,target,5,'not implemented')


if __name__ == '__main__':
    unittest.main()