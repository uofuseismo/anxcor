import unittest
class TestBandpassFiltering(unittest.TestCase):

    def test_filter_frequency_out_of_band(self):
        """
         tests to make sure that frequencies added outside the bandpassed range are not retained

         currently not implemented

        """
        source_tau_shift = 0.0
        target_tau_shift = 5.0
        self.assertAlmostEqual(source_tau_shift,target_tau_shift,5,"test not implemented")

    def test_filter_frequency_in_band(self):
        """
         tests to ensure frequencies are retained in band

         currently not implemented

        """
        source_tau_shift = 0.0
        target_tau_shift = 5.0
        self.assertAlmostEqual(source_tau_shift,target_tau_shift,5,"test not implemented")

    def test_phase_shift_not_introduced(self):
        """
         tests to ensure phase shifts are not introduced to bandpass filter

         currently not implemented

        """
        source_tau_shift = 0.0
        target_tau_shift = 5.0
        self.assertAlmostEqual(source_tau_shift,target_tau_shift,5,"test not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        source = 1
        target = 0
        self.assertAlmostEqual(source,target,5,'not implemented')



if __name__ == '__main__':
    unittest.main()