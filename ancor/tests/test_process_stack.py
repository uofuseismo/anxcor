import unittest

class TestSignalProcessingStack(unittest.TestCase):

    def test_signal_processing_stack(self):
        """
         ensure the signal processing steps are what we expect

         currently not implemented

        """
        source_tau_shift = 0.0
        target_tau_shift = 5.0
        self.assertAlmostEqual(source_tau_shift,target_tau_shift,5,"sig process steps not implemented")

class TestCrossCorrelationStack(unittest.TestCase):

    def test_crosscorrelation_stack(self):
        """
         ensure the crosscorrelation steps including noise processing are what we expect

         currently not implemented

        """
        source_tau_shift = 0.0
        target_tau_shift = 5.0
        self.assertAlmostEqual(source_tau_shift,target_tau_shift,5,"crosscorr steps not implemented")



if __name__ == '__main__':
    unittest.main()