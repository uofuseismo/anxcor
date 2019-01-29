import unittest
from ancor import numpy_array_signal_functions
from ancor import numpy_vertical_matrix_signal_functions
from ancor import stream_signal_functions
class TestStreamBandpassFiltering(unittest.TestCase):

    def test_filter_frequency_out_of_band(self):
        """
         tests to make sure that frequencies added outside the bandpassed range are not retained

         currently not implemented

        """
        source_tau_shift = 0.0
        target_tau_shift = 5.0
        source = stream_signal_functions.bandpass(source_tau_shift)
        self.assertAlmostEqual(source,target_tau_shift,5,"test not implemented")

    def test_filter_frequency_in_band(self):
        """
         tests to ensure frequencies are retained in band

         currently not implemented

        """
        source = stream_signal_functions.bandpass(None)
        target_tau_shift = 5.0
        self.assertAlmostEqual(source,target_tau_shift,5,"test not implemented")

    def test_phase_shift_not_introduced(self):
        """
         tests to ensure phase shifts are not introduced to bandpass filter

         currently not implemented

        """
        source = stream_signal_functions.bandpass(None)
        target_tau_shift = 5.0
        self.assertAlmostEqual(source,target_tau_shift,5,"test not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        source = stream_signal_functions.bandpass(None)
        target = 0
        self.assertAlmostEqual(source,target,5,'not implemented')

class TestNumpyBandpassFiltering(unittest.TestCase):

    def test_filter_frequency_out_of_band(self):
        """
         tests to make sure that frequencies added outside the bandpassed range are not retained

         currently not implemented

        """
        source_tau_shift = 0.0
        target_tau_shift = 5.0
        source = numpy_array_signal_functions.bandpass(source_tau_shift)
        self.assertAlmostEqual(source,target_tau_shift,5,"test not implemented")

    def test_filter_frequency_in_band(self):
        """
         tests to ensure frequencies are retained in band

         currently not implemented

        """
        source = numpy_array_signal_functions.bandpass(None)
        target_tau_shift = 5.0
        self.assertAlmostEqual(source,target_tau_shift,5,"test not implemented")

    def test_phase_shift_not_introduced(self):
        """
         tests to ensure phase shifts are not introduced to bandpass filter

         currently not implemented

        """
        source = numpy_array_signal_functions.bandpass(None)
        target_tau_shift = 5.0
        self.assertAlmostEqual(source,target_tau_shift,5,"test not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        source =numpy_array_signal_functions.bandpass(None)
        target = 0
        self.assertAlmostEqual(source,target,5,'not implemented')

class TestNumpyVerticalArrayBandpassFiltering(unittest.TestCase):

    def test_filter_frequency_out_of_band(self):
        """
         tests to make sure that frequencies added outside the bandpassed range are not retained

         currently not implemented

        """
        source_tau_shift = 0.0
        target_tau_shift = 5.0
        source = numpy_vertical_matrix_signal_functions.bandpass(source_tau_shift)
        self.assertAlmostEqual(source,target_tau_shift,5,"test not implemented")

    def test_filter_frequency_in_band(self):
        """
         tests to ensure frequencies are retained in band

         currently not implemented

        """
        source = numpy_vertical_matrix_signal_functions.bandpass(None)
        target_tau_shift = 5.0
        self.assertAlmostEqual(source,target_tau_shift,5,"test not implemented")

    def test_phase_shift_not_introduced(self):
        """
         tests to ensure phase shifts are not introduced to bandpass filter

         currently not implemented

        """
        source =numpy_vertical_matrix_signal_functions.bandpass(None)
        target_tau_shift = 5.0
        self.assertAlmostEqual(source,target_tau_shift,5,"test not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        source =numpy_vertical_matrix_signal_functions.bandpass(None)
        target = 0
        self.assertAlmostEqual(source,target,5,'not implemented')


if __name__ == '__main__':
    unittest.main()