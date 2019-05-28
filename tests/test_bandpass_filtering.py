import unittest
from ancor_processor import BandPass
class TestStreamBandpassFiltering(unittest.TestCase):

    def test_filter_frequency_out_of_band(self):
        """
         tests to make sure that frequencies added outside the bandpassed range are not retained

         currently not implemented

        """
        target=0
        source = stream_signal_functions.bandpass(None)
        self.assertAlmostEqual(source,target,5,"test not implemented")

    def test_filter_frequency_in_band(self):
        """
         tests to ensure frequencies are retained in band

         currently not implemented

        """
        source = stream_signal_functions.bandpass(None)
        target = 0
        self.assertAlmostEqual(source,target,5,"test not implemented")

    def test_phase_shift_not_introduced(self):
        """
         tests to ensure phase shifts are not introduced to bandpass filter

         currently not implemented

        """
        target=0
        source = stream_signal_functions.bandpass(None)
        self.assertAlmostEqual(source,target,5,"test not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        target=0
        source = stream_signal_functions.bandpass(None)
        self.assertAlmostEqual(source,target,5,'not implemented')

class TestNumpyBandpassFiltering(unittest.TestCase):

    def test_filter_frequency_out_of_band(self):
        """
         tests to make sure that frequencies added outside the bandpassed range are not retained

         currently not implemented

        """
        target=0
        source = numpy_array_signal_functions.bandpass(None)
        self.assertAlmostEqual(source,target,5,"test not implemented")

    def test_filter_frequency_in_band(self):
        """
         tests to ensure frequencies are retained in band

         currently not implemented

        """
        target=0
        source = numpy_array_signal_functions.bandpass(None)
        self.assertAlmostEqual(source,target,5,"test not implemented")

    def test_phase_shift_not_introduced(self):
        """
         tests to ensure phase shifts are not introduced to bandpass filter

         currently not implemented

        """
        target=0
        source = numpy_array_signal_functions.bandpass(None)
        self.assertAlmostEqual(source,target,5,"test not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        target = 0
        source =numpy_array_signal_functions.bandpass(None)
        self.assertAlmostEqual(source,target,5,'not implemented')

class TestNumpyVerticalArrayBandpassFiltering(unittest.TestCase):

    def test_filter_frequency_out_of_band(self):
        """
         tests to make sure that frequencies added outside the bandpassed range are not retained

         currently not implemented

        """
        target = 0
        source = numpy_vertical_matrix_signal_functions.bandpass(None)
        self.assertAlmostEqual(source,target,5,"test not implemented")

    def test_filter_frequency_in_band(self):
        """
         tests to ensure frequencies are retained in band

         currently not implemented

        """
        target=0
        source = numpy_vertical_matrix_signal_functions.bandpass(None)
        self.assertAlmostEqual(source,target,5,"test not implemented")

    def test_phase_shift_not_introduced(self):
        """
         tests to ensure phase shifts are not introduced to bandpass filter

         currently not implemented

        """
        target=0
        source =numpy_vertical_matrix_signal_functions.bandpass(None)
        self.assertAlmostEqual(source,target,5,"test not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        target=0
        source =numpy_vertical_matrix_signal_functions.bandpass(None)
        self.assertAlmostEqual(source,target,5,'not implemented')


if __name__ == '__main__':
    unittest.main()