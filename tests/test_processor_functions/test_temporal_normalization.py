import unittest
from ancor import numpy_array_signal_functions
from ancor import numpy_vertical_matrix_signal_functions
from ancor import stream_signal_functions

class TestTemporalNormalizationObspyStream(unittest.TestCase):

    def test_temporal_normalization_spectrum(self):
        """
         ensure the temporal normalized spectrum is what we expect

         currently not implemented

        """
        target = 0
        source = stream_signal_functions.t_normalize(None)
        self.assertAlmostEqual(source,target,5,"normalization spectrum test not implemented")

    def test_temporal_normalization_norm(self):
        """
         check to see the temporal normalization norm is what is expected

        """
        target = 0
        source = stream_signal_functions.t_normalize(None)
        self.assertAlmostEqual(source,target, 5, "normalization norm not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        target = 0
        source = stream_signal_functions.t_normalize(None)
        self.assertAlmostEqual(source,target,5,'not implemented')


class TestTemporalNormalizationNumpyArray(unittest.TestCase):

    def test_temporal_normalization_spectrum(self):
        """
         ensure the temporal normalized spectrum is what we expect

         currently not implemented

        """
        target = 0
        source = numpy_array_signal_functions.t_normalize(None)
        self.assertAlmostEqual(source,target,5,"normalization spectrum test not implemented")

    def test_temporal_normalization_norm(self):
        """
         check to see the temporal normalization norm is what is expected

        """
        target = 0
        source = numpy_array_signal_functions.t_normalize(None)
        self.assertAlmostEqual(source,target, 5, "normalization norm not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        target = 0
        source = numpy_array_signal_functions.t_normalize(None)
        self.assertAlmostEqual(source,target,5,'not implemented')


class TestTemporalNormalizationNumpyMatrix(unittest.TestCase):

    def test_temporal_normalization_spectrum(self):
        """
         ensure the temporal normalized spectrum is what we expect

         currently not implemented

        """
        target = 0
        source = numpy_vertical_matrix_signal_functions.t_normalize(None)
        self.assertAlmostEqual(source, target, 5, "normalization spectrum test not implemented")

    def test_temporal_normalization_norm(self):
        """
         check to see the temporal normalization norm is what is expected

        """
        target = 0
        source = numpy_vertical_matrix_signal_functions.t_normalize(None)
        self.assertAlmostEqual(source, target, 5, "normalization norm not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        target = 0
        source = numpy_vertical_matrix_signal_functions.t_normalize(None)
        self.assertAlmostEqual(source, target, 5, 'not implemented')


if __name__ == '__main__':
    unittest.main()