import unittest
from ancor import numpy_array_signal_functions
from ancor import numpy_vertical_matrix_signal_functions
from ancor import stream_signal_functions
precision = 1e-6


class TestNumpyArrayApproach(unittest.TestCase):

    def test_tau_shift_max(self):
        """
         Tests simple tau shifts from a synthetic dataset

         currently not implemented

        """
        target=0
        source = numpy_array_signal_functions.cross_correlate(None,None)
        self.assertAlmostEqual(source,target,5,"tau shift not implemented")

    def test_correlation_identity(self):
        """
         Tests (g*h)*(g*h) = (g*g)*(h*h) identity

         currently not implemented

        """
        target = 0
        source = numpy_array_signal_functions.cross_correlate(None, None)
        self.assertAlmostEqual(source,target, 5, "corr identity not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        target = 0
        source = numpy_array_signal_functions.cross_correlate(None, None)
        self.assertAlmostEqual(source,target,5,'not implemented')



class TestObspyStreamApproach(unittest.TestCase):

    def test_tau_shift_max(self):
        """
         Tests simple tau shifts from a synthetic dataset

         currently not implemented

        """
        target=0
        source = numpy_vertical_matrix_signal_functions.cross_correlate(None,None)
        self.assertAlmostEqual(source,target,5,"tau shift not implemented")

    def test_correlation_identity(self):
        """
         Tests (g*h)*(g*h) = (g*g)*(h*h) identity

         currently not implemented

        """
        target = 0
        source = numpy_vertical_matrix_signal_functions.cross_correlate(None,None)
        self.assertAlmostEqual(source,target, 5, "corr identity not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        target = 0
        source = numpy_vertical_matrix_signal_functions.cross_correlate(None,None)
        self.assertAlmostEqual(source,target,5,'not implemented')


class TestNumpyMatrixApproach(unittest.TestCase):

    def test_tau_shift_max(self):
        """
         Tests simple tau shifts from a synthetic dataset

         currently not implemented

        """
        target=0
        source =  stream_signal_functions.cross_correlate(None,None)
        self.assertAlmostEqual(source,target,5,"tau shift not implemented")

    def test_correlation_identity(self):
        """
         Tests (g*h)*(g*h) = (g*g)*(h*h) identity

         currently not implemented

        """
        target = 0
        source =  stream_signal_functions.cross_correlate(None,None)
        self.assertAlmostEqual(source,target, 5, "corr identity not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        target = 0
        source = stream_signal_functions.cross_correlate(None,None)
        self.assertAlmostEqual(source,target,5,'not implemented')

if __name__ == '__main__':
    unittest.main()