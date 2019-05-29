import unittest

from ancor import numpy_array_signal_functions
from ancor import numpy_vertical_matrix_signal_functions
from ancor import stream_signal_functions

class TestWhitenObspyStream(unittest.TestCase):

    def test_whitening_spectrum(self):
        """
         ensure the whitened spectrum is what we expect

         currently not implemented

        """
        target=0
        source= stream_signal_functions.whiten(None)
        self.assertAlmostEqual(source,target,5,"whitening spectrum test not implemented")

    def test_whitening_norm(self):
        """
         check to see the whitening norm difference is small

        """
        target = 0
        source = stream_signal_functions.whiten(None)
        self.assertAlmostEqual(source, target, 5, "whitening spectrum test not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        target = 0
        source = stream_signal_functions.whiten(None)
        self.assertAlmostEqual(source, target, 5, "whitening spectrum test not implemented")



class TestWhitenNumpyArray(unittest.TestCase):

    def test_whitening_spectrum(self):
        """
         ensure the whitened spectrum is what we expect

         currently not implemented

        """
        target=0
        source= numpy_array_signal_functions.whiten(None)
        self.assertAlmostEqual(source,target,5,"whitening spectrum test not implemented")

    def test_whitening_norm(self):
        """
         check to see the whitening norm difference is small

        """
        target = 0
        source = numpy_array_signal_functions.whiten(None)
        self.assertAlmostEqual(source, target, 5, "whitening spectrum test not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        target = 0
        source = numpy_array_signal_functions.whiten(None)
        self.assertAlmostEqual(source, target, 5, "whitening spectrum test not implemented")

class TestWhitenNumpyMatrix(unittest.TestCase):

    def test_whitening_spectrum(self):
        """
         ensure the whitened spectrum is what we expect

         currently not implemented

        """
        target = 0
        source = numpy_vertical_matrix_signal_functions.whiten(None)
        self.assertAlmostEqual(source,target,5,"whitening spectrum test not implemented")

    def test_whitening_norm(self):
        """
         check to see the whitening norm difference is small

        """
        target = 0
        source = numpy_vertical_matrix_signal_functions.whiten(None)
        self.assertAlmostEqual(source, target, 5, "whitening spectrum test not implemented")

    def test_for_edge_effects(self):
        """
        ensure no gibbs phenomena

        """
        target = 0
        source = numpy_vertical_matrix_signal_functions.whiten(None)
        self.assertAlmostEqual(source, target, 5, "whitening spectrum test not implemented")


if __name__ == '__main__':
    unittest.main()