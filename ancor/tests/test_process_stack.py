import unittest
from ancor import numpy_array_signal_functions
from ancor import numpy_vertical_matrix_signal_functions
from ancor import stream_signal_functions

class TestSignalProcessingStackObspyStreams(unittest.TestCase):

    def test_signal_processing_stack(self):
        """
         ensure the signal processing steps are what we expect

         currently not implemented

        """
        target=0
        source = stream_signal_functions.process_all(None)
        self.assertAlmostEqual(source,target,5,"sig process steps not implemented")


class TestSignalProcessingStackNumpyArray(unittest.TestCase):

    def test_signal_processing_stack(self):
        """
         ensure the signal processing steps are what we expect

         currently not implemented

        """
        target=0
        source = numpy_array_signal_functions.process_all(None)
        self.assertAlmostEqual(source,target,5,"sig process steps not implemented")
        

class TestSignalProcessingStackNumpyMatrix(unittest.TestCase):

    def test_signal_processing_stack(self):
        """
         ensure the signal processing steps are what we expect

         currently not implemented

        """
        target=0
        source = numpy_vertical_matrix_signal_functions.process_all(None)
        self.assertAlmostEqual(source,target,5,"sig process steps not implemented")



class TestCrossCorrelationStackObspyStream(unittest.TestCase):

    def test_crosscorrelation_stack(self):
        """
         ensure the crosscorrelation steps including noise processing are what we expect

         currently not implemented

        """
        target = 0
        processed_data1 = stream_signal_functions.process_all(None)
        processed_data2 = stream_signal_functions.process_all(None)
        source          = stream_signal_functions.cross_correlate(processed_data1, processed_data2)
        self.assertAlmostEqual(source, target, 5, "crosscorr steps not implemented")


class TestCrossCorrelationStackNumpyArray(unittest.TestCase):

    def test_crosscorrelation_stack(self):
        """
         ensure the crosscorrelation steps including noise processing are what we expect

         currently not implemented

        """
        target = 0
        processed_data1 = numpy_array_signal_functions.process_all(None)
        processed_data2 = numpy_array_signal_functions.process_all(None)
        source          = numpy_array_signal_functions.cross_correlate(processed_data1, processed_data2)
        self.assertAlmostEqual(source, target, 5, "crosscorr steps not implemented")


class TestCrossCorrelationStackNumpyArray(unittest.TestCase):

    def test_crosscorrelation_stack(self):
        """
         ensure the crosscorrelation steps including noise processing are what we expect

         currently not implemented

        """
        target = 0
        processed_data1 = numpy_vertical_matrix_signal_functions.process_all(None)
        processed_data2 = numpy_vertical_matrix_signal_functions.process_all(None)
        source          = numpy_vertical_matrix_signal_functions.cross_correlate(processed_data1, processed_data2)
        self.assertAlmostEqual(source, target, 5, "crosscorr steps not implemented")



if __name__ == '__main__':
    unittest.main()