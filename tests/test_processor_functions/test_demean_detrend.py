import unittest
from worker_processes import RemoveMeanTrend
from .synthetic_trace_factory import linear_ramp_trend
import numpy as np

class TestDemean(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_demean(self):
        '''
        Test for removing the mean for the whole file

        Subtract target file from processed file: should equal zero

        :return:



        '''
        trace = linear_ramp_trend()
        process = RemoveMeanTrend()
        trace = process([trace])[0]
        mean = np.mean(trace)
        self.assertAlmostEqual(mean,0,5,"mean not removed")


    def test_detrend(self):
        '''
        Test for removing the mean for the whole file

        Subtract target file from processed file: should equal zero

        :return:



        '''
        trace = linear_ramp_trend()
        process = RemoveMeanTrend()
        trace = process([trace])[0]
        coeffs= np.polyfit(np.linspace(0,1,num=len(trace.data)),trace.data, 1)
        self.assertAlmostEqual(coeffs[0],0,5,"trend not removed")

if __name__ == '__main__':
    unittest.main()