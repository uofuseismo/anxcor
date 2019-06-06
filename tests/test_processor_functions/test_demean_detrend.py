import unittest
from worker_processes import RemoveMeanTrend
from .synthetic_trace_factory import linear_ramp_trend
import numpy as np

class TestDemean(unittest.TestCase):

    def test_demean(self):
        trace = linear_ramp_trend()
        process = RemoveMeanTrend()
        trace = process([trace])[0]
        mean = np.mean(trace)
        self.assertAlmostEqual(mean,0,5,"mean not removed")


    def test_detrend(self):
        trace = linear_ramp_trend()
        process = RemoveMeanTrend()
        trace = process([trace])[0]
        coeffs= np.polyfit(np.linspace(0,1,num=len(trace.data)),trace.data, 1)
        self.assertAlmostEqual(coeffs[0],0,5,"trend not removed")
