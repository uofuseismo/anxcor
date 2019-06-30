import unittest
from xarray_routines import XArrayConverter, XArrayRemoveMeanTrend
from .synthetic_trace_factory import linear_ramp_trend
import numpy as np
process = XArrayRemoveMeanTrend()
converter = XArrayConverter()
class TestDemean(unittest.TestCase):

    def test_demean(self):
        trace = converter([linear_ramp_trend()],starttime=0,station=0)
        trace = process(trace,starttime=0,station=0)
        mean  = np.sum(trace.data)
        self.assertAlmostEqual(mean,0,5,"mean not removed")


    def test_detrend(self):
        trace = converter([linear_ramp_trend()],starttime=0,station=0)
        trace = process(trace,starttime=0,station=0)
        coeffs= np.polyfit(np.linspace(0,1,num=len(trace.data.ravel())),trace.data.ravel(), 1)
        self.assertAlmostEqual(coeffs[0],0,5,"trend not removed")

    def test_nonetype_in_out(self):
        result = process(None,starttime=0,station=0)
        self.assertEqual(result,None)

if __name__ == '__main__':
    unittest.main()
