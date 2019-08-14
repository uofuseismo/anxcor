import unittest
from anxcor.xarray_routines import XArrayConverter, XArrayRemoveMeanTrend
from tests.synthetic_trace_factory import linear_ramp_trend
import numpy as np
process = XArrayRemoveMeanTrend()
converter = XArrayConverter()
class TestDemean(unittest.TestCase):

    def test_demean(self):
        trace = converter(linear_ramp_trend())
        trace = process(trace)
        mean  = np.sum(trace.data)
        assert round(abs(mean-0), 5) == 0,"mean not removed"


    def test_detrend(self):
        trace = converter(linear_ramp_trend())
        trace = process(trace)
        coeffs= np.polyfit(np.linspace(0,1,num=len(trace.data.ravel())),trace.data.ravel(), 1)
        assert round(abs(coeffs[0]-0), 5) == 0,"trend not removed"

    def test_nonetype_in_out(self):
        result = process(None,starttime=0,station=0)
        assert result == None

if __name__ == '__main__':
    unittest.main()
