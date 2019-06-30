import unittest
from .synthetic_trace_factory import create_sinsoidal_trace
import numpy as np
from xarray_routines import XResample, XArrayConverter

converter =XArrayConverter()

class TestDownsample(unittest.TestCase):

    def test_phase_shift_not_introduced(self):
        target_rate     = 20
        process         = XResample(target_rate=target_rate)
        trace           = create_sinsoidal_trace(sampling_rate=100, period=0.5,    duration=0.5)
        starttime       = trace.stats.starttime.timestamp
        trace_initial   = converter([create_sinsoidal_trace(sampling_rate=100, period=0.5,    duration=0.5)],
                                    starttime=0.0,station=0)
        trace_processed = converter([create_sinsoidal_trace(sampling_rate=100, period=0.5, duration=0.5)],
                                    starttime=0.0,station=0)
        trace_processed = process(trace_processed,starttime=starttime,station=0)
        target        = np.argmax(trace_initial.data.ravel())   * trace_initial.attrs['delta']
        source        = np.argmax(trace_processed.data.ravel()) * trace_processed.attrs['delta']

        self.assertAlmostEqual(target,source,int(np.log10(1/target_rate)),"filter introduced phase shift")

    def test_nonetype_in_out(self):
        result = converter(None, starttime=0, station=0)
        self.assertEqual(result,None)