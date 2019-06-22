import unittest
from .synthetic_trace_factory import create_random_trace, create_sinsoidal_trace_w_decay, create_triangle_trace
from xarray_routines import XArrayXCorrelate, XArrayConverter
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

converter = XArrayConverter()
class TestCorrelation(unittest.TestCase):
    def test_convert_is_accurate(self):
        max_tau_shift = None
        correlator = XArrayXCorrelate(max_tau_shift=max_tau_shift)

        e2 = create_sinsoidal_trace_w_decay(decay=0.8, station='k', network='v', channel='e', duration=20)
        n2 = create_random_trace(station='k', network='v', channel='n', duration=20)
        z2 = create_sinsoidal_trace_w_decay(decay=0.3, station='k', network='v', channel='z', duration=20)
        b2 = create_random_trace(station='k', network='v', channel='b', duration=20)
        syth_trace2 = converter([e2, n2, z2, b2])

        result_1 = syth_trace2.loc['e',:,:].data.ravel() -  e2.data
        self.assertEqual(0, np.sum(result_1))
        result_1 = syth_trace2.loc['n', :, :].data.ravel() - n2.data
        self.assertEqual(0, np.sum(result_1))
        result_1 = syth_trace2.loc['z', :, :].data.ravel() - z2.data
        self.assertEqual(0, np.sum(result_1))
        result_1 = syth_trace2.loc['b', :, :].data.ravel() - b2.data
        self.assertEqual(0, np.sum(result_1))