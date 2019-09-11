import unittest
from anxcor.xarray_routines import XArrayWhiten, XArrayConverter, XArrayResample, XArrayRemoveMeanTrend
# for travis build
from synthetic_trace_factory import  create_sinsoidal_trace_w_decay
from anxcor.xarray_routines import XArrayComponentNormalizer, XArray9ComponentNormalizer, XArrayXCorrelate
# for local build
#from synthetic_trace_factory import  create_sinsoidal_trace
from scipy.signal import correlate
import scipy.fftpack as fft
from obspy.clients.fdsn import Client
import pytest
from obspy.core import UTCDateTime, Stream
import numpy as np
import anxcor.filters as filts


def create_example_xarrays():
    converter = XArrayConverter()
    e1 = create_sinsoidal_trace_w_decay(decay=0.9, station='h', network='v', channel='e', duration=20)
    n1 = create_sinsoidal_trace_w_decay(decay=0.3, station='h', network='v', channel='n', duration=20)
    z1 = create_sinsoidal_trace_w_decay(decay=0.4, station='h', network='v', channel='z', duration=20)

    e2 = create_sinsoidal_trace_w_decay(decay=0.8, station='k', network='v', channel='e', duration=20)
    n2 = create_sinsoidal_trace_w_decay(decay=0.7, station='k', network='v', channel='n', duration=20)
    z2 = create_sinsoidal_trace_w_decay(decay=0.6, station='k', network='v', channel='z', duration=20)

    syth_trace1 = converter(e1.copy() + z1.copy() + n1.copy())
    syth_trace2 = converter(e2.copy() + z2.copy() + n2.copy())
    return syth_trace1, syth_trace2

class TestSpectralWhitening(unittest.TestCase):

    def test_normalize_single_trace(self):
        convert = XArrayConverter()
        trace    = convert(create_sinsoidal_trace_w_decay(sampling_rate=100,period=0.5,    duration=3))
        trace.data*=5
        component_norm = XArrayComponentNormalizer()
        norm_component = component_norm(trace)
        assert np.max(np.abs(norm_component.data.ravel()))<=1.0


    def test_normalize_crosscorrelation(self):
        correlator = XArrayXCorrelate(max_tau_shift=5.0)
        normer     = XArray9ComponentNormalizer()
        xarray_1, xarray_2 = create_example_xarrays()
        correlation = correlator(xarray_1,xarray_2)
        normed_component = normer(correlation)
        assert np.max(np.abs(normed_component.loc[dict(src_chan='z',rec_chan='z')].data.ravel()))<=1.0