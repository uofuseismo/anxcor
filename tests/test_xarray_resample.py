import unittest
# travis execution
from tests.synthetic_trace_factory import create_sinsoidal_trace
#ide testing
#from synthetic_trace_factory import create_sinsoidal_trace
import numpy as np
from anxcor.xarray_routines import XArrayResample, XArrayConverter
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
converter =XArrayConverter()

class TestDownsample(unittest.TestCase):

    def test_phase_shift_not_introduced(self):
        target_rate     = 20
        process         = XArrayResample(target_rate=target_rate)
        trace_initial   = converter(create_sinsoidal_trace(sampling_rate=100, period=0.5,    duration=0.5))
        trace_processed = converter(create_sinsoidal_trace(sampling_rate=100, period=0.5, duration=0.5))
        trace_processed = process(trace_processed)
        target        = np.argmax(trace_initial.data.ravel())   * trace_initial.attrs['delta']
        source        = np.argmax(trace_processed.data.ravel()) * trace_processed.attrs['delta']

        assert round(abs(target-source), int(np.log10(1/target_rate))) == 0,"filter introduced phase shift"

    def test_nonetype_in_out(self):
        result = converter(None)
        assert result == None

    def test_not_null(self):
        client = Client("IRIS")
        t = UTCDateTime("2018-12-25 12:00:00").timestamp
        st = client.get_waveforms("UU", "SPU", "*", "H*", t, t + 6 * 60 * 60, attach_response=True)
        pre_filt = (0.003, 0.005, 40.0, 45.0)
        st.remove_response(output='DISP', pre_filt=pre_filt)
        xarray = converter(st)
        resampler = XArrayResample()
        resampled_array = resampler(xarray)
        assert not np.any(np.isnan(resampled_array.data))