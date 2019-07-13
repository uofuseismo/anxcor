import unittest
from anxcor.xarray_routines import XArrayWhiten, XArrayConverter, XArrayResample, XArrayRemoveMeanTrend
# for travis build
#from tests.synthetic_trace_factory import  create_sinsoidal_trace
# for local build
from synthetic_trace_factory import  create_sinsoidal_trace
import scipy.fftpack as fft
from obspy.clients.fdsn import Client
import pytest
from obspy.core import UTCDateTime
import numpy as np
whiten = XArrayWhiten(smoothing_window_ratio=0.25, upper_frequency=25.0, lower_frequency=0.001,
                      order=2,rolling_metric='mean')
convert = XArrayConverter()

class TestSpectralWhitening(unittest.TestCase):

    def test_whitened_success(self):

        trace    = convert([create_sinsoidal_trace(sampling_rate=100,period=0.5,    duration=3)])
        freq_2   = convert([create_sinsoidal_trace(sampling_rate=100, period=0.1,   duration=3)])
        trace     = trace +  freq_2
        trace.attrs = freq_2.attrs
        pow_period_original = self.get_power_at_freq(10.0,trace)
        trace   = whiten(trace,starttime=0,station=0)
        pow_period_final   = self.get_power_at_freq(10.0, trace)
        assert pow_period_original > pow_period_final,"whitening failed"

    def test_array_is_real(self):
        tr_1 = create_sinsoidal_trace(sampling_rate=100, period=0.5, duration=3)
        tr_2 = create_sinsoidal_trace(sampling_rate=100, period=0.1, duration=3)
        trace = convert([tr_1,tr_1])
        freq_2 = convert([tr_2,tr_2])
        trace = trace + freq_2
        trace.attrs = freq_2.attrs
        trace = whiten(trace)

        assert trace.data.dtype == np.float64,'improper data type'


    def get_power_at_freq(self, frequency, xarray):
        data         = xarray.data.ravel()
        delta        = xarray.attrs['delta']
        target_width = fft.next_fast_len(data.shape[0])
        spectrum     = fft.fftshift(fft.fft(data, target_width))
        frequencies  = fft.fftshift(fft.fftfreq(target_width, d=delta))
        index_val    = self.find_nearest(frequencies,frequency)

        value_at_freq = spectrum[index_val]

        power_at_freq = np.abs(value_at_freq * np.conjugate(value_at_freq))

        return power_at_freq

    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def test_nonetype_in_out(self):
        result = whiten(None,starttime=0,station=0)
        assert True

    def test_jupyter_tutorial(self):
        client = Client("IRIS")
        t = UTCDateTime("2018-12-25 12:00:00").timestamp
        st = client.get_waveforms("UU", "SPU", "*", "H*", t, t + 10 * 60, attach_response=True)
        pre_filt = (0.003, 0.005, 40.0, 45.0)
        st.remove_response(output='DISP', pre_filt=pre_filt)
        converter = XArrayConverter()
        resampler = XArrayResample(target_rate=10.0)
        rmmean_trend = XArrayRemoveMeanTrend()

        xarray = converter(st)
        resampled_array = resampler(xarray)
        rmm_array = rmmean_trend(resampled_array)
        whitening_op = XArrayWhiten(taper=0.05, whiten_type='cross_component', upper_frequency=3.0,
                                    lower_frequency=0.01, smoothing_window_ratio=0.01)

        whitened_array = whitening_op(rmm_array)
        assert whitened_array.data[0,0,0]==pytest.approx(0,abs=1e-3)

if __name__ == '__main__':
    unittest.main()