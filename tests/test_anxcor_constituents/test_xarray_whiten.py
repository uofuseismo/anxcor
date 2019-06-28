import unittest
from xarray_routines import XArrayWhiten, XArrayConverter
from .synthetic_trace_factory import  create_sinsoidal_trace
import scipy.fftpack as fft
import numpy as np
whiten = XArrayWhiten(smoothing_window_ratio=0.025, upper_frequency=25.0, lower_frequency=0.001, order=2)
convert = XArrayConverter()

class TestSpectralWhitening(unittest.TestCase):

    def test_whitened_success(self):

        trace    = convert([create_sinsoidal_trace(sampling_rate=100,period=0.5,    duration=3)],starttime=0,station=0)
        freq_2   = convert([create_sinsoidal_trace(sampling_rate=100, period=0.1,   duration=3)],starttime=0,station=0)
        trace     = trace +  freq_2
        trace.attrs = freq_2.attrs
        pow_period_original = self.get_power_at_freq(10.0,trace)
        trace   = whiten(trace,starttime=0,station=0)
        pow_period_final   = self.get_power_at_freq(10.0, trace)
        self.assertGreater(pow_period_original,pow_period_final,"whitening failed")

    def test_array_is_real(self):
        trace = convert([create_sinsoidal_trace(sampling_rate=100, period=0.5, duration=3)],starttime=0,station=0)
        freq_2 = convert([create_sinsoidal_trace(sampling_rate=100, period=0.1, duration=3)],starttime=0,station=0)
        trace = trace + freq_2
        trace.attrs = freq_2.attrs
        trace = whiten(trace,starttime=0,station=0)

        self.assertEqual(trace.data.dtype,np.float64,'improper data type')


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


if __name__ == '__main__':
    unittest.main()