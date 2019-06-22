import unittest
from .synthetic_trace_factory import  create_random_trace, create_sinsoidal_trace
import numpy as np
from xarray_routines import XArrayConverter, XArrayBandpass
from scipy import fftpack
from scipy import interpolate, signal

converter = XArrayConverter()
class TestBandpassFiltering(unittest.TestCase):

    def _get_frequency_of_trace(self,trace,sample_point=20.0):
        amplitudes = fftpack.fft(trace.data.ravel())
        amplitudes = np.abs(amplitudes * np.conjugate(amplitudes))
        amplitudes/= np.mean(amplitudes)
        freqs = fftpack.fftfreq(trace.data.shape[2],trace.attrs['delta'])
        spectrum = interpolate.interp1d(freqs, amplitudes)
        amplitude = spectrum(sample_point)
        return amplitude

    def test_filter_frequency_out_of_band(self):
        process = XArrayBandpass(lower_frequency=0.5,upper_frequency=20.0)
        trace  = converter([create_random_trace(sampling_rate=100)])
        trace  = process(trace)
        source = self._get_frequency_of_trace(trace,sample_point=40.0)
        target = 0
        self.assertAlmostEqual(source,target,1,"frequency not removed")

    def test_filter_frequency_in_band(self):
        process = XArrayBandpass(lower_frequency=0.5,upper_frequency=20.0)
        trace = converter([create_random_trace(sampling_rate=100)])
        trace = process(trace)
        source = self._get_frequency_of_trace(trace, sample_point=5)
        target = 0.1
        self.assertGreater(source,target,"bandpass filter removing desired frequency")

    def test_phase_shift_not_introduced(self):
        process         = XArrayBandpass(lower_frequency=0.5,upper_frequency=20.0)
        trace_initial   = converter([create_sinsoidal_trace(sampling_rate=100,period=0.25,    duration=10)])
        trace_processed = process(trace_initial.copy())
        source_1   = np.argmax( signal.correlate(trace_initial.data,trace_processed.data))
        correction =  source_1 - (trace_initial.data.shape[2]*2 -1)//2
        target = 0
        self.assertEqual(correction,target,"filter introduced phase shift")


if __name__ == '__main__':
    unittest.main()
