import unittest
from .synthetic_trace_factory import create_triangle_trace,create_random_trace, create_sinsoidal_trace
import numpy as np
from process import BandPass
from scipy import fftpack
from scipy import interpolate, signal
import pandas as pd


class TestStreamBandpassFiltering(unittest.TestCase):

    def _get_frequency_of_trace(self,trace,sample_point=20.0):
        sample_rate = trace.stats['sampling_rate']
        amplitudes = fftpack.fft(trace.data)
        amplitudes = np.abs(amplitudes * np.conjugate(amplitudes))
        padded_data = np.pad(amplitudes, pad_width=15, mode='constant')
        amplitudes = pd.Series(padded_data).rolling(window=30).mean().iloc[30:].values
        amplitudes/= np.max(amplitudes)
        freqs = fftpack.fftfreq(len(trace.data)) * trace.stats['sampling_rate']
        spectrum = interpolate.interp1d(freqs, amplitudes)
        amplitude = spectrum(sample_point)
        return amplitude

    def test_filter_frequency_out_of_band(self):
        """
         tests to make sure that frequencies added outside the bandpassed range are not retained

         currently not implemented

        """
        process = BandPass(freqmin=0.5, freqmax=20)
        trace  = create_random_trace(sampling_rate=100)
        trace  = process([trace])[0]
        source = self._get_frequency_of_trace(trace,sample_point=40.0)
        target = 0
        self.assertAlmostEqual(source,target,3,"frequency not removed")

    def test_filter_frequency_in_band(self):
        """
         tests to ensure frequencies are retained in band

         currently not implemented

        """
        process = BandPass(freqmin=0.5, freqmax=20)
        trace = create_random_trace(sampling_rate=100)
        trace = process([trace])[0]
        source = self._get_frequency_of_trace(trace, sample_point=5)
        target = 0.1
        self.assertGreater(source,target,"bandpass filter removing desired frequency")

    def test_phase_shift_not_introduced(self):
        """
         tests to ensure phase shifts are not introduced to bandpass filter

         currently not implemented

        """
        process         = BandPass(freqmin=0.25, freqmax=20,zerophase=True)
        trace_initial   = create_sinsoidal_trace(sampling_rate=100,period=0.25,    duration=0.25)
        trace_processed = create_sinsoidal_trace(sampling_rate=100, period=0.25, duration=0.25)
        trace_processed = process([trace_processed])[0]
        source_1   = np.argmax( signal.correlate(trace_initial.data,trace_processed.data))
        correction =  source_1 - (len(trace_initial.data) -1)
        target = 0
        self.assertEqual(correction,target,"filter introduced phase shift")


if __name__ == '__main__':
    unittest.main()