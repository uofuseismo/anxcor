import unittest
from .synthetic_trace_factory import create_triangle_trace,create_random_trace, create_sinsoidal_trace
import numpy as np
from worker_processes import SpectralWhiten, Taper
import matplotlib.pyplot as plt
import scipy.fftpack as fft

class TestStreamBandpassFiltering(unittest.TestCase):

    def test_whitened_success(self):
        """
         tests to ensure phase shifts are not introduced to bandpass filter

         currently not implemented

        """
        process         = SpectralWhiten(0.125,0.1)

        trace    = create_sinsoidal_trace(sampling_rate=100,period=0.5,    duration=3)
        freq_2   = create_sinsoidal_trace(sampling_rate=100, period=0.1,   duration=3)
        trace.data += freq_2.data

        pow_period_original = self.get_power_at_freq(3.0,trace)
        trace   = process([trace])[0]
        pow_period_final   = self.get_power_at_freq(3.0, trace)

        self.assertGreater(pow_period_final,pow_period_original,"whitening failed")

    def test_array_len_same(self):
        """
         tests to ensure phase shifts are not introduced to bandpass filter

         currently not implemented

        """
        process = SpectralWhiten(0.125, 0.1)

        trace = create_sinsoidal_trace(sampling_rate=100, period=0.5, duration=3)
        freq_2 = create_sinsoidal_trace(sampling_rate=100, period=0.1, duration=3)
        trace.data += freq_2.data

        original = len(trace.data)

        trace = process([trace])[0]

        final   = len(trace.data)

        self.assertEqual(original, final, "whitening failed")

    def test_zero_freq(self):
        """
         tests to ensure phase shifts are not introduced to bandpass filter

         currently not implemented

        """
        process = SpectralWhiten(0.125, 0.1)

        trace = create_sinsoidal_trace(sampling_rate=100, period=0.5, duration=3)
        freq_2 = create_sinsoidal_trace(sampling_rate=100, period=0.1, duration=3)
        trace.data += freq_2.data

        pow_period_original = self.get_power_at_freq(0.0, trace)
        trace = process([trace])[0]
        pow_period_final = self.get_power_at_freq(0.0, trace)

        self.assertAlmostEqual(pow_period_original,pow_period_final,10, "whitening failed")

    def get_power_at_freq(self,frequency, trace):
        data         = trace.data
        delta        = trace.stats.delta
        target_width = fft.next_fast_len(len(data))
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