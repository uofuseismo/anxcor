import unittest
from .synthetic_trace_factory import create_triangle_trace,create_random_trace, create_sinsoidal_trace
import numpy as np
from worker_processes import Downsample


class TestDownsample(unittest.TestCase):

    def test_phase_shift_not_introduced(self):
        target_rate     = 20
        process         = Downsample(target_rate=target_rate)
        trace_initial   = create_sinsoidal_trace(sampling_rate=100,period=0.5,    duration=0.5)
        trace_processed = create_sinsoidal_trace(sampling_rate=100, period=0.5, duration=0.5)
        trace_processed = process([trace_processed])[0]
        target        = np.argmax(trace_initial) * trace_initial.stats.delta
        source        = np.argmax(trace_processed) * trace_processed.stats.delta

        self.assertAlmostEqual(target,source,int(np.log10(1/target_rate)),"filter introduced phase shift")
