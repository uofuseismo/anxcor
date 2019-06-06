import unittest
from .synthetic_trace_factory import create_random_trace, create_sinsoidal_trace, create_sinsoidal_trace_w_decay
from ancor.correlate_processes import BaseSingleThreadCorrelate
import numpy as np
import matplotlib.pyplot as plt
def shift_trace(samples,trace):
    trace     = trace.copy()
    tr_len = len(trace.data)
    data       = trace.data
    new_data   = np.hstack((np.random.uniform(-1,1,samples),data))[:tr_len]
    new_t  = trace.copy()
    new_t.data = new_data
    return new_t

class TestBandpassFiltering(unittest.TestCase):


    def test_autocorrelation(self):
        syth_trace1 = create_random_trace()
        syth_trace2 = create_random_trace()
        correlator = BaseSingleThreadCorrelate(tau_shift=1.0)
        def v_pair(x,y):
            return True
        correlator._valid_corr_pair = v_pair
        auto_correlation = correlator([syth_trace1,syth_trace2])[0]

        zero_target_index = len(auto_correlation.data)//2

        zero_source_index = np.argmax(auto_correlation.data)
        self.assertEqual(zero_source_index,zero_target_index,'autocorrelation failed')


    def test_reject_same_trace(self):
        syth_trace1 = create_random_trace(station='h',network='v',channel='z')
        syth_trace2 = create_random_trace(station='h',network='v',channel='z')
        correlator = BaseSingleThreadCorrelate(tau_shift=1.0)

        correlation = correlator([syth_trace1,syth_trace2])

        self.assertEqual(len(correlation),0,'failed to reject auto correlation')


    def test_shift_trace(self):
        syth_trace2 = create_sinsoidal_trace_w_decay(decay=0.6,station='h',network='v',channel='z')
        syth_trace1 = create_sinsoidal_trace_w_decay(decay=0.4,station='h',network='w',channel='z')
        syth_trace2 = shift_trace(10,syth_trace2)
        correlator = BaseSingleThreadCorrelate(tau_shift=1.0)

        correlation = correlator([syth_trace1,syth_trace2])[0]
        zero_index = len(correlation.data) // 2
        max_index  = np.argmax(correlation.data)
        self.assertEqual(max_index,zero_index-10,'failed to correctly identify tau shift')
