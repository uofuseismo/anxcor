import unittest
from .synthetic_trace_factory import create_random_trace, create_sinsoidal_trace_w_decay
from xarray_routines import XArrayXCorrelate, XArrayConverter
import numpy as np
import xarray as xr

def shift_trace(data,time=1.0,delta=0.1):
    sampling_rate = int(1.0/delta)
    random_data   = np.random.uniform(-1,1,(data.shape[0],data.shape[1],int(sampling_rate*time)))
    new_data   = np.concatenate((random_data,data),axis=2)[:,:,:data.shape[2]]
    return new_data

converter = XArrayConverter()
class TestCorrelation(unittest.TestCase):


    def test_autocorrelation(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter([create_random_trace(station='1',network='v',duration=20)])

        correlation = correlator(syth_trace1,syth_trace1)

        zero_target_index = correlation.data.shape[2]//2+1

        zero_source_index = np.argmax(correlation.data[0,0,:])
        self.assertEqual(zero_source_index,zero_target_index,'autocorrelation failed')


    def test_correlation_length(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter([create_random_trace(station='1',network='v')])
        syth_trace2 = converter([create_random_trace(station='2',network='v')])
        try:
            correlation = correlator(syth_trace1,syth_trace2)
            self.assertTrue(False,'failed to catch exception')
        except Exception:
            self.assertTrue(True, 'failed to catch exception')


    def test_shift_trace(self):
        correlator  = XArrayXCorrelate(max_tau_shift=4.0)
        syth_trace2 = converter([create_sinsoidal_trace_w_decay(decay=0.6,station='h',network='v',channel='z',duration=20)])
        syth_trace1 = converter([create_sinsoidal_trace_w_decay(decay=0.4,station='h',network='w',channel='z',duration=20)])
        syth_trace2 = xr.apply_ufunc(shift_trace,syth_trace2,input_core_dims=[['time']],
                                                             output_core_dims=[['time']],
                                                             kwargs={'time': 3.0,'delta':syth_trace2.attrs['delta']},keep_attrs=True)

        correlation = correlator(syth_trace1,syth_trace2)
        zero_index = 4*40 + 3*40 + 2 # added two because of noise effects
        max_index  = np.argmax(correlation.data)
        self.assertEqual(max_index,zero_index,'failed to correctly identify tau shift')

if __name__ == '__main__':
    unittest.main()
