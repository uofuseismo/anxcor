import unittest
from ancor.worker_processes import RunningAbsoluteMeanNorm, Downsample
from .synthetic_trace_factory import create_random_trace, create_sinsoidal_trace
from obspy import read
import numpy as np

source_file = 'test_data/test_temp_norm/test_teleseism.BHE.SAC'

def source_earthquake():
    earthquake_trace       = read(source_file, format='sac')[0]

    earthquake_trace.data /= max(earthquake_trace.data)

    return  earthquake_trace


def shift_trace(samples,trace):
    trace     = trace.copy()
    tr_len = len(trace.data)
    data       = trace.data
    new_data   = np.hstack((np.random.uniform(-1,1,samples),data))[:tr_len]
    new_t  = trace.copy()
    new_t.data = new_data
    return new_t

class TestBasicTemporalNormalization(unittest.TestCase):

    def test_reduce_earthquake_xcorr_func(self):
        # first, make a noise trace and shift it by tau * sampling rate\
        file = source_file
        duration = self.get_duration_of_eq(file)
        sampling_rate = 40.0

        tau_shift   = 35
        sample_shift= int(tau_shift * sampling_rate)
        noise_loc_1 = create_random_trace(sampling_rate=40,duration=duration)
        noise_loc_2 = shift_trace(sample_shift,noise_loc_1)

        # next, add an eq teleseism from file to both noise streams

        noise_loc_1_eq = noise_loc_1.copy()
        noise_loc_2_eq = noise_loc_2.copy()

        source_eq   = source_earthquake()
        noise_loc_1_eq.data += source_eq.data
        noise_loc_2_eq.data += source_eq.data

        # downsample both to 10hz sampling rate

        down = Downsample(target_rate=10)
        t_norm = RunningAbsoluteMeanNorm(time_window=10)

        noise_loc_1    = down([noise_loc_1])[0]
        noise_loc_2    = down([noise_loc_2])[0]

        noise_loc_1_eq = down([noise_loc_1_eq])[0]
        noise_loc_2_eq = down([noise_loc_2_eq])[0]

        noise_loc_2_eq_tnorm = t_norm([noise_loc_2_eq.copy()])[0]
        noise_loc_1_eq_tnorm = t_norm([noise_loc_1_eq.copy()])[0]

        # get the xcorr with and without temporal normalization

        x_corr         = self.correlation_func(noise_loc_1,noise_loc_2)
        x_corr_eq      = self.correlation_func(noise_loc_1_eq,noise_loc_2_eq)
        x_corr_eq_tnorm = self.correlation_func(noise_loc_1_eq_tnorm, noise_loc_2_eq_tnorm)

        zero_index     = len(x_corr_eq)//2


        self.assertGreater(x_corr_eq[zero_index],x_corr_eq_tnorm[zero_index])

    def get_duration_of_eq(self,file):
        earthquake_trace = read(file, format='sac')[0]
        duration = earthquake_trace.stats.endtime - earthquake_trace.stats.starttime + 1 / 40
        return duration

    def get_max_tau(self,sampling_rate,data):
        zero_time_ind = (len(data)-1)/2
        idx = np.argmax(data)
        return idx/sampling_rate

    def correlation_func(self, earthquake_trace, earthquake_source):

        corr_func = np.correlate(earthquake_source.data,earthquake_trace.data,mode='full')
        corr_func/=np.mean(np.abs(corr_func))
        return corr_func
