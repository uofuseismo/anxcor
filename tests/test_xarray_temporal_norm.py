import unittest
from anxcor.xarray_routines import XArrayWhiten, XArrayConverter, \
    XArrayTemporalNorm, XArrayResample, XArrayXCorrelate, XArrayRolling
# travis import
from tests.synthetic_trace_factory import create_random_trace, create_sinsoidal_trace
# relative import
#from synthetic_trace_factory import create_random_trace, create_sinsoidal_trace
from obspy.core import read
from scipy.signal import correlate
from anxcor.xarray_routines import XArrayRemoveMeanTrend
import pytest
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
import numpy as np
import xarray as xr
whiten    = XArrayWhiten(smoothing_window_ratio=0.025, upper_frequency=25.0, lower_frequency=0.001, order=2)
convert   = XArrayConverter()
xarraycorrelate = XArrayXCorrelate(max_tau_shift=40)
source_file = 'tests/test_data/test_teleseism/test_teleseism.BHE.SAC'


def source_earthquake():
    earthquake_trace       = read(source_file, format='sac')[0]
    earthquake_trace.stats.data_type='eq'
    earthquake_trace.data /= max(earthquake_trace.data)
    earthquake_trace.stats.starttime=0
    earthquake_trace.stats.channel='Z'
    earthquake_trace.stats.station = 'test'
    earthquake_trace.stats.network = ''


    return  convert([earthquake_trace],starttime=0,station=0)


def shift_trace(data,time=1.0,delta=0.1):
    sampling_rate = int(1.0/delta)
    random_data   = np.random.uniform(-1,1,(data.shape[0],data.shape[1],int(sampling_rate*time)))
    new_data      = np.concatenate((random_data,data),axis=2)[:,:,:data.shape[2]]
    return new_data

def combine_xarrays_along_dim(full_array, earthquake):
    full_array[:,:,:earthquake.shape[2]]+=earthquake
    return full_array

class TestBasicTemporalNormalization(unittest.TestCase):

    def test_increase_earthquake_shift_time_xcorr_func(self):
        # first, make a noise trace and shift it by tau * sampling rate\
        file = source_file
        duration = 400
        shift = 3
        noise_loc_1 = create_random_trace(sampling_rate=40, duration=duration)
        noise_loc_1[0].data+= create_random_trace(sampling_rate=40, duration=duration)[0].data
        noise_loc_1 = convert(noise_loc_1,starttime=0,station=0)
        noise_loc_2 = xr.apply_ufunc(shift_trace, noise_loc_1, input_core_dims=[['time']],
                                     output_core_dims=[['time']],
                                     kwargs={'time': shift, 'delta': noise_loc_1.attrs['delta']}, keep_attrs=True)

        # next, add an eq teleseism from file to both noise streams

        noise_loc_1_eq = noise_loc_1.copy()
        noise_loc_2_eq = noise_loc_2.copy()

        attrs = noise_loc_1.attrs

        source_eq   = source_earthquake()
        noise_loc_1_eq[:, :, :source_eq.data.shape[2]] += source_eq.data[:, :, :]
        noise_loc_2_eq[:, :, :source_eq.data.shape[2]] += source_eq.data[:, :, :]

        noise_loc_1.attrs = attrs
        noise_loc_2.attrs = attrs
        noise_loc_2_eq.attrs = attrs
        noise_loc_1_eq.attrs = attrs

        # downsample both to 10hz sampling rate

        down   = XArrayResample(10)
        t_norm =XArrayTemporalNorm(time_window=2.0)

        noise_loc_1_eq = down(noise_loc_1_eq)
        noise_loc_2_eq = down(noise_loc_2_eq)

        noise_loc_2_eq_tnorm = t_norm(noise_loc_2_eq.copy())
        noise_loc_1_eq_tnorm = t_norm(noise_loc_1_eq.copy())


        x_corr_eq      = self.max_corr_norm(noise_loc_1_eq, noise_loc_2_eq)
        x_corr_eq_tnorm= self.max_corr_norm(noise_loc_1_eq_tnorm, noise_loc_2_eq_tnorm)

        zero_index     = len(x_corr_eq)//2


        assert x_corr_eq[zero_index+int(40*shift)] < x_corr_eq_tnorm[zero_index+int(40*shift)]

    def test_variable_type(self):
        # first, make a noise trace and shift it by tau * sampling rate\
        file = source_file
        duration = 400
        shift = 30
        noise_loc_1 = create_random_trace(sampling_rate=40, duration=duration)
        noise_loc_1[0].data+= create_random_trace(sampling_rate=40, duration=duration)[0].data
        noise_loc_1 = convert(noise_loc_1)
        noise_loc_2 = xr.apply_ufunc(shift_trace, noise_loc_1, input_core_dims=[['time']],
                                     output_core_dims=[['time']],
                                     kwargs={'time': shift, 'delta': noise_loc_1.attrs['delta']}, keep_attrs=True)

        # next, add an eq teleseism from file to both noise streams

        noise_loc_1_eq = noise_loc_1.copy()

        down   = XArrayResample(10)
        t_norm =XArrayTemporalNorm(time_window=2.0)

        noise_loc_2_eq       = down(noise_loc_1_eq)

        noise_loc_2_eq_tnorm = t_norm(noise_loc_2_eq)


        assert noise_loc_2_eq_tnorm.dtype == np.float64

    def get_duration_of_eq(self,file):
        earthquake_trace = read(file, format='sac')[0]
        duration = earthquake_trace.stats.endtime - earthquake_trace.stats.starttime + 1 / 40
        return duration

    def get_max_tau(self,sampling_rate,data):
        zero_time_ind = (len(data)-1)/2
        idx = np.argmax(data)
        return idx/sampling_rate

    def max_corr_norm(self, one, two):
        corr_func= xarraycorrelate(one, two)
        corr_func=corr_func.data.ravel()
        corr_func/=np.max(abs(corr_func))
        return corr_func

    def test_nonetype_in_out(self):
        t_norm = XArrayTemporalNorm()
        result = t_norm(None)
        assert result is None

    def test_jupyter_example(self):
        client = Client("IRIS")
        t = UTCDateTime("2018-12-25 12:00:00").timestamp
        st = client.get_waveforms("UU", "SPU", "*", "H*", t, t + 6 * 60 * 60, attach_response=True)
        pre_filt = (0.003, 0.005, 40.0, 45.0)
        st.remove_response(output='DISP', pre_filt=pre_filt)
        converter = XArrayConverter()
        resampler = XArrayResample(target_rate=10.0)
        rmmean_trend = XArrayRemoveMeanTrend()
        temp_normalizer = XArrayTemporalNorm(t_norm_type='reduce_metric',
                                             lower_frequency=0.001, upper_frequency=0.1,
                                             time_window=2.0, rolling_procedure='mean',
                                             reduce_metric='max')
        xarray = converter(st)
        resampled_array = resampler(xarray)
        rmm_array = rmmean_trend(resampled_array)
        t_normed_array = temp_normalizer(rmm_array)
        assert True

    def test_phase_shift(self):
        stream   = create_sinsoidal_trace(sampling_rate=40.0, duration = 2000.0,
                                                               period=0.3)
        converter = XArrayConverter()
        xarray = converter(stream)
        t_norm_op = XArrayTemporalNorm(t_norm_type='reduce_metric',
                                             lower_frequency=0.001, upper_frequency=4,
                                             time_window=2.0, rolling_procedure='mean',
                                             reduce_metric='max')

        whitened_array = t_norm_op(xarray)
        a = whitened_array.data[0,0,:].squeeze()
        b = xarray.data[0,0,:].squeeze()
        xcorr = correlate(a, b)
        dt = np.arange(1 - a.shape[-1], a.shape[-1])

        recovered_time_shift = dt[xcorr.argmax()]

        assert recovered_time_shift==0

    def test_symmetric_output(self):
        converter = XArrayConverter()
        signal_length = 1000
        sampling_rate = 20.0
        center_index  = int(signal_length*sampling_rate) //2
        stream = create_sinsoidal_trace(sampling_rate=sampling_rate, duration=signal_length,
                                        period=2)
        stream1 = create_sinsoidal_trace(sampling_rate=sampling_rate, duration=signal_length,
                                         period=0.5)
        stream[0].data += stream1[0].data
        xarray = converter(stream)
        t_norm_op = XArrayTemporalNorm(t_norm_type='reduce_metric', taper=0.1,
                                       lower_frequency=0.02, upper_frequency=3.0,
                                       window=5.0, rolling_procedure='mean',
                                       reduce_metric='max')
        t_normed_array = t_norm_op(xarray)
        xarray_squeezed = t_normed_array[0, 0, :].data.squeeze()
        difference  = xarray_squeezed[:center_index]- xarray_squeezed[-center_index:]
        cumdiff     = abs(np.cumsum(difference)[-1]/(signal_length*sampling_rate))
        assert pytest.approx(0,abs=1e-2)==cumdiff


