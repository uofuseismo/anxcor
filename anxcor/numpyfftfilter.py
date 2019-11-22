import numpy as np
from obspy.core import UTCDateTime
import pandas as pd
import xarray as xr
from scipy.signal import fftconvolve
from scipy.fftpack import next_fast_len
STARTTIME_NS_PRECISION = 100.0
DELTA_MS_PRECISION     = 100.0/1
ZERO                   = np.datetime64(UTCDateTime(0.0).datetime)

def xarray_crosscorrelate(source_xarray, receiver_xarray, max_tau_shift=None,**kwargs):
    src_channels   = list(source_xarray['channel'].values)
    rec_channels   = list(receiver_xarray['channel'].values)
    pair = [list(source_xarray.coords['station_id'].values)[0],list(receiver_xarray.coords['station_id'].values)[0]]
    xcorr_np_mat = _cross_correlate_xarray_data_scipy_fftconvolve(source_xarray, receiver_xarray)

    tau_array    = _get_new_time_array(source_xarray)

    xcorr_np_mat = xcorr_np_mat.reshape((1,1,len(src_channels),len(rec_channels),xcorr_np_mat.shape[-1]))
    xcorr_np_mat, tau_array=_correct_for_time_misalignment_if_necessary(tau_array,xcorr_np_mat)
    xarray = xr.DataArray(xcorr_np_mat, coords=(('src',[pair[0]]),
                                                ('rec',[pair[1]]),
                                                ('src_chan', src_channels),
                                                ('rec_chan', rec_channels),
                                                ('time',tau_array)))


    if max_tau_shift is not None:
        xarray = _slice_xarray_tau(xarray,max_tau_shift)
    return xarray


def _correct_for_time_misalignment_if_necessary(time_array,mat_array):
    if mat_array.shape[-1] != len(time_array):
        cut_end = True
        while mat_array.shape[-1]!=len(time_array):
            if mat_array.shape[-1]>len(time_array):
                if cut_end:
                    mat_array=mat_array[:,:,:,:,:-1]
                    cut_end=False
                else:
                    mat_array = mat_array[:, :, :, :, 1:]
                    cut_end=True
            else:
                if cut_end:
                    time_array=time_array[:-1]
                    cut_end=False
                else:
                    time_array=time_array[1:]
                    cut_end=True

    return mat_array,time_array

def _get_new_time_array(source_xarray):
    delta       = source_xarray.attrs['delta']
    timedelta   = pd.Timedelta(delta, 's').to_timedelta64()
    data_length = source_xarray.shape[-1]
    min_time    = ZERO - data_length * timedelta + timedelta
    max_time    = ZERO + data_length * timedelta
    tau_array   = np.arange(min_time,max_time, timedelta)
    return tau_array

def _slice_xarray_tau(xarray,max_tau_shift):
    delta=pd.Timedelta(max_tau_shift * 1e9, unit='N').to_timedelta64()
    return xarray.sel(time=slice(ZERO - delta, ZERO + delta))


def _cross_correlate_xarray_data_scipy_fftconvolve(source_xarray, receiver_xarray):
    src_chan_size = source_xarray.data.shape[0]
    rec_chan_size = receiver_xarray.data.shape[0]
    t_1           = source_xarray.data.shape[-1]
    t_2           = receiver_xarray.data.shape[-1]
    src_data      = source_xarray.data.reshape(src_chan_size,t_1)
    receiver_data = receiver_xarray.data.reshape(rec_chan_size,t_2)

    zero_mat = np.zeros((src_chan_size,
                         rec_chan_size,
                         t_1+t_2-1))

    for src_chan in range(0,src_chan_size):
        for rec_chan in range(0,rec_chan_size):
            result = fftconvolve( src_data[src_chan],receiver_data[rec_chan][::-1],mode='full')
            zero_mat[src_chan,rec_chan,:]=result

    return zero_mat




def _check_if_inputs_make_sense(source_array,  max_tau_shift):
    time = source_array.attrs['delta'] * (source_array.data.shape[2]-1)
    total_time = time
    if max_tau_shift is not None and total_time <= max_tau_shift:
        raise Exception('given tau shift is too large for input array')


def _will_not_correlate_message(source_xarray,receiver_xarray):
    start1 = UTCDateTime(source_xarray.attrs['starttime'])
    station_1 = list(source_xarray.coords['station_id'].values)[0]
    station_2 = list(receiver_xarray.coords['station_id'].values)[0]
    print('*' * 10)
    print('will not correlate windows!')
    print('src {} \nrec {}'.format(station_1,station_2))
    print('window: {}'.format(start1))
    print('*'*10)
