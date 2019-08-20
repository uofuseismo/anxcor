import numpy as np
from obspy.core import UTCDateTime
import pandas as pd
import xarray as xr
import scipy.fftpack as fftpack
STARTTIME_NS_PRECISION = 100.0
DELTA_MS_PRECISION     = 100.0/1
ZERO                   = np.datetime64(UTCDateTime(0.0).datetime)

def xarray_crosscorrelate(source_xarray, receiver_xarray,
                          taper=0.1, max_tau_shift=None, dummy_task=False,**kwargs):
    try:
        _check_xcorrelate_assumptions(source_xarray,receiver_xarray,taper,max_tau_shift,dummy_task,**kwargs)
    except Exception as exp:
        print(exp)
        _will_not_correlate_message(source_xarray,receiver_xarray)
        return None
    src_channels   = list(source_xarray['channel'].values)
    rec_channels   = list(receiver_xarray['channel'].values)
    pair = ['src:' + list(source_xarray.coords['station_id'].values)[0] + \
            'rec:' + list(receiver_xarray.coords['station_id'].values)[0]]

    xcorr_np_mat = _cross_correlate_xarray_data(source_xarray,receiver_xarray,**kwargs)

    tau_array    = _get_new_time_array(source_xarray)

    xcorr_np_mat = xcorr_np_mat.reshape((len(src_channels),len(rec_channels),1,xcorr_np_mat.shape[-1]))

    xarray = xr.DataArray(xcorr_np_mat, coords=(('src_chan', src_channels),
                                                ('rec_chan', rec_channels),
                                                ('pair', pair),
                                                ('time',tau_array)))
    if max_tau_shift is not None:
        xarray = _slice_xarray_tau(xarray,max_tau_shift)
    return xarray

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

def _cross_correlate_xarray_data(source_xarray, receiver_xarray,**kwargs):
    src_chan_size = source_xarray.data.shape[0]
    rec_chan_size = receiver_xarray.data.shape[0]
    t_1           = source_xarray.data.shape[-1]
    t_2           = receiver_xarray.data.shape[-1]
    src_data      = source_xarray.data.reshape(  src_chan_size, t_1)
    receiver_data = receiver_xarray.data.reshape(rec_chan_size, t_2)

    corr_length     = t_1+t_2 - 1
    target_length   = fftpack.next_fast_len(corr_length)

    fft_src = np.fft.rfft(src_data,target_length)
    fft_rec = np.conj(np.fft.rfft(receiver_data,target_length))
    result  = _multiply_in_mat(fft_src,fft_rec)

    xcorr_mat = np.fft.fftshift(np.real(np.fft.irfft(result,corr_length, axis=-1).astype(np.float64)),axes=-1)

    return xcorr_mat / corr_length

def _multiply_in_mat(one,two,dtype=np.complex64):
    zero_mat = np.zeros((one.shape[0],
                         two.shape[0],
                         one.shape[-1]), dtype=dtype)

    for ri in range(0,two.shape[0]):
        zero_mat[:, ri, :]  = one

    for si in range(0,one.shape[0]):
        zero_mat[si, :, :] *= two

    return zero_mat


def _check_if_inputs_make_sense(source_array,  max_tau_shift):
    time = source_array.attrs['delta'] * (source_array.data.shape[2]-1)
    total_time = time
    if max_tau_shift is not None and total_time <= max_tau_shift:
        raise Exception('given tau shift is too large for input array')


def _check_xcorrelate_assumptions(source_xarray, receiver_xarray, taper, max_tau_shift, dummy_task, **kwargs):
    assert int(source_xarray.attrs['delta']   * DELTA_MS_PRECISION)/DELTA_MS_PRECISION == \
           int(receiver_xarray.attrs['delta'] * DELTA_MS_PRECISION)/DELTA_MS_PRECISION, \
                'array deltas are not equal!!'
    assert int(source_xarray.attrs['starttime']   * STARTTIME_NS_PRECISION)/STARTTIME_NS_PRECISION==\
           int(receiver_xarray.attrs['starttime'] * STARTTIME_NS_PRECISION)/STARTTIME_NS_PRECISION, \
                'differring starttimes!!'+\
                                             ' will not correlate'
    assert source_xarray.data.shape[-1]==receiver_xarray.data.shape[-1], \
        'xarray shapes are different! will not proceed'
    if max_tau_shift is not None:
        timedelta = 2*(source_xarray.coords['time'].values.max() - source_xarray.coords['time'].values.min())
        max_delta = pd.Timedelta(max_tau_shift*2*1e9,unit='N').to_timedelta64()
        assert timedelta >= max_delta, 'target tau shift delta is too long. aborting'

def _will_not_correlate_message(source_xarray,receiver_xarray):
    start1 = UTCDateTime(source_xarray.attrs['starttime'])
    station_1 = list(source_xarray.coords['station_id'].values)[0]
    station_2 = list(receiver_xarray.coords['station_id'].values)[0]
    print('*' * 10)
    print('will not correlate windows!')
    print('src {} \nrec {}'.format(station_1,station_2))
    print('window: {}'.format(start1))
    print('*'*10)
