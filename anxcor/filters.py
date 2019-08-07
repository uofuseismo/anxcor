from scipy.signal import butter, sosfiltfilt, sosfreqz, get_window, detrend,filtfilt
import scipy.fftpack as fftpack
import xarray as xr
from obspy.core import UTCDateTime
import numpy as np
import pandas as pd
from datetime import datetime
STARTTIME_NS_PRECISION = 100.0
DELTA_MS_PRECISION     = 100.0/1
ZERO                   = np.datetime64(UTCDateTime(0.0).datetime)
"""
filters contained in this module:
- lowpass
- bandpass
- bandpass in freq
- taper
- crosscorrelation

helper methods contained in this module:
- xarray time to frequency
- xarray frequency to time

"""
#################################################### filters ###########################################################
def lowpass_filter(data, upper_frequency=0.5, sample_rate=1, order=2, axis=-1,padtype='odd',**kwargs):
    """
     lowpass filter for n-dimensional arrays
    Parameters
    ----------
    data: np.ndarray
        n-d numpy array
    cutoff:
        upper cutoff frequency to filter
    fs:
        sampling rate
    order:
        order of the filter. defaults to 2
    axis:
        axis to apply the filter to

    Returns
    -------
        a filtered nd array of same dimensions as the input data
    """
    sos = _butter_lowpass(upper_frequency, sample_rate, order=order)
    y = sosfiltfilt(sos, data,axis=axis,padtype=padtype)
    return y

def bandpass_in_time_domain_sos(data, lower_frequency=0.01, upper_frequency=1.0, sample_rate=0.5,
                                order=2, axis=-1, padtype='odd', **kwargs):
    sos = _butter_bandpass(lower_frequency, upper_frequency, sample_rate, order=order)
    y = sosfiltfilt(sos, data,axis=axis,padtype=padtype)
    return y

def bandpass_in_time_domain_filtfilt(data, lower_frequency=0.01, upper_frequency=1.0, sample_rate=0.5,
                                order=2, axis=-1, padtype='odd', **kwargs):
    b,a = _butter_bandpass_filtfilt(lower_frequency, upper_frequency, sample_rate, order=order)
    y = filtfilt(b,a, data,axis=axis,padtype=padtype)
    return y


def taper_func(data, taper=0.1, axis=-1, window_type='hanning', taper_objective='zeros', constant=0.0, **kwargs):
    assert taper <= 1.0, 'taper is too big. Must be less than 1.0:{}'.format(taper)
    assert taper >= 0 , 'taper is too small. Must be bigger than 0.0:{}'.format(taper)
    taper_length = int(taper*data.shape[axis])
    if (taper_length % 2) == 0:
        taper_length-=1

    center = (taper_length-1) // 2
    full_window    = get_window(window_type,taper_length,fftbins=False)
    full_window[0] = 0
    full_window[-1]= 0
    ones        = np.ones(data.shape[-1])

    ones[:center+1] *=full_window[:center+1]
    ones[-1-center:]*=full_window[center:]


    result = data * ones

    if taper_objective=='constant':
        new_window = (1 - ones)*constant
        result+= new_window
    return result

def xarray_const_taper(array1, array2, window_type='hanning', axis=-1, taper=1.0, **kwargs):
    filtered_array = xr.apply_ufunc(taper_func, array1,
                                    input_core_dims=[['time']],
                                    output_core_dims=[['time']],
                                    kwargs={**dict(window_type=window_type,axis=axis,taper=taper),**kwargs}, keep_attrs=True)

    taper_length = int(taper*array1.data.shape[axis])
    if (taper_length % 2) == 0:
        taper_length -= 1
    center = (taper_length - 1) // 2
    full_window    = get_window(window_type,taper_length,fftbins=False)
    full_window[0] = 0
    full_window[-1]= 0
    ones        = np.ones(array1.data.shape)

    ones[:,:,:center+1]  *=full_window[:center+1]
    ones[:,:,-1-center:]*=full_window[center:]

    xarray_copy = array1.copy()
    xarray_copy.data = 1-ones
    xarray_copy *= array2
    filtered_array += xarray_copy
    return filtered_array


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


################################################# converters ###########################################################

def xarray_time_2_freq(xarray : xr.DataArray,minimum_size=None):
    freq_domain = _into_frequency_domain(xarray.data,
                                         axis=xarray.get_axis_num('time'),
                                         minimum_size=minimum_size)
    frequencies = _get_deltaf(xarray.data.shape[-1],xarray.attrs['delta'])
    channels    = list(xarray.coords['channel'].values)
    station_ids = list(xarray.coords['station_id'].values)
    frequencies = frequencies.tolist()
    xarray_freq = xr.DataArray(freq_domain, coords=[channels,station_ids,frequencies],
                                            dims  =['channel', 'station_id', 'frequency'])
    return xarray_freq


def xarray_freq_2_time(freq_array : xr.DataArray, array_original):
    time_data = np.fft.fftshift(np.real(np.fft.irfft(freq_array.data, array_original.data.shape[-1], axis=-1),axes=-1).astype(np.float64))
    array_new = array_original.copy()
    array_new.data = time_data
    return array_new


def xarray_freq_2_time_xcorr(array_fourier : np.ndarray, array_original):
    corr_length = array_original.data.shape[-1]*2-1
    time_data   = np.fft.fftshift(np.real(np.fft.irfft(array_fourier.data,corr_length, axis=-1),axes=-1).astype(np.float64))
    array_new      = array_original.copy()
    array_new.data = time_data
    return array_new

def xarray_triple_by_reflection(xarray: xr.DataArray):
    time_span = (pd.to_datetime(min(list(xarray.coords['time'].values))),
                 pd.to_datetime(max(list(xarray.coords['time'].values))))
    time_delta= time_span[1]-time_span[0]
    new_start = time_span[0]-time_delta
    new_end   = time_span[1]+time_delta
    negative_time_series = pd.date_range(new_start,    time_span[0],periods=xarray.data.shape[-1])
    positive_time_series = pd.date_range(time_span[1], new_end,  periods=xarray.data.shape[-1])

    negative_xarray = xarray.copy()
    negative_xarray.data = np.flip(xarray.data,axis=-1)
    negative_xarray.coords['time']=negative_time_series

    positive_xarray = xarray.copy()
    positive_xarray.data = np.flip(xarray.data,axis=-1)
    positive_xarray.coords['time']=positive_time_series

    negative_xarray=negative_xarray.combine_first(xarray)
    negative_xarray=negative_xarray.combine_first(positive_xarray)

    return negative_xarray

def xarray_center_third_time(larger_xarray : xr.DataArray, original_xarray : xr.DataArray):
    new_data  = original_slice_extract(larger_xarray.data, original_xarray.data)
    orig_copy = original_xarray.copy()
    orig_copy.data=new_data
    return orig_copy

################################################# pure numpy funcs #####################################################

def original_slice_extract(padded_data,original_data):
    pad_length    = (padded_data.shape[-1] - original_data.shape[-1])//2
    return padded_data.take(indices=range(pad_length,pad_length+original_data.shape[-1]),axis=-1)

################################################# helper methods #######################################################


def _butter_lowpass(cutoff, fs, order=5,**kwargs):
    sos = butter(order, cutoff, btype='lowpass',output='sos',analog=False,fs=fs*1.00000001)
    return sos

def _butter_bandpass(lowcut, highcut, fs, order=5):
    sos = butter(order, [lowcut, highcut], btype='bandpass',output='sos',analog=False,fs=fs*1.00000001)
    return sos

def _butter_bandpass_filtfilt(lowcut, highcut, fs, order=5):
    b,a = butter(order, [lowcut, highcut], btype='bandpass',analog=False,fs=fs*1.00000001)
    return b,a


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


def _cross_correlate_xarray_data(source_xarray, receiver_xarray,gpu_enable=False,torch=None,**kwargs):
    src_chan_size = source_xarray.data.shape[0]
    rec_chan_size = receiver_xarray.data.shape[0]
    time_size     = source_xarray.data.shape[-1]
    src_data      = source_xarray.data.reshape(  src_chan_size, time_size)
    receiver_data = receiver_xarray.data.reshape(rec_chan_size, time_size)


    corr_length     = time_size * 2 - 1
    target_length = fftpack.next_fast_len(corr_length)

    fft_src = np.conj(np.fft.rfft(src_data, target_length, axis=-1))
    fft_rec = np.fft.rfft(receiver_data, target_length, axis=-1)

    result = _multiply_in_mat(fft_src,fft_rec)

    xcorr_mat = np.fft.fftshift(np.real(np.fft.irfft(result,corr_length, axis=-1)).astype(np.float64),axes=-1)

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


def _into_frequency_domain(array,axis=-1,minimum_size=None):
    target_length = _get_minimum_fft_freq_size(array, axis, minimum_size)
    fft           = np.fft.rfft(array, target_length, axis=axis)
    return fft


def _get_minimum_fft_freq_size(array, axis, minimum_size):
    if minimum_size is None:
        target_length = fftpack.next_fast_len(array.shape[axis])
    else:
        target_length = fftpack.next_fast_len(minimum_size)
    return target_length


def _get_deltaf(time_window_length,delta):
    target_length = fftpack.next_fast_len(time_window_length)
    frequencies   = np.fft.rfftfreq(target_length, d=delta)
    return frequencies

