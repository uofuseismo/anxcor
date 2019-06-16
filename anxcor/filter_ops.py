from scipy.signal import butter, filtfilt, detrend
import scipy.fftpack as fftpack
import xarray as xr
import matplotlib.pyplot as plt
from obspy.core import UTCDateTime
import numpy as np
import pandas as pd


def _butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, upper_frequency=0.5, sampling_rate=1, order=2,axis=-1):
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
    b, a = _butter_lowpass(upper_frequency, sampling_rate, order=order)
    y = filtfilt(b, a, data,axis=axis)
    return y

def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.500000000001 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lower_frequency=0.01, upper_frequency=1.0, sample_rate=0.5, order=2,**kwargs):
    b, a = _butter_bandpass(lower_frequency, upper_frequency, sample_rate, order=order)
    y = filtfilt(b, a, data)
    return y

def _taper(np_arr,taper=0.1):
    time_length = np_arr.shape[0]
    taper_half_length = int(taper*time_length/2)

    full_window = np.hamming(int(taper*time_length))
    ones        = np.ones(time_length)
    ones[ :taper_half_length]*=full_window[:taper_half_length]
    ones[-taper_half_length:]*=full_window[-taper_half_length:]

    result = np.multiply(np_arr,ones)

    return result

def taper(data, taper=0.1,axis=-1,**kwargs):
    time_length = data.shape[axis]
    taper_half_length = int(taper*time_length/2)

    full_window = np.hamming(int(taper*time_length))
    ones        = np.ones(time_length)
    ones[ :taper_half_length]*=full_window[:taper_half_length]
    ones[-taper_half_length:]*=full_window[-taper_half_length:]

    result = data * ones

    return result

def xarray_whiten(data, taper=0.1, smoothing_interval=10.0, delta=0.04,axis=-1,**kwargs):

    len_time_axis = data.shape[axis]

    source_array  = np.apply_along_axis(_taper,axis,data,taper=taper)
    freq_domain   = _into_frequency_domain(source_array,axis=axis)
    deltaf        = _get_deltaf(source_array,delta,axis=axis)
    smoothing_pnts= int(-(-smoothing_interval // deltaf ))
    convolve_ones = np.ones((smoothing_pnts,)) / smoothing_pnts
    running_spec  = np.apply_along_axis(np.convolve, axis, np.abs(freq_domain), convolve_ones, mode='same')
    freq_domain  /= running_spec
    rsult         = _into_time_domain(freq_domain,axis=axis)[:,:,:len_time_axis]
    time_domain   = np.real(fftpack.ifftshift(rsult,axes=axis))
    time_domain   = np.apply_along_axis(_taper,axis,time_domain, taper=taper)
    time_domain   = np.apply_along_axis(butter_bandpass_filter,axis,time_domain, **kwargs)
    return time_domain


def _check_if_inputs_make_sense(source_array,  max_tau_shift):
    time = source_array.attrs['delta'] * (source_array.data.shape[2]-1)
    total_time = time
    if total_time <= max_tau_shift:
        raise Exception('given tau shift is too large for input array')


def xarray_crosscorrelate(source_xarray, receiver_xarray, taper=0.1, max_tau_shift=40.0):
    _check_if_inputs_make_sense(source_xarray, max_tau_shift)
    time_axis     = source_xarray.get_axis_num('time')

    source_chans  = list(source_xarray.coords['channel'].values)
    source_stat   = list(source_xarray.coords['station_id'].values)[0]
    attrs_src     = source_xarray.attrs
    name_src      = source_xarray.name
    source_array  = source_xarray.data

    receiver_chans= list(receiver_xarray['channel'].values)
    receiver_stat = list(receiver_xarray['station_id'].values)[0]
    attrs_rec     = receiver_xarray.attrs
    name_rec      = receiver_xarray.name
    receiver_array= receiver_xarray.data

    source_array  = np.apply_along_axis(_taper,time_axis,source_array,taper=taper)
    receiver_array= np.apply_along_axis(_taper,time_axis,receiver_array,taper=taper)

    sig, chan_coordinates = _cross_correlate(source_array,
                                             source_chans,
                                             receiver_array,
                                             receiver_chans,
                                             time_axis)

    tau_array        = _get_new_time_array(attrs_rec, max_tau_shift)
    chan_coordinates = _extract_new_channel_coordinates(chan_coordinates, sig)
    correlation      = _extract_center_of_ndarray(attrs_rec, max_tau_shift, sig)
    correlation_pair = ['src:' + source_stat + ' rec:' + receiver_stat]
    correlation_type = name_src + ':' + name_rec

    correlation_array=xr.DataArray(correlation,coords=[('channel',chan_coordinates),
                                                       ('correlation_pair',correlation_pair),
                                                       ('time',tau_array)])
    correlation_array.attrs = attrs_src
    correlation_array.name  = correlation_type
    return correlation_array


def _extract_new_channel_coordinates(chan_coordinates, sig):
    chan_coordinates = np.asarray(chan_coordinates)
    dim_1 = sig.shape[0]
    dim_2 = sig.shape[1]
    chan_coordinates = chan_coordinates.reshape((dim_1 * dim_2)).ravel().tolist()
    return chan_coordinates


def _extract_center_of_ndarray(attrs_rec, max_tau_shift, sig):
    dim_1 = sig.shape[0]
    dim_2 = sig.shape[1]
    dim_3 = sig.shape[2]
    sig   = sig.reshape((dim_1 * dim_2, 1, dim_3))
    # xcorr gives an array of size 2*n -1. if you autocorrelate one random array of length 100,
    # you'll find the max index is 99.
    center = (dim_3 +1)// 2 - 1
    shift_npts = int(max_tau_shift / attrs_rec['delta'])
    sig = sig[:, :, center - shift_npts: center + shift_npts + 1]
    return sig


def _get_new_time_array(attrs_rec, max_tau_shift):
    zero      = np.datetime64(UTCDateTime(0.0).datetime)
    timedelta = pd.Timedelta(attrs_rec['delta'], 's').to_timedelta64()
    max_delta_shift = pd.Timedelta(max_tau_shift, 's').to_timedelta64()
    positive_tau    = np.arange(zero, zero + max_delta_shift + timedelta, timedelta)
    negative_tau    = np.arange(zero - max_delta_shift, zero,  timedelta)
    tau_array       = np.concatenate((negative_tau, positive_tau))
    return tau_array


def _cross_correlate( source_array, source_chans, receiver_array,
                     receiver_chans, time_axis):
    fft_src = _into_frequency_domain(source_array,axis=time_axis)
    fft_rec = np.conj(_into_frequency_domain(receiver_array,axis=time_axis))
    freq_mat, chan_coordinates = _tensor_frequency_multiply(fft_src,
                                                            fft_rec,
                                                            receiver_chans,
                                                            source_chans,
                                                            time_axis)
    time_domain =_into_time_domain(freq_mat,axis=time_axis,expected_length=receiver_array.shape[2]*2-1)
    time_domain = fftpack.ifftshift(time_domain)
    return time_domain, chan_coordinates


def _tensor_frequency_multiply(fft_src, fft_rec, receiver_chans, source_chans, time_axis):
    freq_mat = np.zeros((len(source_chans), len(receiver_chans), fft_rec.shape[time_axis]),dtype=np.complex128)
    chan_coordinates = []
    for rec_w, rec_chan in enumerate(receiver_chans):
        new_row = []
        for src_w, src_chan in enumerate(source_chans):
            freq_mat[src_w, rec_w, :] = fft_src[src_w, 0, :] * fft_rec[rec_w, 0, :]
            new_channel = src_chan + '.' + rec_chan
            new_row.append(new_channel)
        chan_coordinates.append(new_row)
    return freq_mat, chan_coordinates


def _into_frequency_domain(array,axis=-1):
    target_length = fftpack.next_fast_len(array.shape[axis]*2)
    fft           = fftpack.fft(array, target_length, axis=axis)
    return fft

def _into_time_domain(array,axis=-1,expected_length=None):
    if expected_length is not None:
        return np.real(fftpack.ifft(array,expected_length, axis=axis))
    else:
        return np.real(fftpack.ifft(array, axis=axis))

def _get_deltaf(array,delta,axis=-1):
    frequencies = fftpack.fftfreq(array.shape[axis], d=delta)
    return frequencies[1]