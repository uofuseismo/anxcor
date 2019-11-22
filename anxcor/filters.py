from scipy.signal import butter, sosfilt,sosfiltfilt, sosfreqz, get_window, detrend, iirfilter,zpk2sos
import xarray as xr
from obspy.core import UTCDateTime
import numpy as np
from scipy.fftpack import rfft, irfft, next_fast_len, rfftfreq
import pandas as pd
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
######################## filters ###########################################################
def lowpass_filter(data, freqmax=0.5, sample_rate=1, order=2,zerophase=True,axis=-1, **kwargs):
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
    fe = 0.5 * sample_rate
    norm_max_freq = freqmax / fe
    # raise for some bad scenarios
    if norm_max_freq - 1.0 > -1e-6:
        print("freqmax ({}) of bandpass is at or above Nyquist ({}). Ignoring.").format(freqmax, fe)
        return data

    if norm_max_freq > 1:
        print("Selected freqmax ({}) is above Nyquist. Ignoring".format(norm_max_freq))
        return data
    z, p, k = iirfilter(order, [norm_max_freq], btype='lowpass', ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        result = sosfilt(sos, data)
        return np.flip(sosfilt(sos, np.flip(result, axis=axis)), axis=axis)
    else:
        return sosfilt(sos, data)

def bandpass_in_time_domain_sos(data, freqmin=0.01, freqmax=1.0, sample_rate=0.5,
                                order=2, axis=-1,taper=None,zerophase=True, **kwargs):
    fe = 0.5 * sample_rate
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        print("freqmax ({}) of bandpass is at or above Nyquist ({}). Ignoring.").format(freqmax, fe)
        return data

    if low > 1:
        print("Selected freqmin ({}) is above Nyquist. Ignoring".format(freqmin))
        return data
    z, p, k = iirfilter(order, [low, high], btype='band',ftype='butter', output='zpk')#,fs=sample_rate)
    sos = zpk2sos(z, p, k)
    if zerophase:
        result = sosfilt(sos, data)
        return np.flip(sosfilt(sos, np.flip(result,axis=axis)),axis=axis)
    else:
        return sosfilt(sos, data,axis=axis)


def bandpass_in_time_domain_filtfilt(data, freqmin=0.01, freqmax=1.0, sample_rate=0.5,
                                     order=2, axis=-1, zerophase=True, **kwargs):
    fe = 0.5 * sample_rate
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        print("freqmax ({}) of bandpass is at or above Nyquist ({}). Ignoring.").format(freqmax, fe)
        return data

    if low > 1:
        print("Selected freqmin ({}) is above Nyquist. Ignoring".format(freqmin))
        return data
    z, p, k = iirfilter(order, [low, high], btype='band', ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)


def taper_func(data, taper=0.1, axis=-1, type='hanning', taper_objective='zeros', constant=0.0, **kwargs):
    assert taper <= 1.0, 'taper is too big. Must be less than 1.0:{}'.format(taper)
    assert taper >= 0 , 'taper is too small. Must be bigger than 0.0:{}'.format(taper)
    taper_length = int(taper*data.shape[axis]*2+1)
    full_window    = get_window(type,taper_length,fftbins=False)
    ones        = np.ones(data.shape[-1])
    center      = taper_length//2
    if taper_length%2!=0:
        center+=1
    ones[:center] *=full_window[:center]
    ones[-center:]*=full_window[-center:]

    result = data * ones

    if taper_objective=='constant':
        new_window = (1 - ones)*constant
        result+= new_window
    return result



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
    time_data  = irfft(freq_array.data, axis=-1)[:,:,:array_original.data.shape[-1]]
    array_new = array_original.copy()
    array_new.data = time_data
    return array_new


def xarray_freq_2_time_xcorr(array_fourier : np.ndarray, array_original):
    corr_length = array_original.data.shape[-1]*2-1
    time_data   = np.real(irfft(array_fourier.data, corr_length, axis=-1)).astype(np.float64)[:,:,:array_original.data.shape[-1]]
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


def _into_frequency_domain(array,axis=-1,minimum_size=None):
    target_length = _get_minimum_fft_freq_size(array, axis, minimum_size)
    fft           = rfft(array, n=target_length, axis=axis)
    return fft


def _get_minimum_fft_freq_size(array, axis, minimum_size):
    if minimum_size is None:
        target_length = next_fast_len(array.shape[axis])
    else:
        target_length = next_fast_len(minimum_size)
    return target_length


def _get_deltaf(time_window_length,delta):
    target_length = next_fast_len(time_window_length)
    frequencies   = rfftfreq(target_length, d=delta)
    return frequencies

