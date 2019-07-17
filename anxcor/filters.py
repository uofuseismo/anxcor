from scipy.signal import butter, sosfiltfilt, sosfreqz, detrend, convolve, get_window
import scipy.fftpack as fftpack
import xarray as xr
from obspy.core import UTCDateTime
import numpy as np
import pandas as pd
STARTTIME_NS_PRECISION = 100.0
DELTA_MS_PRECISION     = 100.0/1
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
def lowpass_filter(data, upper_frequency=0.5, sample_rate=1, order=2, axis=-1):
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
    y = sosfiltfilt(sos, data,axis=axis)
    return y

def bandpass_in_time_domain(data, lower_frequency=0.01, upper_frequency=1.0, sample_rate=0.5, order=2,axis=-1, **kwargs):
    sos = _butter_bandpass(lower_frequency, upper_frequency, sample_rate, order=order)
    y = sosfiltfilt(sos, data,axis=axis)
    return y

def bandpass_in_frequency_domain(xarray,**kwargs):
    bandpass_response= _create_bandpass_frequency_multiplier(xarray,**kwargs)
    xarray*=bandpass_response
    return xarray

def taper(data, taper=0.1,axis=-1,window_type='hanning',one_taper=False,**kwargs):
    taper_length = int(taper*data.shape[-1])
    if (taper_length % 2) == 0:
        taper_length+=1

    center = (taper_length-1) // 2
    full_window    = get_window(window_type,taper_length,fftbins=False)
    full_window[0] = 0
    full_window[-1]= 0
    ones        = np.ones(data.shape[-1])

    ones[:center+1] *=full_window[:center+1]
    ones[-1-center:]*=full_window[center:]


    result = data * ones

    if one_taper:
        ones_window = 1 - ones
        result+= ones_window
    return result


def xarray_crosscorrelate(source_xarray, receiver_xarray,
                          taper=0.1, max_tau_shift=None, dummy_task=False,**kwargs):
    try:
        _check_xcorrelate_assumptions(source_xarray,receiver_xarray,taper,max_tau_shift,dummy_task,**kwargs)
    except Exception:
        _will_not_correlate_message(source_xarray,receiver_xarray)
        dummy_task=True
    src_channels   = list(source_xarray['channel'].values)
    rec_channels   = list(receiver_xarray['channel'].values)
    pair = ['src:' + list(source_xarray.coords['station_id'].values)[0] + \
            'rec:' + list(receiver_xarray.coords['station_id'].values)[0]]

    if not dummy_task:
        xcorr_np_mat = _cross_correlate_xarray_data(source_xarray,receiver_xarray,**kwargs)
    else:
        xcorr_np_mat = _dummy_correlate(source_xarray, receiver_xarray)

    xcorr_np_mat = _extract_center_of_ndarray(xcorr_np_mat, max_tau_shift, source_xarray)

    xcorr_np_mat = xcorr_np_mat.reshape((len(src_channels),len(rec_channels),1,xcorr_np_mat.shape[-1]))

    tau_array    = _get_new_time_array(source_xarray, max_tau_shift, xcorr_np_mat.shape[-1])


    xarray = xr.DataArray(xcorr_np_mat, coords=(('src_chan', src_channels),
                                                ('rec_chan', rec_channels),
                                                ('pair', pair),
                                                ('time',tau_array)))

    return xarray

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


def xarray_freq_2_time(array_fourier : xr.DataArray, array_original):
    time_data =np.real(np.fft.irfft(array_fourier.data,array_original.shape[-1], axis=-1)).astype(np.float64)
    array_new      = array_original.copy()
    array_new.data = time_data[:,:,:array_original.shape[-1]]
    return array_new


def xarray_freq_2_time_xcorr(array_fourier : np.ndarray, array_original):
    corr_length = array_original.data.shape[-1]*2-1
    time_data = np.real(np.fft.irfft(array_fourier.data, axis=-1)).astype(np.float64)[:,:,:corr_length].astype(np.float64)
    array_new      = array_original.copy()
    array_new.data = time_data[:,:,:array_original.shape[-1]]
    return array_new

################################################# helper methods #######################################################

def _butter_lowpass(cutoff, fs, order=5,**kwargs):
    sos = butter(order, cutoff, btype='lowpass',output='sos',analog=False,fs=fs*1.00000001)
    return sos

def _butter_bandpass(lowcut, highcut, fs, order=5):
    sos = butter(order, [lowcut, highcut], btype='bandpass',output='sos',analog=False,fs=fs*1.00000001)
    return sos


def _create_bandpass_frequency_multiplier(xarray,upper_frequency,
                                          lower_frequency,order=4,
                                          filter_power=3,delta=0.01,**kwargs):
    nyquist = 0.5 / delta
    if upper_frequency > nyquist:
        upper_frequency = nyquist
    sos = _butter_bandpass(lower_frequency, upper_frequency, 1 / delta,order=order)

    normalized_freqs = np.asarray(list(xarray.coords['frequency'].values)) * delta *2* np.pi

    w, resp = sosfreqz(sos, worN=normalized_freqs)
    resp    = np.power(resp,filter_power)
    return resp

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

def _will_not_correlate_message(source_xarray,receiver_xarray):
    start1 = UTCDateTime(source_xarray.attrs['starttime'])
    station_1 = list(source_xarray.coords['station_id'].values)[0]
    station_2 = list(receiver_xarray.coords['station_id'].values)[0]
    print('*' * 10)
    print('will not correlate windows!')
    print('src {} \nrec {}'.format(station_1,station_2))
    print('window: {}'.format(start1))
    print('*'*10)


def _extract_center_of_ndarray(corr_mat, tau_shift,xarray):
    if tau_shift is not None:
        center = (corr_mat.shape[-1] - 1 )//2
        shift_npts = int(tau_shift / xarray.attrs['delta'])
        corr_mat = corr_mat[:,:, center - shift_npts: center + shift_npts + 1]
    return corr_mat


def _get_new_time_array(xarray, max_tau_shift, max_samples):
    delta           = xarray.attrs['delta']
    zero            = np.datetime64(UTCDateTime(0.0).datetime)
    timedelta       = pd.Timedelta(delta, 's').to_timedelta64()
    if max_tau_shift is None:
        max_delta_shift = pd.Timedelta((max_samples-2) * delta/2 , 's').to_timedelta64()
    else:
        max_delta_shift = pd.Timedelta(max_tau_shift, 's').to_timedelta64()
    positive_tau    = np.arange(zero, zero + max_delta_shift + timedelta, timedelta)
    negative_tau    = np.arange(zero - max_delta_shift, zero,  timedelta)
    tau_array       = np.concatenate((negative_tau, positive_tau))
    return tau_array



def _cross_correlate_xarray_data(source_xarray, receiver_xarray,gpu_enable=False,torch=None,**kwargs):
    src_chan_size = source_xarray.data.shape[0]
    rec_chan_size = receiver_xarray.data.shape[0]
    time_size     = source_xarray.data.shape[-1]
    src_data      = source_xarray.data.reshape(  src_chan_size, time_size)
    receiver_data = receiver_xarray.data.reshape(rec_chan_size, time_size)


    corr_length     = time_size * 2 - 1
    target_length = fftpack.next_fast_len(corr_length)
    if not gpu_enable:

        fft_src = np.conj(np.fft.rfft(src_data, target_length, axis=-1))
        fft_rec = np.fft.rfft(receiver_data, target_length, axis=-1)

        result = _multiply_in_mat(fft_src,fft_rec)

        xcorr_mat = np.fft.fftshift(np.real(np.fft.irfft(result,corr_length, axis=-1)).astype(np.float64))

    else:

        zero_tensor_src = torch.zeros([target_length, src_chan_size], dtype=torch.float32, device=torch.device('cuda'))
        zero_tensor_rec = torch.zeros([target_length, rec_chan_size], dtype=torch.float32, device=torch.device('cuda'))
        zero_tensor_result= torch.zeros([target_length,src_chan_size,rec_chan_size],
                                        dtype=torch.float32, device=torch.device('cuda'))
        zero_tensor_src[:target_length,:] = np.swapaxes(src_data,1,0)
        zero_tensor_rec[:target_length,:] = np.swapaxes(receiver_data,1,0)

        fft_src  = torch.rfft(zero_tensor_src, 1)
        fft_rec  = torch.rfft(zero_tensor_src, 1)
        return None


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
    if minimum_size is None:
        target_length = fftpack.next_fast_len(array.shape[axis])
    else:
        target_length = fftpack.next_fast_len(minimum_size)
    fft               = np.fft.rfft(array, target_length, axis=axis)
    return fft


def _get_deltaf(time_window_length,delta):
    target_length = fftpack.next_fast_len(time_window_length)
    frequencies   = np.fft.rfftfreq(target_length, d=delta)
    return frequencies

def _dummy_correlate(source_array,
                     receiver_array):
    src_len = source_array.data.shape[0]
    rec_len = receiver_array.data.shape[0]
    time    = source_array.data.shape[-1]
    mat = np.real(_multiply_in_mat(source_array.data.reshape((src_len,time)),receiver_array.data.reshape(rec_len,time)))

    return mat
