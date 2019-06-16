import numpy as np
import xarray as xr
import anxcor.filter_ops as filt_ops
import pandas as pd
from obspy.core import UTCDateTime
import abstract_behaviors as ab
import os_utils as os_utils

class XArrayCombine(ab._XDaskTask):

    def __init__(self):
        super().__init__()

    def _single_thread_execute(self,first_data, second_data,**kwargs):
        ds = xr.merge([first_data, second_data])
        ds.attrs = first_data.attrs
        return ds


    def _io_result(self, result, *args, **kwargs):
        return result


class XArrayConverter(ab._XDaskTask):
    """
    dynamic args:
    - trace data structure
    static args:
    - metadata to persist. specifically how to access certain things

    returns:
    xarray structure with channel, station, and time dimensions, plus persistent metadata
    """

    def __init__(self):
        super().__init__()

    def _create_xarray_dataset(self, trace_list):
        return None

    def _attach_metadata(stream, xarr):
        return None

    def _get_station_id(self,trace):
        network = trace.stats.network
        station = trace.stats.station
        return network + '.' + station

    def _single_thread_execute(self, stream, **kwargs):
        channels  = []
        stations  = []
        delta     = None
        time_array= None
        data_type = None
        starttime = None
        for trace in stream:
            station_code = self._get_station_id(trace)
            if station_code not in stations:
                stations.append(station_code)

            channel =trace.stats.channel
            if channel not in channels:
                channels.append(channel)
            if time_array is None:
                starttime = np.datetime64(trace.stats.starttime.datetime)
                endtime   = np.datetime64(trace.stats.endtime.datetime)

                delta     = trace.stats.delta
                timedelta = pd.Timedelta(delta,'s').to_timedelta64()
                time_array= np.arange(starttime, endtime+timedelta, timedelta)
                starttime = trace.stats.starttime.timestamp
                data_type = trace.stats.data_type


        empty_array = np.zeros((len(channels),len(stations),len(time_array)))
        for trace in stream:
            chan       = channels.index(trace.stats.channel)
            station_id = stations.index(self._get_station_id(trace))

            empty_array[chan,station_id,:]=trace.data

        xarray      = xr.DataArray(empty_array,coords=[channels, stations, time_array],
                                   dims=['channel','station_id','time'])
        xarray.name              = data_type
        xarray.attrs['delta']    = delta
        xarray.attrs['starttime']= starttime

        return xarray

    def _get_process_signature(self):
        return 'xconvert'

    def _get_station_key(self, result):
        return list(result['station_id'].values)[0]


class XArrayBandpass(ab._XDaskTask):

    def __init__(self,upper_frequency=10.0,lower_frequency=0.01,order=2,taper=0.1):
        super().__init__()
        self._kwargs = {'upper_frequency':upper_frequency,
                        'lower_frequency':lower_frequency,
                        'order':order,
                        'taper':taper}

    def _single_thread_execute(self, xarray: xr.DataArray, **kwargs):
        sampling_rate = 1.0 / xarray.attrs['delta']
        attrs = xarray.attrs
        tapered_array  = xr.apply_ufunc(filt_ops.taper,xarray,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs=self._kwargs)
        filtered_array = xr.apply_ufunc(filt_ops.butter_bandpass_filter, tapered_array,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={**self._kwargs,**{
                                                'sample_rate': sampling_rate}})
        filtered_array.attrs = attrs

        return filtered_array

class XArrayTaper(ab._XDaskTask):

    def __init__(self,taper=0.1):
        super().__init__()
        self._kwargs = {'taper':taper}

    def _single_thread_execute(self, xarray: xr.DataArray, **kwargs):
        attrs = xarray.attrs
        filtered_array = xr.apply_ufunc(filt_ops.taper, xarray,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={**self._kwargs})
        filtered_array.attrs = attrs

        return filtered_array


class XResample(ab._XDaskTask):

    def __init__(self, target_rate=10.0):
        super().__init__()
        self.target = target_rate
        self.target_rule = str(int((1.0 / target_rate) * 1e9)) + 'N'

    def _single_thread_execute(self, xarray: xr.DataArray, **kwargs):
        sampling_rate = 1.0 / xarray.attrs['delta']
        nyquist = self.target / 2.0

        mean_array     = xarray.mean(dim=['time'])
        demeaned_array = xarray - mean_array
        detrend_array  = xr.apply_ufunc(filt_ops.detrend, demeaned_array,
                                        input_core_dims=[['time']],
                                        output_core_dims = [['time']],
                                        kwargs={'type':'linear'})
        filtered_array = xr.apply_ufunc(filt_ops.lowpass_filter,detrend_array,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={'upper_frequency':    nyquist,
                                                'sampling_rate': sampling_rate})
        resampled_array= filtered_array.resample(time=self.target_rule).interpolate('linear').bfill('time')
        resampled_array.attrs = xarray.attrs
        resampled_array.attrs['delta']= 1/self.target

        return resampled_array

    def _get_process_signature(self):
        return 'resample'

    def _get_station_key(self, result):
        return list(result['station_id'].values)[0]



class XArrayXCorrelate(ab._XDaskTask):


    def __init__(self,max_tau_shift=100.0):
        super().__init__()
        self._max_tau_shift = max_tau_shift

    def _single_thread_execute(self, source_xarray: xr.DataArray, receiver_xarray: xr.DataArray, **kwargs):
        correlation = filt_ops.xarray_crosscorrelate(source_xarray,
                                             receiver_xarray,
                                             max_tau_shift=self._max_tau_shift)
        return correlation

    def _get_process_signature(self):
        return 'crosscorrelate'

    def _get_station_key(self, result):
        return list(result['correlation_pair'].values)[0]

class XArrayRemoveMeanTrend(ab._XDaskTask):

    def __init__(self):
        super().__init__()


    def _single_thread_execute(self,xarray,*args,**kwargs):

        mean_array = xarray.mean(dim=['time'])
        demeaned_array = xarray - mean_array
        detrend_array = xr.apply_ufunc(filt_ops.detrend, demeaned_array,
                                       input_core_dims=[['time']],
                                       output_core_dims=[['time']],
                                       kwargs={'type': 'linear'},keep_attrs=True)

        return detrend_array


class XArrayTemporalNorm(ab._XDaskTask):


    def __init__(self,time_mean=10.0, lower_frequency=0.001,
                 upper_frequency=5.0, type='hv_preserve'):
        super().__init__()
        self._type  = type
        self._time_mean = time_mean
        self._lower = lower_frequency
        self._upper = upper_frequency

    def _single_thread_execute(self, xarray: xr.DataArray, **kwargs):
        sampling_rate   = 1.0/xarray.attrs['delta']
        rolling_samples = int(sampling_rate*self._time_mean)
        bandpassed_array = xr.apply_ufunc(filt_ops.butter_bandpass_filter,xarray,
                                        input_core_dims=[['time']],
                                        output_core_dims = [['time']],
                                        kwargs={'lower_frequency':self._lower,
                                                'upper_frequency':self._upper,
                                                'sample_rate':sampling_rate})

    def _single_thread_execute(self, xarray: xr.DataArray, **kwargs):
        sampling_rate   = 1.0/xarray.attrs['delta']
        rolling_samples = int(sampling_rate*self._time_mean)
        bandpassed_array = xr.apply_ufunc(filt_ops.butter_bandpass_filter,xarray,
                                        input_core_dims=[['time']],
                                        output_core_dims = [['time']],
                                        kwargs={'lower_frequency':self._lower,
                                                'upper_frequency':self._upper,
                                                'sample_rate':sampling_rate})
        if self._type == 'hv_preserve':
            bandpassed_array = abs(bandpassed_array).rolling(time=rolling_samples,center=True).mean()\
            .ffill('time').bfill('time').max(dim='channel')
        else:
            bandpassed_array = abs(bandpassed_array).rolling(time=rolling_samples, center=True).mean() \
                .ffill('time').bfill('time')

        attrs  = xarray.attrs
        name   = xarray.name
        xarray = xarray / bandpassed_array
        xarray.attrs = attrs
        xarray.name = name
        return xarray

    def _get_process_signature(self):
        return 'temp norm'

    def _get_station_key(self, result):
        return list(result['station_id'].values)[0]



class XArrayWhiten(ab._XDaskTask):

    def __init__(self, smoothing_interval=10.0, lower_frequency=0.001,
                 upper_frequency=5.0,taper=0.1,order=2):
        super().__init__()
        self._type = type
        self._kwargs = {'smoothing_interval': smoothing_interval,
                       'lower_frequency': lower_frequency,
                       'upper_frequency': upper_frequency,
                       'taper':taper,
                        'order':order}

    def _single_thread_execute(self, xarray: xr.DataArray, **kwargs):
        sampling_rate = 1.0 / xarray.attrs['delta']
        new_array = xr.apply_ufunc(filt_ops.xarray_whiten, xarray,
                                          input_core_dims=[['time']],
                                          output_core_dims=[['time']],
                                          kwargs={**self._kwargs ,
                                                  **{'delta':xarray.attrs['delta'],
                                                     'sample_rate': sampling_rate
                                                  }})

        mean_array     = new_array.mean(dim=['time'])
        demeaned_array = new_array - mean_array
        detrend_array  = xr.apply_ufunc(filt_ops.detrend, demeaned_array,
                                       input_core_dims=[['time']],
                                       output_core_dims=[['time']],
                                       kwargs={'type': 'linear'})

        attrs = xarray.attrs
        name = xarray.name
        detrend_array.attrs = attrs
        detrend_array.name=name
        return  detrend_array

    def _get_process_signature(self):
        return 'whiten'

    def _get_station_key(self, result):
        return list(result['station_id'].values)[0]



class XArrayStack(ab._XDaskTask):

    def __init__(self):
        super().__init__()


    def _single_thread_execute(self,first: xr.DataArray, second: xr.DataArray):
        result       = first + second
        result.attrs['delta']     = first.attrs['delta']
        result.attrs['starttime'] = first.attrs['starttime']
        return result

    def _get_process_signature(self):
        return 'crosscorrelate'

    def _get_station_key(self, result):
        return result.name

    def _get_window_key(self, result):
        return UTCDateTime(result.attrs['starttime']).isoformat() + \
               '|'+ str(len(result.attrs['windows']))

    def _io_result(self, result, *args, **kwargs):
        return result

class XArrayIO:


    def __init__(self):
        self._file = None


    def write_to_file(self,file):
        if not os_utils.folder_exists(file):
            os_utils.make_dir(file)
        self._file = file


    def __call__(self, result, extension,file, dask_client=None,**kwargs):
        if self._file is not None:
            path = self._file + os_utils.sep + extension
            if not os_utils.folder_exists(path):
                os_utils.make_dir(path)
            path = path + os_utils.sep + file + '.nc'
            result.to_netcdf(path)
