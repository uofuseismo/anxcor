import numpy as np
import xarray as xr
import anxcor.filter_ops as filt_ops
import pandas as pd
from obspy.core import UTCDateTime
import abstract_behaviors as ab
import os_utils as os_utils

OPERATIONS_SEPARATION_CHARACTER = '\n'
SECONDS_2_NANOSECONDS = 1e9


class XArrayConverter(ab.XArrayProcessor):
    """
    dynamic args:
    - trace data structure
    static args:
    - metadata to persist. specifically how to access certain things

    returns:
    xarray structure with channel, station, and time dimensions, plus persistent metadata
    """

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.writer = ab._XArrayWrite(None)
        self.reader = ab._XArrayRead(None)

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
        latitude  = None
        longitude = None
        elevation = None
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
                stats_keys = trace.stats.keys()
                if 'latitude' in stats_keys:
                    latitude = trace.stats.latitude
                if 'longitude' in stats_keys:
                    longitude = trace.stats.longitude
                if 'elevation' in stats_keys:
                    elevation = trace.stats.elevation


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
        xarray.attrs['operations']='xconvert'
        if latitude is not None and longitude is not None:
            if elevation is not None:
                xarray.attrs['location'] = (latitude, longitude, elevation)
            else:
                xarray.attrs['location'] = (latitude, longitude)
        return xarray

    def _metadata_to_persist(self, *param, **kwargs):
        return None

    def _get_process(self):
        return 'xconvert'

    def _get_name(self,*args):
        return None


class XArrayBandpass(ab.XArrayProcessor):

    def __init__(self,upper_frequency=10.0,lower_frequency=0.01,order=2,taper=0.1,**kwargs):
        super().__init__(**kwargs)
        self._kwargs = {'upper_frequency':upper_frequency,
                        'lower_frequency':lower_frequency,
                        'order':order,
                        'taper':taper}

    def _single_thread_execute(self, xarray: xr.DataArray, **kwargs):
        sampling_rate = 1.0 / xarray.attrs['delta']
        tapered_array  = xr.apply_ufunc(filt_ops.taper,xarray,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs=self._kwargs)
        filtered_array = xr.apply_ufunc(filt_ops.butter_bandpass_filter, tapered_array,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={**self._kwargs,**{
                                                'sample_rate': sampling_rate}})

        return filtered_array

    def _add_operation_string(self):
        return 'bandpass@{}<{}'.format(self._kwargs['lower_frequency'],
                                       self._kwargs['upper_frequency'])

    def _get_process(self):
        return 'bandpass'


class XArrayTaper(ab.XArrayProcessor):

    def __init__(self,taper=0.1,**kwargs):
        super().__init__(**kwargs)
        self._kwargs = {'taper':taper}

    def _single_thread_execute(self, xarray: xr.DataArray, **kwargs):
        filtered_array = xr.apply_ufunc(filt_ops.taper, xarray,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={**self._kwargs})

        return filtered_array

    def _add_operation_string(self):
        return 'taper@{}%'.format(self._kwargs['taper']*100)

    def _get_process(self):
        return 'taper'


class XResample(ab.XArrayProcessor):

    def __init__(self, target_rate=10.0,**kwargs):
        super().__init__(**kwargs)
        self.target = target_rate
        self.target_rule = str(int((1.0 / target_rate) * SECONDS_2_NANOSECONDS)) + 'N'

    def _single_thread_execute(self, xarray: xr.DataArray, **kwargs):
        sampling_rate = 1.0 / xarray.attrs['delta']
        nyquist       = self.target / 2.0

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

        return resampled_array

    def _add_metadata_key(self):
        return ('delta',1.0/self.target)

    def _get_process(self):
        return 'resample'

    def _add_operation_string(self):
        return 'resampled@{}Hz'.format(self.target)



class XArrayXCorrelate(ab.XArrayProcessor):


    def __init__(self,max_tau_shift=100.0,**kwargs):
        super().__init__(**kwargs)
        self._max_tau_shift = max_tau_shift

    def _single_thread_execute(self, source_xarray: xr.DataArray, receiver_xarray: xr.DataArray, **kwargs):
        correlation = filt_ops.xarray_crosscorrelate(source_xarray,
                                             receiver_xarray,
                                             max_tau_shift=self._max_tau_shift,**self._kwargs)
        return correlation

    def _get_process(self):
        return 'crosscorrelate'


    def _metadata_to_persist(self, xarray_1,xarray_2, **kwargs):
        attrs = {'delta'    : xarray_1.attrs['delta'],
                 'starttime': xarray_1.attrs['starttime'],
                 'stacks'   : 1,
                 'endtime'  : xarray_1.attrs['starttime'] + xarray_1.attrs['delta'] * xarray_1.data.shape[-1],
                 'operations': xarray_1.attrs['operations'] + OPERATIONS_SEPARATION_CHARACTER + \
                               'correlated@{}<t<{}'.format(self._max_tau_shift,self._max_tau_shift)}
        if 'location' in xarray_1.attrs.keys() and 'location' in xarray_2.attrs.keys():
            attrs['location'] = {'src':xarray_1.attrs['location'],
                                 'rec':xarray_2.attrs['location']}
        return attrs

    def _add_operation_string(self):
        return None

    def _get_name(self,one,two):
        return 'src:{} rec:{}'.format(one.name, two.name)

class XArrayRemoveMeanTrend(ab.XArrayProcessor):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)


    def _single_thread_execute(self,xarray,*args,**kwargs):

        mean_array = xarray.mean(dim=['time'])
        demeaned_array = xarray - mean_array
        detrend_array = xr.apply_ufunc(filt_ops.detrend, demeaned_array,
                                       input_core_dims=[['time']],
                                       output_core_dims=[['time']],
                                       kwargs={'type': 'linear'},keep_attrs=True)

        return detrend_array

    def _add_operation_string(self):
        return 'remove_Mean&Trend'

    def _get_process(self):
        return 'remove_mean_trend'


class XArrayTemporalNorm(ab.XArrayProcessor):


    def __init__(self,time_mean=10.0, lower_frequency=0.001,
                 upper_frequency=5.0, type='hv_preserve',**kwargs):
        super().__init__(**kwargs)
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
        if self._type == 'hv_preserve':
            bandpassed_array = abs(bandpassed_array).rolling(time=rolling_samples,center=True).mean()\
            .ffill('time').bfill('time').max(dim='channel')
        else:
            bandpassed_array = abs(bandpassed_array).rolling(time=rolling_samples, center=True).mean() \
                .ffill('time').bfill('time')

        xarray = xarray / bandpassed_array
        return xarray

    def _get_process(self):
        return 'temp_norm'


    def _add_operation_string(self):
        return 'temporal_norm@type({}),time_mean({}),f_norm_basis({})<f<({})'.format( \
            self._type,self._time_mean,self._lower,self._upper)



class XArrayWhiten(ab.XArrayProcessor):

    def __init__(self, smoothing_window_ratio=10.0, lower_frequency=0.001,
                 upper_frequency=5.0, order=2, **kwargs):
        super().__init__(**kwargs)
        self._kwargs = {'smoothing_window_ratio': smoothing_window_ratio,
                       'lower_frequency': lower_frequency,
                       'upper_frequency': upper_frequency,
                        'order':order}

    def _single_thread_execute(self, xarray: xr.DataArray, **kwargs):
        new_array = xr.apply_ufunc(filt_ops.xarray_whiten, xarray,
                                          input_core_dims=[['time']],
                                          output_core_dims=[['time']],
                                          kwargs={**self._kwargs ,
                                                  **{'delta':xarray.attrs['delta'],
                                                     'axis' :xarray.get_axis_num('time')}})

        return  new_array

    def _get_process(self):
        return 'whiten'

    def _add_operation_string(self):
        return 'f_whiten@window_ratio({}),frequency_window({})<f<({})hz'.format( \
            self._kwargs['smoothing_window_ratio'],self._kwargs['lower_frequency'],
            self._kwargs['upper_frequency'])




