"""
xarray.DataArray operations for use with Anxcor processing routines

"""

import numpy as np
import xarray as xr
import anxcor.filters as filt_ops
import pandas as pd
import  anxcor.abstractions as ab
from obspy.core import UTCDateTime
OPERATIONS_SEPARATION_CHARACTER = '\n'
SECONDS_2_NANOSECONDS = 1e9


class XArrayConverter(ab.XArrayProcessor):
    """
    converts an obspy stream into an xarray

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

    def _single_thread_execute(self, stream,*args,starttime=0, **kwargs):
        if stream is not None and len(stream)>0:
            return self._convert_trace_2_xarray(stream)
        return None

    def _convert_trace_2_xarray(self, stream):
        channels = []
        stations = []
        delta = None
        time_array = None
        data_type = None
        starttime = None
        latitude = None
        longitude = None
        elevation = None
        for trace in stream:
            station_code = self._get_station_id(trace)
            if station_code not in stations:
                stations.append(station_code)

            channel = trace.stats.channel
            if channel not in channels:
                channels.append(channel)
            if time_array is None:
                starttime = np.datetime64(trace.stats.starttime.datetime)
                endtime = np.datetime64(trace.stats.endtime.datetime)

                delta = trace.stats.delta
                timedelta = pd.Timedelta(delta, 's').to_timedelta64()

                time_array = np.arange(starttime, endtime + timedelta, timedelta)
                starttime = trace.stats.starttime.timestamp
                data_type = trace.stats.data_type
                stats_keys = trace.stats.keys()
                if 'latitude' in stats_keys:
                    latitude = trace.stats.latitude
                if 'longitude' in stats_keys:
                    longitude = trace.stats.longitude
                if 'elevation' in stats_keys:
                    elevation = trace.stats.elevation
        empty_array = np.zeros((len(channels), len(stations), len(time_array)))
        for trace in stream:
            chan = channels.index(trace.stats.channel)
            station_id = stations.index(self._get_station_id(trace))

            empty_array[chan, station_id, :] = trace.data
        return self._create_xarray(channels, data_type, delta, elevation, empty_array, latitude, longitude, starttime,
                                   stations, time_array)

    def _create_xarray(self, channels, data_type, delta, elevation, empty_array, latitude, longitude, starttime,
                       stations, time_array):
        xarray = xr.DataArray(empty_array, coords=[channels, stations, time_array],
                              dims=['channel', 'station_id', 'time'])
        xarray.name = data_type
        xarray.attrs['delta'] = delta
        xarray.attrs['starttime'] = starttime
        xarray.attrs['operations'] = 'xconvert'
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
    """
    applies a bandpass filter to a provided xarray
    """

    def __init__(self,upper_frequency=10.0,lower_frequency=0.01,order=2,taper=0.1,**kwargs):
        super().__init__(**kwargs)
        self._kwargs = {'upper_frequency':upper_frequency,
                        'lower_frequency':lower_frequency,
                        'order':order,
                        'taper':taper}

    def _single_thread_execute(self, xarray: xr.DataArray,*args, **kwargs):
        sampling_rate = 1.0 / xarray.attrs['delta']
        ufunc_kwargs = {**self._kwargs}

        if self._kwargs['upper_frequency'] > sampling_rate / 2:
            ufunc_kwargs['upper_frequency'] = sampling_rate / 2

        tapered_array  = xr.apply_ufunc(filt_ops.taper,xarray,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs=ufunc_kwargs)
        filtered_array = xr.apply_ufunc(filt_ops.butter_bandpass_filter, tapered_array,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={**ufunc_kwargs,**{
                                                'sample_rate': sampling_rate}})

        return filtered_array

    def _add_operation_string(self):
        return 'bandpass@{}<{}'.format(self._kwargs['lower_frequency'],
                                       self._kwargs['upper_frequency'])

    def _get_process(self):
        return 'bandpass'


class XArrayTaper(ab.XArrayProcessor):
    """
    tapers signals on an xarray timeseries

    Note
    --------
    most XArrayProcessors which operate in the frequency domain
    have tapering as part of the process.

    """

    def __init__(self,taper=0.1,**kwargs):
        super().__init__(**kwargs)
        self._kwargs = {'taper':taper}

    def _single_thread_execute(self, xarray: xr.DataArray,*args, **kwargs):
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
    """
    resamples the provided xarray to a lower frequency
    """

    def __init__(self, target_rate=10.0,**kwargs):
        super().__init__(**kwargs)
        self._kwargs['target_rate'] = target_rate

    def _single_thread_execute(self, xarray: xr.DataArray,*args,starttime=0,**kwargs):
        delta =  xarray.attrs['delta']
        sampling_rate = 1.0 / delta
        nyquist       = self._kwargs['target_rate'] / 2.0
        target_rule = str(int((1.0 /self._kwargs['target_rate']) * SECONDS_2_NANOSECONDS)) + 'N'

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

        resampled_array= filtered_array.resample(time=target_rule)\
            .interpolate('linear')

        starttime_datetime = np.datetime64(UTCDateTime(starttime).datetime)
        endtime_datetime   = np.datetime64(UTCDateTime(starttime +
                                                       delta*resampled_array.data.shape[-1]).datetime)
        timedelta = pd.Timedelta(delta, 's').to_timedelta64()
        time_array = np.arange(starttime_datetime, endtime_datetime + timedelta, timedelta)
        resampled_array = resampled_array.interp(time=time_array).bfill('time').ffill('time')


        return resampled_array

    def _add_metadata_key(self):
        return ('delta',1.0/self._kwargs['target_rate'])

    def _get_process(self):
        return 'resample'

    def _add_operation_string(self):
        return 'resampled@{}Hz'.format(self._kwargs['target_rate'])



class XArrayXCorrelate(ab.XArrayProcessor):
    """
    correlates two xarrays channel-wise in the frequency domain.

    """

    def __init__(self,max_tau_shift=100.0,**kwargs):
        super().__init__(**kwargs)
        self._kwargs['max_tau_shift']=max_tau_shift

    def _single_thread_execute(self, source_xarray: xr.DataArray, receiver_xarray: xr.DataArray,*args, **kwargs):
        if source_xarray is not None and receiver_xarray is not None:
            correlation = filt_ops.xarray_crosscorrelate(source_xarray,
                                             receiver_xarray,
                                                     **self._kwargs)
            return correlation
        return None

    def _get_process(self):
        return 'crosscorrelate'


    def _metadata_to_persist(self, xarray_1,xarray_2, **kwargs):
        if xarray_2 is None or xarray_1 is None:
            return None

        attrs = {'delta'    : xarray_1.attrs['delta'],
                 'starttime': xarray_1.attrs['starttime'],
                 'stacks'   : 1,
                 'endtime'  : xarray_1.attrs['starttime'] + xarray_1.attrs['delta'] * xarray_1.data.shape[-1],
                 'operations': xarray_1.attrs['operations'] + OPERATIONS_SEPARATION_CHARACTER + \
                               'correlated@{}<t<{}'.format(self._kwargs['max_tau_shift'],self._kwargs['max_tau_shift'])}
        if 'location' in xarray_1.attrs.keys() and 'location' in xarray_2.attrs.keys():
            attrs['location'] = {'src':xarray_1.attrs['location'],
                                 'rec':xarray_2.attrs['location']}
        return attrs

    def _add_operation_string(self):
        return None

    def  _should_process(self,xarray1, xarray2, *args):
        return xarray1 is not None and xarray2 is not None

    def _get_name(self,one,two):
        if one is not None and two is not None:
            return 'src:{} rec:{}'.format(one.name, two.name)
        else:
            return None

class XArrayRemoveMeanTrend(ab.XArrayProcessor):
    """
    removes the mean and trend of an xarray timeseries
    """
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
    """
    applies a temporal norm operation to an xarray timeseries
    """

    def __init__(self, time_window=10.0, lower_frequency=0.001,
                 upper_frequency=5.0, t_norm_type='reduce_channel',
                 rolling_procedure='mean',
                 reduction_procedure='max', **kwargs):
        super().__init__(**kwargs)
        self._kwargs['t_norm_type']         = t_norm_type
        self._kwargs['time_window']         = time_window
        self._kwargs['lower_frequency']     = lower_frequency
        self._kwargs['upper_frequency']     = upper_frequency
        self._kwargs['rolling_procedure']   = rolling_procedure
        self._kwargs['reduction_procedure'] = reduction_procedure


    def _single_thread_execute(self, xarray: xr.DataArray,*args, **kwargs):
        time_average         = self._kwargs['time_window']

        sampling_rate   = 1.0/xarray.attrs['delta']

        bandpassed_array = xr.apply_ufunc(filt_ops.butter_bandpass_filter,xarray,
                                        input_core_dims=[['time']],
                                        output_core_dims = [['time']],
                                        kwargs={**self._kwargs,
                                                **{'sample_rate':sampling_rate}})

        bandpassed_array = self._apply_rolling_method(bandpassed_array, sampling_rate)
        bandpassed_array = bandpassed_array.ffill('time').bfill('time')
        bandpassed_array = self._reduce_by_channel(bandpassed_array)


        xarray = xarray / bandpassed_array
        return xarray

    def _apply_rolling_method(self, bandpassed_array, sampling_rate):
        time_window       = self._kwargs['time_window']
        rolling_samples   = int(sampling_rate * time_window)
        rolling_procedure = self._kwargs['rolling_procedure']

        if rolling_procedure == 'mean':
            bandpassed_array = abs(bandpassed_array).rolling(time=rolling_samples,
                                                             min_periods=1,center=True).mean()
        elif rolling_procedure == 'median':
            bandpassed_array = abs(bandpassed_array).rolling(time=rolling_samples,
                                                             min_periods=1,center=True).median()
        elif rolling_procedure == 'min':
            bandpassed_array = abs(bandpassed_array).rolling(time=rolling_samples,
                                                             min_periods=1,center=True).min()
        elif rolling_procedure == 'max':
            bandpassed_array = abs(bandpassed_array).rolling(time=rolling_samples,
                                                             min_periods=1,center=True).max()
        else:
            bandpassed_array = abs(bandpassed_array)
        return bandpassed_array

    def _reduce_by_channel(self, bandpassed_array):
        reduction_procedure = self._kwargs['reduction_procedure']
        norm_type           = self._kwargs['t_norm_type']
        if norm_type == 'reduce_channel':
            if reduction_procedure   == 'mean':
                bandpassed_array = abs(bandpassed_array).mean(dim='channel')
            elif reduction_procedure == 'median':
                bandpassed_array = abs(bandpassed_array).median(dim='channel')
            elif reduction_procedure == 'min':
                bandpassed_array = abs(bandpassed_array).min(dim='channel')
            elif reduction_procedure == 'max':
                bandpassed_array = abs(bandpassed_array).max(dim='channel')
        return bandpassed_array

    def _get_process(self):
        return 'temp_norm'


    def _add_operation_string(self):
        return 'temporal_norm@type({}),rolling({}),reduce_by({}).'+\
               'time_mean({}),f_norm_basis({})<f<({})'.format(
            self._kwargs['t_norm_type'],self._kwargs['time_window'],
            self._kwargs['rolling_procedure'], self._kwargs['reduction_procedure'],
            self._kwargs['lower_frequency'],self._kwargs['upper_frequency'])



class XArrayWhiten(ab.XArrayProcessor):
    """
    whitens the frequency spectrum of a given xarray
    """
    def __init__(self, smoothing_window_ratio=10.0, lower_frequency=0.001,
                 upper_frequency=5.0, order=2, **kwargs):
        super().__init__(**kwargs)
        self._kwargs = {'smoothing_window_ratio': smoothing_window_ratio,
                       'lower_frequency': lower_frequency,
                       'upper_frequency': upper_frequency,
                        'order':order}

    def _single_thread_execute(self, xarray: xr.DataArray,*args, **kwargs):
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






