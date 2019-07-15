"""
xarray.DataArray operations for use with Anxcor processing routines

"""
import anxcor.constants as c
import numpy as np
import xarray as xr
import anxcor.filters as filt_ops
import pandas as pd
import  anxcor.abstractions as ab



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

    def _single_thread_execute(self, stream,*args, **kwargs):
        if stream is not None and len(stream)>0:
            return self._convert_trace_2_xarray(stream)
        return None

    def _convert_trace_2_xarray(self, stream):
        channels = []
        stations = []

        delta = None
        time_array = None
        data_type  = 'default'
        starttime  = None
        latitude   = None
        longitude  = None
        elevation  = None
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
                if hasattr(trace.stats,'data_type'):
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

    def __init__(self,upper_frequency=c.UPPER_CUTOFF_FREQ,
                    lower_frequency=c.LOWER_CUTOFF_FREQ,
                    order=c.FILTER_ORDER_BANDPASS,
                    taper=c.TAPER_DEFAULT,**kwargs):
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

        filtered_array = xr.apply_ufunc(filt_ops.taper, xarray,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={**self._kwargs})
        filtered_array = xr.apply_ufunc(filt_ops.butter_bandpass_filter, filtered_array,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={**ufunc_kwargs,**{
                                                'sample_rate': sampling_rate}})

        return filtered_array

    def _add_operation_string(self):
        return 'bandpass@{}<x(t)<{}'.format(self._kwargs['lower_frequency'],
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

    def __init__(self,taper=c.TAPER_DEFAULT,**kwargs):
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


class XArrayResample(ab.XArrayProcessor):
    """
    resamples the provided xarray to a lower frequency
    """

    def __init__(self, target_rate=c.RESAMPLE_DEFAULT,
                 taper=c.TAPER_DEFAULT,**kwargs):
        super().__init__(**kwargs)
        self._kwargs['target_rate'] = target_rate
        self._kwargs['taper']       = taper

    def _single_thread_execute(self, xarray: xr.DataArray,*args,starttime=0,**kwargs):
        delta =  xarray.attrs['delta']
        sampling_rate = 1.0 / delta
        nyquist       = self._kwargs['target_rate'] / 2.0
        target_rule = str(int((1.0 /self._kwargs['target_rate']) * c.SECONDS_2_NANOSECONDS)) + 'N'

        mean_array     = xarray.mean(dim=['time'])
        demeaned_array = xarray - mean_array
        detrend_array  = xr.apply_ufunc(filt_ops.detrend, demeaned_array,
                                        input_core_dims=[['time']],
                                        output_core_dims = [['time']],
                                        kwargs={'type':'linear'})
        tapered_array = xr.apply_ufunc(filt_ops.taper, detrend_array,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={**self._kwargs})
        filtered_array = xr.apply_ufunc(filt_ops.lowpass_filter,tapered_array,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={'upper_frequency':    nyquist,
                                                'sampling_rate': sampling_rate})

        resampled_array= filtered_array.resample(time=target_rule)\
            .interpolate('linear').bfill('time').ffill('time')
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

    def __init__(self,max_tau_shift=c.MAX_TAU_DEFAULT,
                 taper=c.TAPER_DEFAULT,**kwargs):
        super().__init__(**kwargs)
        self._kwargs['max_tau_shift']=max_tau_shift
        self._kwargs['taper'] = taper

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

        xarray_1 = xr.apply_ufunc(filt_ops.taper, xarray_1,
                                  input_core_dims=[['time']],
                                  output_core_dims=[['time']],
                                  kwargs={**self._kwargs})
        xarray_2 = xr.apply_ufunc(filt_ops.taper, xarray_2,
                                  input_core_dims=[['time']],
                                  output_core_dims=[['time']],
                                  kwargs={**self._kwargs})

        attrs = {'delta'    : xarray_1.attrs['delta'],
                 'starttime': xarray_1.attrs['starttime'],
                 'stacks'   : 1,
                 'endtime'  : xarray_1.attrs['starttime'] + xarray_1.attrs['delta'] * xarray_1.data.shape[-1],
                 'operations': xarray_1.attrs['operations'] + c.OPERATIONS_SEPARATION_CHARACTER + \
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

        detrend_array = xr.apply_ufunc(filt_ops.detrend, xarray,
                                       input_core_dims=[['time']],
                                       output_core_dims=[['time']],
                                       kwargs={'type': 'linear'},
                                       keep_attrs=True)
        demeaned = xr.apply_ufunc(filt_ops.detrend, detrend_array,
                                       input_core_dims=[['time']],
                                       output_core_dims=[['time']],
                                       kwargs={'type': 'constant'},
                                       keep_attrs=True)

        return demeaned

    def _add_operation_string(self):
        return 'remove_Mean&Trend'

    def _get_process(self):
        return 'remove_mean_trend'


class XArrayRolling(ab.XArrayProcessor):
    """
    whitens the frequency spectrum of a given xarray
    """
    def __init__(self, window=1.0,approach='rcc',center=True,
                 rolling_metric='mean',reduce_metric='mean',**kwargs):
        super().__init__(**kwargs)
        self._kwargs = {
            'window' : window,
            'approach' : approach,
            'center' : center,
            'reduce_metric' : reduce_metric,
            'rolling_metric': rolling_metric}

    def _single_thread_execute(self, xarray: xr.DataArray,*args, **kwargs):

        processed_array   = self._preprocess(xarray)
        pre_rolling       = self._pre_rolling_process(processed_array, xarray)
        rolling_array     = self._apply_rolling_method(pre_rolling, xarray)

        dim               = self._get_longest_dim_name(rolling_array)

        rolling_array     = rolling_array.ffill(dim).bfill(dim)
        post_rolling      = self._post_rolling_process(rolling_array,xarray)
        rolling_processed = self._reduce_by_channel(post_rolling)
        normalized_array  = processed_array / rolling_processed
        final_processed   = self._postprocess(normalized_array,xarray)
        return final_processed

    def _get_longest_dim_name(self,xarray):
        coords  = list(xarray.coords.values)
        index = np.argmax(xarray.data.shape)
        return coords[index]

    def _preprocess(self, xarray : xr.DataArray) -> xr.DataArray:
        return xarray

    def _pre_rolling_process(self,processed_array : xr.DataArray, xarray : xr.DataArray)-> xr.DataArray:
        return processed_array

    def _post_rolling_process(self,rolled_array : xr.DataArray, xarray : xr.DataArray)-> xr.DataArray:
        return rolled_array

    def _postprocess(self,normed_array : xr.DataArray, xarray : xr.DataArray)-> xr.DataArray:
        return normed_array

    def _get_rolling_samples(self,processed_xarray : xr.DataArray, xarray: xr.DataArray)-> int:
        return int(self._kwargs['window'] / xarray.attrs['delta'])


    def _apply_rolling_method(self, processed_xarray, original_xarray):
        rolling_samples = self._get_rolling_samples(processed_xarray, original_xarray)
        rolling_procedure = self._kwargs['rolling_metric']
        dim = self._get_longest_dim_name(processed_xarray)
        rolling_dict = {  dim : rolling_samples,
                         'min_periods':1,
                         'center': self._kwargs['center']
                          }
        if rolling_procedure == 'mean':
            xarray = abs(processed_xarray).rolling(**rolling_dict).mean()
        elif rolling_procedure == 'median':
            xarray = abs(processed_xarray).rolling(**rolling_dict).median()
        elif rolling_procedure == 'min':
            xarray = abs(processed_xarray).rolling(**rolling_dict).min()
        elif rolling_procedure == 'max':
            xarray = abs(processed_xarray).rolling(**rolling_dict).max()
        else:
            xarray = abs(processed_xarray)
        return xarray

    def _reduce_by_channel(self, xarray):
        approach = self._kwargs['apprach']
        if approach == 'src':
            reduction_procedure = self._kwargs['reduce_metric']
            if reduction_procedure == 'mean' or reduction_procedure is None:
                xarray = abs(xarray).mean(dim='channel')
            elif reduction_procedure == 'median':
                xarray = abs(xarray).median(dim='channel')
            elif reduction_procedure == 'min':
                xarray = abs(xarray).min(dim='channel')
            elif reduction_procedure == 'max':
                xarray = abs(xarray).max(dim='channel')
            elif 'z' in reduction_procedure.lower() or 'n' in reduction_procedure.lower() \
                    or 'e' in reduction_procedure.lower():
                for coordinate in list(xarray.coords['channel'].values):
                    if reduction_procedure in coordinate.lower():
                        return xarray[dict(channel=coordinate)]
        return xarray

    def _get_process(self):
        return 'rolling operation'



class XArrayTemporalNorm(XArrayRolling):
    """
    applies a temporal norm operation to an xarray timeseries
    """

    def __init__(self,lower_frequency=0.01,upper_frequency=5.0,order=3,taper=0.1, **kwargs):
        super().__init__(**kwargs)
        self._kwargs['lower_frequency']     = lower_frequency
        self._kwargs['upper_frequency']     = upper_frequency
        self._kwargs['taper']               = taper


    def _pre_rolling_process(self,processed_array : xr.DataArray, xarray : xr.DataArray):
        sampling_rate = 1.0 /xarray.attrs['delta']
        filtered_array = xr.apply_ufunc(filt_ops.taper, processed_array,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={**self._kwargs})
        filtered_array = xr.apply_ufunc(filt_ops.butter_bandpass_filter, filtered_array,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={**self._kwargs, **{
                                            'sample_rate': sampling_rate}})
        return filtered_array

    def _get_process(self):
        return 'temp_norm'


    def _add_operation_string(self):
        if self._kwargs['t_norm_type']=='src':
            op='temporal_norm@approach:{} ' \
               'window: {},' \
               'rolling_metric: {},' \
               'reduce_metric: {},' \
               'taper: {},' \
               'bandpass: {}<x(t)<{}'.format(
            self._kwargs['approach'],
            self._kwargs['window'],
            self._kwargs['rolling_metric'],
            self._kwargs['reduce_metric'],
            self._kwargs['taper'],
            self._kwargs['lower_frequency'],
            self._kwargs['upper_frequency'])
        else:
            op = 'temporal_norm@approach:{} ' \
                 'window: {},' \
                 'rolling_metric: {},' \
                 'taper: {},' \
                 'bandpass: {}<x(t)<{}'.format(
                self._kwargs['approach'],
                self._kwargs['window'],
                self._kwargs['rolling_metric'],
                self._kwargs['taper'],
                self._kwargs['lower_frequency'],
                self._kwargs['upper_frequency'])
        return op


class XArrayWhiten(XArrayRolling):
    """
    whitens the frequency spectrum of a given xarray
    """
    def __init__(self, lower_frequency=0.01,upper_frequency=5.0,order=3,taper=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self._kwargs = {
            'lower_frequency': lower_frequency,
            'upper_frequency': upper_frequency,
            'order':order,
            'taper':taper}


    def _preprocess(self,xarray):
        tapered_array = xr.apply_ufunc(filt_ops.taper, xarray,
                                       input_core_dims=[['time']],
                                       output_core_dims=[['time']],
                                       kwargs={**self._kwargs}, keep_attrs=True)
        fourier_array = filt_ops.create_fourier_xarray(tapered_array)
        return fourier_array

    def _postprocess(self,normed_array, xarray):
        return filt_ops.create_time_domain_array1(normed_array,xarray.copy())

    def _get_rolling_samples(self,processed_xarray, xarray):
        return int(self._kwargs['window'] * xarray.data.shape[-1])

    def _get_process(self):
        return 'whiten'

    def _add_operation_string(self):
        if self._kwargs['whiten_type'] == 'reduce_metric':
            op = 'Whiten@type:{} window_ratio: {},rolling_metric: {},reduce_metric: {},taper: {},bandpass: {}<x(t)<{}'.format(
                self._kwargs['approach'],
                self._kwargs['window'],
                self._kwargs['rolling_metric'],
                self._kwargs['reduce_metric'],
                self._kwargs['taper'],
                self._kwargs['lower_frequency'],
                self._kwargs['upper_frequency'])
        else:
            op = 'Whiten@type:{} window_ratio: {},rolling_metric: {},taper: {},bandpass: {}<x(t)<{}'.format(
                self._kwargs['approach'],
                self._kwargs['window'],
                self._kwargs['rolling_metric'],
                self._kwargs['taper'],
                self._kwargs['lower_frequency'],
                self._kwargs['upper_frequency'])
        return op






