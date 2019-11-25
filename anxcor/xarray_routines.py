"""
xarray.DataArray operations for use with Anxcor processing routines

"""
import numpy as np
import xarray as xr
import anxcor.filters as filt_ops
import anxcor.numpyfftfilter as npfilt_ops
import copy
import pandas as pd
from anxcor.abstractions import XArrayRolling, XArrayProcessor, _XArrayRead, _XArrayWrite


TAPER_DEFAULT   =0.05
RESAMPLE_DEFAULT=10.0
UPPER_CUTOFF_FREQ=5.0
LOWER_CUTOFF_FREQ=0.01
MAX_TAU_DEFAULT=100.0
FILTER_ORDER_BANDPASS=4
SECONDS_2_NANOSECONDS = 1e9
OPERATIONS_SEPARATION_CHARACTER = '->:'

## t-norm constants
T_NORM_TYPE='reduce_metric'
T_NORM_ROLLING_METRIC= 'mean'
T_NORM_REDUCE_METRIC = 'max'
T_NORM_WINDOW=10.0
T_NORM_LOWER_FREQ=0.001
T_NORM_UPPER_FREQ=0.05

## Whitening constants
WHITEN_REDUCE_METRIC = None
WHITEN_ROLLING_METRIC='mean'
WHITEN_WINDOW_RATIO=0.01
FILTER_ORDER_WHITEN=3
WHITEN_TYPE='reduce_metric'

class XArrayConverter(XArrayProcessor):
    """
    converts an obspy stream into an xarray

    """

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.writer = _XArrayWrite(None)
        self.reader = _XArrayRead(None)

    def execute(self, stream, *args, starttime=0,**kwargs):
        if stream is not None and len(stream)>0:
            return self._convert_trace_2_xarray(stream,starttime)
        return None

    def _convert_trace_2_xarray(self, stream,starttime):
        timeseries   = self._get_timeseries(stream)
        coordinates  = self._get_coordinates(stream)
        delta        = self._get_delta(stream)
        station_code = self._get_station_id(stream)
        name         = self._get_dataname(stream)
        channels     = self._get_channels(stream)

        data = self._create_numpy_data(channels, stream)

        metadata={'coords': {'time'    :timeseries,
                             'channels':channels,
                             'station_id':[station_code]},
                  'name':name,
                  'geographic_coordinates':coordinates,
                  'delta':delta,
                  'starttime':starttime
                  }
        xarray = self._create_xarray(data, metadata)
        return xarray

    def _create_xarray(self, data, metadata):
        xarray = xr.DataArray(data, coords=[metadata['coords']['channels'],
                                             metadata['coords']['station_id'],
                                             metadata['coords']['time']],
                              dims=['channel', 'station_id', 'time'])
        xarray.name = metadata['name']
        xarray.attrs['delta'] =  metadata['delta']
        xarray.attrs['starttime'] =  metadata['starttime']
        xarray.attrs['operations'] = 'xconvert'
        if metadata['geographic_coordinates'] is not None:
            xarray.attrs['location'] = metadata['geographic_coordinates']
        return xarray

    def _get_channels(self,stream):
        return [trace.stats.channel for trace in stream ]

    def _create_numpy_data(self, channels, stream):
        data = np.zeros((len(channels), 1, len(stream[0].data)))
        for trace in stream:
            chan = channels.index(trace.stats.channel)
            data[chan, 0, :] = trace.data
        return data

    def _get_station_id(self, stream):
        network = stream[0].stats.network
        station = stream[0].stats.station
        return network + '.' + station

    def _get_dataname(self, stream):
        name = 'default'
        if hasattr(stream[0].stats,'name'):
            name = stream[0].stats.name
        return name

    def _get_delta(self,stream):
        return stream[0].stats.delta

    def _get_coordinates(self,stream):
        if hasattr(stream[0].stats,'coordinates'):
            return stream[0].stats.coordinates
        return None

    def _get_timeseries(self, stream):
        starttime = np.datetime64(stream[0].stats.starttime.datetime)
        endtime   = np.datetime64(stream[0].stats.endtime.datetime)
        delta     = stream[0].stats.delta
        timedelta = pd.Timedelta(delta, 's').to_timedelta64()
        time_array= np.arange(starttime, endtime, timedelta)
        delta_num = 1
        while len(stream[0].data)!=len(time_array):
            if len(stream[0].data)>len(time_array):
                time_array=np.append(time_array,endtime+delta_num*timedelta)
                delta_num+=1
            else:
                time_array=time_array[:-1]
        return time_array

    def _get_starttime(self,stream):
        return stream[0].stats.starttime.timestamp

    def _persist_metadata(self, *param, **kwargs):
        return None

    def get_name(self):
        return 'xconvert'

    def _get_name(self,*args):
        return None



class XArrayBandpass(XArrayProcessor):
    """
    applies a bandpass filter to a provided xarray
    """

    def __init__(self, freqmax=10.0,
                 freqmin=0.001,
                 order=2,
                 taper=0.01, **kwargs):
        super().__init__(**kwargs)
        self._kwargs = {'freqmax':freqmax,
                        'freqmin':freqmin,
                        'order':order,
                        'taper':taper}

    def execute(self, xarray: xr.DataArray, *args, **kwargs):
        if 'delta' in xarray.attrs.keys():
            delta = xarray.attrs['delta']
        else:
            delta = xarray.attrs['df']['delta'].values[0]
        sampling_rate = 1.0 / delta
        ufunc_kwargs = {**self._kwargs}

        if self._kwargs['freqmax'] > sampling_rate / 2:
            ufunc_kwargs['freqmax'] = sampling_rate / 2


        filtered_array = xr.apply_ufunc(filt_ops.bandpass_in_time_domain_sos,xarray,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={**ufunc_kwargs,**{
                                                'sample_rate': sampling_rate}})

        return filtered_array

    def _add_operation_string(self):
        return 'bandpass@{}<x(t)<{}'.format(self._kwargs['freqmin'],
                                       self._kwargs['freqmax'])

    def get_name(self):
        return 'bandpass'


class XArrayNormalizer(XArrayProcessor):
    """
    applies a bandpass filter to a provided xarray
    """

    def __init__(self,norm_type=0,**kwargs):
        """

        Normalizes a cross-correlation
        """
        super().__init__(**kwargs)
        self._kwargs['norm_type']=norm_type

    def execute(self, xarray: xr.DataArray, *args, **kwargs):
        norm_type = self._kwargs['norm_type']
        if norm_type==0:
            pass
        elif norm_type==1:
            pass
        elif norm_type==2:
            pass
        else:
            pass
        return xarray

    def _add_operation_string(self):
        return 'bandpass@{}<x(t)<{}'.format(self._kwargs['lower_frequency'],
                                       self._kwargs['upper_frequency'])

    def get_name(self):
        return 'bandpass'


class XArrayTaper(XArrayProcessor):
    """
    tapers signals on an xarray timeseries

    Note
    --------
    most XArrayProcessors which operate in the frequency domain
    have tapering as part of the process.

    """

    def __init__(self,taper=0.05,type='hann',**kwargs):
        super().__init__(**kwargs)
        self._kwargs['taper']=taper
        self._kwargs['type']=type

    def execute(self, xarray: xr.DataArray, *args, **kwargs):
        filtered_array = xr.apply_ufunc(filt_ops.taper_func, xarray,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={**self._kwargs})

        return filtered_array

    def _add_operation_string(self):
        return 'taper@{}%'.format(self._kwargs['taper']*100)

    def get_name(self):
        return 'taper'


class XArrayResample(XArrayProcessor):
    """
    resamples the provided xarray to a lower frequency
    """

    def __init__(self, target_rate=RESAMPLE_DEFAULT,
                 taper=TAPER_DEFAULT,order=1,**kwargs):
        super().__init__(**kwargs)
        self._kwargs['target_rate'] = target_rate
        self._kwargs['taper']       = taper
        self._kwargs['order']       = order

    def execute(self, xarray: xr.DataArray, *args, starttime=0, **kwargs):
        delta =  xarray.attrs['delta']
        order = self._kwargs['order']
        sampling_rate = 1.0 / delta
        nyquist       = self._kwargs['target_rate'] / 2.0
        target_rule = str(int((1.0 /self._kwargs['target_rate']) * SECONDS_2_NANOSECONDS)) + 'N'

        mean_array     = xarray.mean(dim=['time'])
        demeaned_array = xarray - mean_array
        detrend_array  = xr.apply_ufunc(filt_ops.detrend, demeaned_array,
                                        input_core_dims=[['time']],
                                        output_core_dims = [['time']],
                                        kwargs={'type':'linear'})
        tapered_array = xr.apply_ufunc(filt_ops.taper_func, detrend_array,
                                       input_core_dims=[['time']],
                                       output_core_dims=[['time']],
                                       kwargs={**self._kwargs})
        filtered_array = xr.apply_ufunc(filt_ops.lowpass_filter,tapered_array,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={'upper_frequency':    nyquist,
                                                'sample_rate': sampling_rate,
                                                'order': order})

        resampled_array= filtered_array.resample(time=target_rule)\
            .interpolate('linear').bfill('time').ffill('time')
        return resampled_array

    def _add_metadata_key(self):
        return ('delta',1.0/self._kwargs['target_rate'])

    def get_name(self):
        return 'resample'

    def _add_operation_string(self):
        return 'resampled@{}Hz'.format(self._kwargs['target_rate'])



class XArrayXCorrelate(XArrayProcessor):
    """
    correlates two xarrays channel-wise in the frequency domain.

    """

    def __init__(self,max_tau_shift=MAX_TAU_DEFAULT,
                 taper=TAPER_DEFAULT,**kwargs):
        super().__init__(**kwargs)
        self._kwargs['max_tau_shift']=max_tau_shift
        self._kwargs['taper'] = taper


    def execute(self, source_xarray: xr.DataArray, receiver_xarray: xr.DataArray, *args, **kwargs):
        if source_xarray is not None and receiver_xarray is not None:
            correlation = npfilt_ops.xarray_crosscorrelate(source_xarray,
                                                           receiver_xarray,
                                                           **self._kwargs)

            return correlation
        return None

    def get_name(self):
        return 'crosscorrelate'


    def _persist_metadata(self, xarray_1, xarray_2, **kwargs):
        if xarray_2 is None or xarray_1 is None:
            return None
        rows=[]
        row ={
                'src':list(xarray_1.coords['station_id'].values)[0],
                'rec':list(xarray_2.coords['station_id'].values)[0],
                'delta'         : xarray_1.attrs['delta'],
                'stacks'        : 1,
                'operations'    : xarray_1.attrs['operations'] + OPERATIONS_SEPARATION_CHARACTER + \
                               'correlated@{}<t<{}'.format(self._kwargs['max_tau_shift'],self._kwargs['max_tau_shift'])}
        if 'location' in xarray_1.attrs.keys() and 'location' in xarray_2.attrs.keys():
            if len(xarray_1.attrs['location'].keys()) > 2:
                row['src_elevation'] = xarray_1.attrs['location']['elevation']
                row['rec_elevation']=xarray_2.attrs['location']['elevation']
            row['rec_latitude']=xarray_2.attrs['location']['latitude']
            row['rec_longitude']=xarray_2.attrs['location']['longitude']
            row['src_latitude']=xarray_1.attrs['location']['latitude']
            row['src_longitude']=xarray_1.attrs['location']['longitude']
        for src_chan in list(xarray_1.coords['channel'].values):
            for rec_chan in list(xarray_2.coords['channel'].values):
                row['src channel']=src_chan
                row['rec channel']=rec_chan
                rows.append(row)
                row = copy.deepcopy(row)
        df = pd.DataFrame(data=rows)
        return {'df':df}

    def _use_operation(self):
        return False

    def _add_operation_string(self):
        return 'correlated@{}<t<{}'.format(self._kwargs['max_tau_shift'],self._kwargs['max_tau_shift'])

    def  _child_can_process(self, xarray1, xarray2, *args):
        return xarray1 is not None and xarray2 is not None

    def _get_name(self,one,two):
        if one is not None and two is not None:
            return 'src:{} rec:{}'.format(one.name, two.name)
        else:
            return None

class XArrayRemoveMeanTrend(XArrayProcessor):
    """
    removes the mean and trend of an xarray timeseries

    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)


    def execute(self, xarray, *args, **kwargs):

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

    def get_name(self):
        return 'remove_mean_trend'

class XArray9ComponentNormalizer(XArrayProcessor):
    """
    normalizes a correlation based on a source and receiver channel
    """
    def __init__(self,src_chan='z',rec_chan='z',**kwargs):
        super().__init__(**kwargs)
        self._kwargs['src_chan']=src_chan
        self._kwargs['rec_chan']=rec_chan

    def execute(self, xarray, *args, **kwargs):
        src_chan = [x for x in list(xarray.coords['src_chan'].values)
                    if self._kwargs['src_chan'] in x.lower()]
        rec_chan = [x for x in list(xarray.coords['rec_chan'].values)
                    if self._kwargs['rec_chan'] in x.lower()]
        try:
            data_slice = xarray.loc[dict(src_chan=src_chan[0],rec_chan=rec_chan[0])].data
        except Exception:
            return None

        norm_chan_max = np.max(np.abs(data_slice.ravel()))
        xarray /= norm_chan_max
        return xarray

    def _add_operation_string(self):
        return '9ch norm'

    def get_name(self):
        return '9ch norm'

    def _persist_metadata(self, first_data, *args, **kwargs):
        return first_data.attrs


class XArrayTemporalNorm(XArrayRolling):
    """
    applies a temporal norm operation to an xarray timeseries
    """

    def __init__(self,lower_frequency=0.01,upper_frequency=5.0,taper=0.1, **kwargs):
        super().__init__(**kwargs)
        self._kwargs['lower_frequency']     = lower_frequency
        self._kwargs['upper_frequency']     = upper_frequency
        self._kwargs['taper']               = taper


    def _pre_rolling_process(self,processed_array : xr.DataArray, xarray : xr.DataArray):
        sample_rate = 1.0 / xarray.attrs['delta']

        tripled_xarray = filt_ops.xarray_triple_by_reflection(processed_array)
        tapered        = xr.apply_ufunc(filt_ops.taper_func, tripled_xarray,
                                       input_core_dims=[['time']],
                                       output_core_dims=[['time']],
                                       kwargs={**self.get_kwargs()},
                                       keep_attrs=True)
        bp_data        = xr.apply_ufunc(filt_ops.bandpass_in_time_domain_sos,tapered,
                                       input_core_dims=[['time']],
                                       output_core_dims=[['time']],
                                       kwargs={**{'sample_rate':sample_rate},**self.get_kwargs()},
                                       keep_attrs=True)
        return abs(bp_data)



    def _post_rolling_process(self,rolled_array : xr.DataArray, xarray : xr.DataArray)-> xr.DataArray:
        original_dims = filt_ops.xarray_center_third_time(rolled_array,xarray)
        return original_dims


    def _postprocess(self, normed_array, xarray):
        filtered_array = xr.apply_ufunc(filt_ops.taper_func, normed_array,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={**self._kwargs})

        return filtered_array

    def get_name(self):
        return 'temp_norm'


    def _add_operation_string(self):
        if self._kwargs['approach']=='src':
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

    see XArrayRolling for general rolling kwargs
    """
    def __init__(self, lower_frequency=0.01,upper_frequency=5.0,order=3,taper=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self._kwargs['lower_frequency'] = lower_frequency
        self._kwargs['upper_frequency'] = upper_frequency
        self._kwargs['order'] = order
        self._kwargs['taper'] = taper


    def _preprocess(self,xarray):
        fourier_array = filt_ops.xarray_time_2_freq(xarray)
        return fourier_array

    def _pre_rolling_process(self,processed_array : xr.DataArray, xarray : xr.DataArray):
        return abs(processed_array)

    def _postprocess(self,normed_array, xarray):
        sample_rate = 1.0/xarray.attrs['delta']
        time_domain_array = filt_ops.xarray_freq_2_time(normed_array, xarray)
        tapered_array     = xr.apply_ufunc(filt_ops.taper_func, time_domain_array,
                                            input_core_dims=[['time']],
                                            output_core_dims=[['time']],
                                            kwargs={**self._kwargs,'sample_rate':sample_rate}, keep_attrs=True)

        return tapered_array

    def _post_rolling_process(self,rolled_array : xr.DataArray, xarray : xr.DataArray)-> xr.DataArray:
        return rolled_array

    def _get_rolling_samples(self,processed_xarray, xarray):
        return int(self._kwargs['window'] * xarray.data.shape[-1]/2)

    def get_name(self):
        return 'whiten'

    def _add_operation_string(self):
        if self._kwargs['approach'] == 'src':
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





