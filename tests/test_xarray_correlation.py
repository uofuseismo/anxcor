import unittest
try:
    from tests.synthetic_trace_factory import create_random_trace, create_sinsoidal_trace_w_decay, create_triangle_trace
except:
    from synthetic_trace_factory import create_random_trace, create_sinsoidal_trace_w_decay, create_triangle_trace
from anxcor.xarray_routines import XArrayXCorrelate, XArrayConverter, XArrayRemoveMeanTrend, XArrayResample, XArrayTaper, XArrayProcessor
import os
from anxcor.containers import XArrayStack, AnxcorDatabase, XArrayCombine
import glob
from anxcor.numpyfftfilter import xarray_crosscorrelate
from anxcor.core import Anxcor
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np
import xarray as xr
from obspy.core import  UTCDateTime, Trace, Stream, read
import matplotlib.pyplot as plt
from obsplus.bank import WaveBank
import pytest
import timeit
if os.path.isdir('tests'):
    basedir='tests/'
else:
    basedir=''

class WavebankWrapper(AnxcorDatabase):

    def __init__(self, directory):
        super().__init__()
        self.bank = WaveBank(directory)
        self.bank.update_index()

    def get_waveforms(self, **kwargs):
        stream =  self.bank.get_waveforms(**kwargs)
        traces = []
        for trace in stream:
            data   = trace.data[:-1]
            if isinstance(data,np.ma.MaskedArray):
                data = np.ma.filled(data,fill_value=np.nan)
            header = {'delta':   trace.stats.delta,
                      'station': trace.stats.station,
                      'starttime':trace.stats.starttime,
                      'channel': trace.stats.channel,
                      'network': trace.stats.network}
            traces.append(Trace(data,header=header))
        return Stream(traces=traces)

    def get_stations(self):
        df = self.bank.get_availability_df()
        uptime = self.bank.get_uptime_df()

        def create_seed(row):
            network = row['network']
            station = row['station']
            return network + '.' + station

        df['seed'] = df.apply(lambda row: create_seed(row), axis=1)
        unique_stations = df['seed'].unique().tolist()
        return unique_stations


class XArrayCustomComponentNormalizer(XArrayProcessor):
    """
    normalizes preprocessed data based on a single component
    """
    def __init__(self,channel_norm='z',**kwargs):
        super().__init__(**kwargs)
        self._kwargs['channel_norm']=channel_norm.lower()

    def execute(self, xarray, *args, **kwargs):
        norm_chan_max = np.amax(np.abs(xarray.loc[dict(src_chan='z',rec_chan='z')]))
        xarray /= norm_chan_max
        return xarray

    def _add_operation_string(self):
        return 'channel normer zz'

    def get_name(self):
        return 'channel normer zz'

    def _persist_metadata(self, first_data, *args, **kwargs):
        return first_data.attrs


class DWellsDecimatedReader(AnxcorDatabase):
    """
    class for reading in time synced filed synced
    like fan chi lin's group likes to do
    """
    time_format = '%Y%m%d'

    def __init__(self, data_dir, station_file, station_number=None, source=None, extension=None):

        self.df = pd.read_csv(station_file, delim_whitespace=True,
                              skiprows=0,
                              names=['station', 'unknown_col', 'latitude', 'longitude'], dtype={'station': object})
        self.data_dir = data_dir
        self.extension = extension

    def get_waveforms(self, starttime=0, endtime=0, station=0, network=0):
        """
        get waveforms yields a stream of traces given a starttitme and endtime in utcdatetime floats
        """
        # print('starttime:{} endtime:{} network: {} station: {}'.format(starttime,endtime,network,station))
        start_format = self.format_utcdatetime(starttime)
        end_format = self.format_utcdatetime(endtime)
        if start_format == end_format:
            file_list = self.get_filelist(start_format, station)
        else:
            file_list = self.get_filelist(start_format, station) + self.get_filelist(end_format, station)
        stream = self.read_in_traces(file_list)
        stream.merge(method=1, interpolation_samples=-1)
        stream.trim(starttime=UTCDateTime(starttime), endtime=UTCDateTime(endtime), fill_value=np.nan)
        stream = self.assign_coordinates(stream)
        traces = []
        for trace in stream:
            if max(trace.data) > 0.0:
                # print('adding trace')
                self.cast_to_simple_channel(trace)
                traces.append(trace)
        new_stream = Stream(traces=traces)
        # print('trace size for query is {} and station is str {}'.format(len(new_stream),isinstance(station,str)))
        return new_stream

    def assign_coordinates(self, stream):
        new_traces = []
        for trace in stream:
            station=trace.stats.station.lstrip('0')
            new_trace = Trace(trace.data,
                                    header={'starttime':trace.stats.starttime,
                                            'delta': np.round(trace.stats.delta,decimals=5),
                                            'channel':trace.stats.channel,
                                            'station':station,
                                            'network':trace.stats.network})
            station   = station
            latitude  = self.df.loc[self.df['station'] == station]['latitude'].values[0]
            longitude = self.df.loc[self.df['station'] == station]['longitude'].values[0]
            new_trace.stats.coordinates = {'latitude': latitude, 'longitude': longitude}
            new_traces.append(new_trace)
        return Stream(traces=new_traces)

    def cast_to_simple_channel(self, trace):
        channel = trace.stats.channel
        target_channel = 'na'
        if 'z' in channel[-1].lower():
            target_channel = 'z'
        elif 'n' in channel[-1].lower():
            target_channel = 'n'
        elif 'e' in channel[-1].lower():
            target_channel = 'e'
        trace.stats.channel = target_channel

    def get_filelist(self, base_format, station):
        if isinstance(station, int):
            station = str(station)
        general_format = self.data_dir + os.sep + station + os.sep + base_format + '*.' + station + '.*sac*'
        if self.extension is not None:
            general_format = general_format + self.extension
        glob_result = glob.glob(general_format)
        return glob_result

    def get_single_sacfile(self, station):
        if isinstance(station, int):
            station = str(station)
        general_format = self.data_dir + os.sep + station + os.sep + '*sac*'
        glob_result = glob.glob(general_format)
        return glob_result

    def get_network_code_from_sacfile(self, station):
        file = self.get_single_sacfile(station)[0]
        stream = read(file)
        trace = stream[0]
        return trace.stats['network']

    def verify_sacfile_exists(self, station):
        file = self.get_single_sacfile(station)
        if file:
            return True
        else:
            return False

    def generate_networks(self):
        self.df['valid_station'] = self.df['station'].apply(lambda x: self.verify_sacfile_exists(x))
        self.df['network'] = np.nan
        self.df['station_id'] = np.nan
        df = self.df
        df.loc[df['valid_station'], 'network'] = df[df['valid_station']]['station'].apply(
            lambda x: self.get_network_code_from_sacfile(x))
        df.loc[df['valid_station'], 'station_id'] = df[df['valid_station']].apply(
            lambda x: str(x['network']) + '.' + str(x['station']), axis=1)
        self.df = df

    def read_in_traces(self, file_list):
        running_stream = Stream()
        for file in file_list:
            running_stream += read(file)
        return running_stream

    def format_utcdatetime(self, time):
        utcdatetime = UTCDateTime(time)
        string = utcdatetime.strftime(self.time_format)
        string = string.replace(" ", "")
        return string

    def get_stations(self):
        if 'network' not in self.df.columns:
            print('generating stream database')
            self.generate_networks()
        return list(self.df[self.df['valid_station'] == True]['station_id'].values)


def build_anxcor(tau):
    broadband_data_dir               = basedir+'test_data/correlation_integration_testing/Broadband'
    broadband_station_location_file  = basedir+'test_data/correlation_integration_testing/broadband_stationlist.txt'
    nodal_data_dir               =     basedir+'test_data/correlation_integration_testing/Nodal'
    nodal_station_location_file  =     basedir+'test_data/correlation_integration_testing/nodal_stationlist.txt'

    broadband_database = DWellsDecimatedReader(broadband_data_dir, broadband_station_location_file)
    nodal_database     = DWellsDecimatedReader(nodal_data_dir,     nodal_station_location_file,extension='d')
    window_length = 10*60.0
    #include_stations = ['Nodal.{}'.format(x) for x in range(1,10)]
    include_stations = ['UU.Broadband','Nodal.1','Nodal.2']

    taper_ratio     = 0.05
    target_rate     = 50.0
    correlate_kwargs= dict(max_tau_shift=tau,taper=taper_ratio)
    resample_kwargs = dict(target_rate=target_rate,lowpass=False)

    anxcor_main = Anxcor()
    anxcor_main.set_window_length(window_length)
    anxcor_main.set_must_only_include_station_pairs(include_stations)
    anxcor_main.add_dataset(broadband_database,'BB')
    anxcor_main.add_dataset(nodal_database, 'Nodal')
    anxcor_main.add_process(XArrayTaper(taper=taper_ratio))
    anxcor_main.add_process(XArrayRemoveMeanTrend())
    anxcor_main.add_process(XArrayTaper(taper=taper_ratio))
    anxcor_main.set_task_kwargs('crosscorrelate',correlate_kwargs)
    anxcor_main.set_task('post-correlate',XArrayCustomComponentNormalizer())
    return anxcor_main

def shift_trace(data,time=1.0,delta=0.1):
    sampling_rate = int(1.0/delta)
    random_data   = np.zeros((data.shape[0],data.shape[1],int(sampling_rate*time)))
    new_data      = np.concatenate((random_data,data),axis=2)[:,:,:data.shape[2]]
    return new_data

def create_example_xarrays(duration=20):
    e1 = create_sinsoidal_trace_w_decay(decay=0.9, station='h', network='v', channel='e', duration=duration)
    n1 = create_sinsoidal_trace_w_decay(decay=0.3, station='h', network='v', channel='n', duration=duration)
    z1 = create_sinsoidal_trace_w_decay(decay=0.4, station='h', network='v', channel='z', duration=duration)

    e2 = create_sinsoidal_trace_w_decay(decay=0.8, station='k', network='v', channel='e', duration=duration)
    n2 = create_sinsoidal_trace_w_decay(decay=0.7, station='k', network='v', channel='n', duration=duration)
    z2 = create_sinsoidal_trace_w_decay(decay=0.6, station='k', network='v', channel='z', duration=duration)

    syth_trace1 = converter(e1.copy() + z1.copy() + n1.copy())
    syth_trace2 = converter(e2.copy() + z2.copy() + n2.copy())
    return syth_trace1, syth_trace2


def create_example_xarrays_missing_channel(duration=20):
    e1 = create_sinsoidal_trace_w_decay(decay=0.9, station='h', network='v', channel='e', duration=duration)
    z1 = create_sinsoidal_trace_w_decay(decay=0.4, station='h', network='v', channel='z', duration=duration)

    e2 = create_sinsoidal_trace_w_decay(decay=0.8, station='k', network='v', channel='e', duration=duration)
    n2 = create_sinsoidal_trace_w_decay(decay=0.7, station='k', network='v', channel='n', duration=duration)
    z2 = create_sinsoidal_trace_w_decay(decay=0.6, station='k', network='v', channel='z', duration=duration)

    syth_trace1 = converter(e1.copy() + z1.copy())
    syth_trace2 = converter(e2.copy() + z2.copy() + n2.copy())
    return syth_trace1, syth_trace2


stacker   = XArrayStack()
converter = XArrayConverter()
class TestCorrelation(unittest.TestCase):


    def test_autocorrelation(self):
        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        synth_trace1 = converter(create_random_trace(station='1',network='v',duration=20))
        correlation = correlator(synth_trace1,synth_trace1)
        zero_target_index = correlation.data.shape[4]//2
        zero_source_index = np.argmax(correlation.data[0,0,0,:][0])
        assert zero_source_index == zero_target_index,'autocorrelation failed'

    def test_stacking_preserves_metadata(self):
        correlator = XArrayXCorrelate(max_tau_shift=5.0)
        combiner = XArrayCombine()
        syth_trace1 = converter(create_random_trace(station='1', network='v', duration=20))

        result_1 = combiner(correlator(syth_trace1, syth_trace1),None)
        result_2 = combiner(correlator(syth_trace1, syth_trace1),None)

        stack = stacker(result_1,result_2)

        df = stack.attrs['df']


        assert df['stacks'].values[0] == 2,'unexpected stack length'
        assert 'operations' in df.columns, 'operations did not persist through stacking'
        assert 'delta' in df.columns, 'delta did not persist through stacking'



    def test_has_delta(self):

        correlator = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1', network='v', duration=20))

        keys = correlator(syth_trace1, syth_trace1).attrs['df'].columns

        assert 'delta' in keys, 'delta not preserved through correlation'

    def test_has_stacks(self):

        correlator = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1', network='v', duration=20))

        keys = correlator(syth_trace1, syth_trace1).attrs['df'].columns

        assert 'stacks' in keys, 'stacks not preserved through correlation'

    def test_one_stack(self):

        correlator = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1', network='v', duration=20))

        attr = correlator(syth_trace1, syth_trace1).attrs

        assert attr['df']['stacks'].values[0] == 1,'stacks assigned improper value'

    def test_autocorrelation_delta_attr(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1',network='v',duration=20))

        correlation = correlator(syth_trace1,syth_trace1)

        assert 'delta' in correlation.attrs['df'].columns,'did not propagate the delta value'


    def test_name(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1',network='v',duration=20))

        correlation = correlator(syth_trace1,syth_trace1)

        assert isinstance(correlation.name,str),'improper name type'

    def test_array_is_real(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1',network='v',duration=20))

        correlation = correlator(syth_trace1,syth_trace1)

        assert correlation.data.dtype == np.float64,'improper data type'

    def test_correlation_length(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1',network='v'))
        syth_trace2 = converter(create_random_trace(station='2',network='v'))
        try:
            correlation = correlator(syth_trace1,syth_trace2)
            assert False,'failed to catch exception'
        except Exception:
            assert True, 'failed to catch exception'


    def test_shift_trace(self):
        max_tau_shift = 19.9
        correlator  = XArrayXCorrelate(max_tau_shift=max_tau_shift)
        time_shift  = 4
        synth_trace1 = converter(create_sinsoidal_trace_w_decay(decay=0.6,station='h',network='v',channel='z',duration=20))
        synth_trace2 = converter(create_sinsoidal_trace_w_decay(decay=0.4,station='h',network='w',channel='z',duration=20))
        synth_trace2 = xr.apply_ufunc(shift_trace,synth_trace2,input_core_dims=[['time']],
                                                             output_core_dims=[['time']],
                                                             kwargs={'time': time_shift,
                                                                     'delta':synth_trace2.attrs['delta']},
                                                             keep_attrs=True)
        correlation = correlator(synth_trace1,synth_trace2)
        correlation/= correlation.max()
        time_array  = correlation.coords['time'].values
        max_index   = np.argmax(correlation.data)
        tau_shift   = (time_array[max_index]- np.datetime64(UTCDateTime(0.0).datetime) ) / np.timedelta64(1, 's')
        assert tau_shift == -time_shift,'failed to correctly identify tau shift'

    def test_shift_trace_time_slice(self):
        max_tau_shift = 10.0
        correlator  = XArrayXCorrelate(max_tau_shift=max_tau_shift)
        time_shift  = 4
        synth_trace1 = converter(create_sinsoidal_trace_w_decay(decay=0.6,station='h',network='v',channel='z',duration=20))
        synth_trace2 = converter(create_sinsoidal_trace_w_decay(decay=0.4,station='h',network='w',channel='z',duration=20))
        synth_trace2 = xr.apply_ufunc(shift_trace,synth_trace2,input_core_dims=[['time']],
                                                             output_core_dims=[['time']],
                                                             kwargs={'time': time_shift,
                                                                     'delta':synth_trace2.attrs['delta']},
                                                             keep_attrs=True)
        correlation = correlator(synth_trace1,synth_trace2)
        correlation/= correlation.max()
        time_array  = correlation.coords['time'].values
        max_index   = np.argmax(correlation.data)
        tau_shift   = (time_array[max_index]- np.datetime64(UTCDateTime(0.0).datetime) ) / np.timedelta64(1, 's')
        assert tau_shift == -time_shift,'failed to correctly identify tau shift'

    def test_correct_pair_ee(self):
        max_tau_shift = 19
        correlator = XArrayXCorrelate(max_tau_shift=max_tau_shift)

        synth_trace_1, synth_trace_2 = create_example_xarrays()
        src_chan='e'
        rec_chan='e'

        correlation_source = correlator(synth_trace_1.copy(), synth_trace_2.copy())

        test_correlation   = correlator(synth_trace_1.sel(dict(channel=src_chan)).expand_dims('channel'),
                                        synth_trace_2.sel(dict(channel=rec_chan)).expand_dims('channel'))
        print(correlation_source)
        result_1     = correlation_source.loc[dict(src='v.h',rec='v.k',src_chan=src_chan, rec_chan=rec_chan)] - test_correlation
        assert 0 == np.sum(result_1.data)


    def test_correct_pair_zz(self):
        max_tau_shift = 19
        correlator = XArrayXCorrelate(max_tau_shift=max_tau_shift)

        synth_trace_1, synth_trace_2 = create_example_xarrays()
        src_chan = 'z'
        rec_chan = 'z'

        correlation_source = correlator(synth_trace_1.copy(), synth_trace_2.copy())

        test_correlation = correlator(synth_trace_1.sel(dict(channel=src_chan)).expand_dims('channel'),
                                      synth_trace_2.sel(dict(channel=rec_chan)).expand_dims('channel'))
        result_1 = correlation_source.loc[dict(src='v.h',rec='v.k',src_chan=src_chan, rec_chan=rec_chan)] - test_correlation
        assert 0 == np.sum(result_1.data)


    def test_scipy_equivalent(self):
        max_tau_shift = 19.99
        correlator = XArrayXCorrelate(max_tau_shift=max_tau_shift)

        synth_trace_1, synth_trace_2 = create_example_xarrays()
        src_chan = 'z'
        rec_chan = 'z'

        correlation_source = correlator(synth_trace_1.copy(), synth_trace_2.copy())

        test_correlation =  sig.correlate(np.asarray(synth_trace_1.sel(dict(channel=src_chan)).expand_dims('channel').data).squeeze(),
                                      np.asarray(synth_trace_2.sel(dict(channel=rec_chan)).expand_dims('channel').data).squeeze(),mode='full')
        corr_source    = np.asarray(correlation_source.loc[dict(src='v.h',rec='v.k',src_chan=src_chan, rec_chan=rec_chan)].data)
        assert np.sum(np.abs(test_correlation-corr_source))==0


    def test_correct_pair_ez(self):
        max_tau_shift = 8
        correlator = XArrayXCorrelate(max_tau_shift=max_tau_shift)

        synth_trace_1, synth_trace_2 = create_example_xarrays()
        src_chan = 'e'
        rec_chan = 'z'

        correlation_source = correlator(synth_trace_1.copy(), synth_trace_2.copy())

        test_correlation = correlator(synth_trace_1.sel(dict(channel=src_chan)).expand_dims('channel'),
                                      synth_trace_2.sel(dict(channel=rec_chan)).expand_dims('channel'))
        result_1 = correlation_source.loc[dict(src='v.h',rec='v.k',src_chan=src_chan, rec_chan=rec_chan)] - test_correlation
        assert 0 == np.sum(result_1.data)

    def test_correct_pair_ze(self):
        max_tau_shift = 8
        correlator = XArrayXCorrelate(max_tau_shift=max_tau_shift)

        synth_trace_1, synth_trace_2 = create_example_xarrays()
        src_chan = 'z'
        rec_chan = 'e'

        correlation_source = correlator(synth_trace_1.copy(), synth_trace_2.copy())

        test_correlation = correlator(synth_trace_1.sel(dict(channel=src_chan)).expand_dims('channel'),
                                      synth_trace_2.sel(dict(channel=rec_chan)).expand_dims('channel'))
        result_1 = correlation_source.loc[dict(src='v.h',rec='v.k',src_chan=src_chan, rec_chan=rec_chan)] - test_correlation
        assert 0 == np.sum(result_1.data)



    def test_correct_pair_ez24(self):
        correlator = XArrayXCorrelate(max_tau_shift=None)
        synth_trace_1, synth_trace_2 = create_example_xarrays_missing_channel()
        src_chan = 'z'
        rec_chan = 'e'

        correlation_source = correlator(synth_trace_1.copy(), synth_trace_2.copy())

        test_correlation = correlator(synth_trace_1.sel(dict(channel=src_chan)).expand_dims('channel'),
                                      synth_trace_2.sel(dict(channel=rec_chan)).expand_dims('channel'))
        result_1 = correlation_source.loc[dict(src='v.h',rec='v.k',src_chan=src_chan, rec_chan=rec_chan)] - test_correlation
        assert 0 == np.sum(result_1.data)


    def test_nonetype_in_out(self):
        correlator = XArrayXCorrelate()
        result = correlator(None,None, starttime=0, station=0)
        assert result == None










