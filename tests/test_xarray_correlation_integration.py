import unittest
from anxcor.xarray_routines import XArrayXCorrelate, XArrayConverter, XArrayRemoveMeanTrend, XArrayTaper, XArrayProcessor
import os
from anxcor.containers import  AnxcorDatabase
import glob
from anxcor.core import Anxcor
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np
import xarray as xr
from obspy.core import  UTCDateTime, Trace, Stream, read
import matplotlib.pyplot as plt
import pytest

def get_dv():
    stream =read('tests/test_data/correlation_integration_testing/Nodal/1/20171001000022180.1.EHZ.Nodal.sac.d')
    stats = stream[0].stats
    stats.delta = 0.02
    stats.channel='z'

    return Stream(traces=[Trace(stream[0].data,header=stats)])

def get_FORU():
    stream = read('tests/test_data/correlation_integration_testing/Broadband/Broadband/20171001.Broadband.EHZ.sac')
    stats = stream[0].stats
    stats.channel = 'z'
    return Stream(traces=[Trace(stream[0].data,header=stats)])


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
    broadband_data_dir               = 'tests/test_data/correlation_integration_testing/Broadband'
    broadband_station_location_file  = 'tests/test_data/correlation_integration_testing/broadband_stationlist.txt'
    nodal_data_dir               =     'tests/test_data/correlation_integration_testing/Nodal'
    nodal_station_location_file  =     'tests/test_data/correlation_integration_testing/nodal_stationlist.txt'

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
   # anxcor_main.set_task('post-correlate',XArrayCustomComponentNormalizer())
    return anxcor_main


converter = XArrayConverter()
correlate = XArrayXCorrelate(max_tau_shift=None)
taper     = XArrayTaper(taper=0.05,type='hann')
rmmean    = XArrayRemoveMeanTrend()
def convert_xarray_to_np_array(xarray,src_chan='z',rec_chan='z'):
    xarray_sub = xarray.loc[dict(src_chan=src_chan,rec_chan=rec_chan)].squeeze()
    return xarray_sub.data

def yield_obspy_default_stream():
    trace = read()[0]
    trace.stats.channel='z'
    return Stream(traces=[trace])

class TestCorrelation(unittest.TestCase):


    def test_conversion_correlation(self):
        stream = yield_obspy_default_stream()
        target_stream = stream.copy()
        source_xarray = converter(stream)

        correlate_xarray = correlate(source_xarray,source_xarray)
        correlated_source_np_array = convert_xarray_to_np_array(correlate_xarray)
        correlated_target_np_array  = sig.fftconvolve(target_stream[0].data,target_stream[0].data[::-1],mode='full')

        np.testing.assert_allclose(correlated_source_np_array,correlated_target_np_array)


    def test_obspy_taper_identical_ones(self):
        stream = yield_obspy_default_stream()
        target_stream = stream.copy()
        source_xarray = converter(stream)
        source_xarray = taper(source_xarray)
        target_stream.taper(0.05,type='hann')
        np.testing.assert_allclose(source_xarray.data.squeeze(),target_stream[0].data)


    def test_obspy_taper_identical(self):
        stream = yield_obspy_default_stream()
        target_stream = stream.copy()
        source_xarray = converter(stream)
        source_xarray = taper(source_xarray)
        target_stream.taper(0.05,type='hann')
        np.testing.assert_allclose(source_xarray.data.squeeze(),target_stream[0].data)


    def test_obspy_taper_demean_identical_ones(self):
        stream = yield_obspy_default_stream()
        stream[0].data = np.ones(stream[0].data.shape)
        target_stream = stream.copy()
        source_xarray = converter(stream)
        source_xarray = taper(source_xarray)
        source_xarray = rmmean(source_xarray)
        source_xarray = taper(source_xarray)


        target_stream.taper(0.05,type='hann')
        target_stream.detrend(type='constant')
        target_stream.detrend(type='linear')
        target_stream.taper(0.05,type='hann')

        source = source_xarray.data.squeeze()
        target = target_stream[0].data

        np.testing.assert_allclose(source,target)

    def test_obspy_taper_demean_identical(self):
        stream = yield_obspy_default_stream()
        target_stream = stream.copy()
        source_xarray = converter(stream)
        source_xarray = taper(source_xarray)
        source_xarray = rmmean(source_xarray)
        source_xarray = taper(source_xarray)


        target_stream.taper(0.05,type='hann')
        target_stream.detrend(type='constant')
        target_stream.detrend(type='linear')
        target_stream.taper(0.05,type='hann')

        source = source_xarray.data.squeeze()
        target = target_stream[0].data

        np.testing.assert_allclose(source,target)

    def test_obspy_taper_demean_correlate_identical_ones(self):
        stream = yield_obspy_default_stream()
        stream[0].data = np.ones(stream[0].data.shape)
        target_stream = stream.copy()

        source_xarray = converter(stream)
        source_xarray = taper(source_xarray)
        source_xarray = rmmean(source_xarray)
        source_xarray = taper(source_xarray)
        correlate_array = correlate(source_xarray,source_xarray)
        source = convert_xarray_to_np_array(correlate_array)

        target_stream.taper(0.05,type='hann')
        target_stream.detrend(type='linear')
        target_stream.detrend(type='constant')
        target_stream.taper(0.05,type='hann')
        target = sig.correlate(target_stream[0].data, target_stream[0].data, mode='full')

        np.testing.assert_allclose(source,target)


    def test_obspy_taper_demean_correlate_identical(self):
        stream = yield_obspy_default_stream()
        stream[0].data = np.ones(stream[0].data.shape)
        target_stream = stream.copy()

        source_xarray = converter(stream)
        source_xarray = taper(source_xarray)
        source_xarray = rmmean(source_xarray)
        source_xarray = taper(source_xarray)
        correlate_array = correlate(source_xarray,source_xarray)
        source = convert_xarray_to_np_array(correlate_array)

        target_stream.taper(0.05,type='hann')
        target_stream.detrend(type='linear')
        target_stream.detrend(type='constant')
        target_stream.taper(0.05,type='hann')
        target = sig.correlate(target_stream[0].data, target_stream[0].data, mode='full')

        np.testing.assert_allclose(source,target)


    def test_obspy_taper_demean_correlate_DV(self):
        stream = get_dv()
        stream[0].data = np.ones(stream[0].data.shape)
        target_stream = stream.copy()

        source_xarray = converter(stream)
        source_xarray = taper(source_xarray)
        source_xarray = rmmean(source_xarray)
        source_xarray = taper(source_xarray)
        correlate_array = correlate(source_xarray,source_xarray)
        source = convert_xarray_to_np_array(correlate_array)

        target_stream.taper(0.05,type='hann')
        target_stream.detrend(type='linear')
        target_stream.detrend(type='constant')
        target_stream.taper(0.05,type='hann')
        target = sig.correlate(target_stream[0].data, target_stream[0].data, mode='full')

        np.testing.assert_allclose(source,target)


    def test_obspy_taper_demean_correlate_different_ones(self):
        dv   = get_dv()
        dv[0].data=np.sin(np.linspace(0,2*np.pi,num=100))
        foru = get_FORU()
        foru[0].data = np.cos(np.linspace(0, 2 * np.pi, num=100))

        dv_xarray     = converter(dv.copy())
        foru_xarray   = converter(foru.copy())

        foru_xarray = taper(foru_xarray)
        foru_xarray = rmmean(foru_xarray)
        foru_xarray = taper(foru_xarray)

        dv_xarray   = taper(dv_xarray)
        dv_xarray   = rmmean(dv_xarray)
        dv_xarray   = taper(dv_xarray)

        dv.taper(0.05)
        dv.detrend(type='linear')
        dv.detrend(type='constant')
        dv.taper(0.05)

        foru.taper(0.05)
        foru.detrend(type='linear')
        foru.detrend(type='constant')
        foru.taper(0.05)

        correlation_xarray = correlate(foru_xarray,dv_xarray)
        numpy_source_array = convert_xarray_to_np_array(correlation_xarray)

        correlation_target = sig.fftconvolve(foru[0].data,dv[0].data[::-1])

        np.testing.assert_allclose(numpy_source_array,correlation_target)


    def test_obspy_taper_demean_correlate_long_vector(self):
        dv   = get_dv()
        dv[0].data=np.sin(np.linspace(0,200*np.pi,num=10000))
        foru = get_FORU()
        foru[0].data = np.cos(np.linspace(0, 200 * np.pi, num=10000))

        dv_xarray     = converter(dv.copy())
        foru_xarray   = converter(foru.copy())

        foru_xarray = taper(foru_xarray)
        foru_xarray = rmmean(foru_xarray)
        foru_xarray = taper(foru_xarray)

        dv_xarray   = taper(dv_xarray)
        dv_xarray   = rmmean(dv_xarray)
        dv_xarray   = taper(dv_xarray)

        dv.taper(0.05)
        dv.detrend(type='linear')
        dv.detrend(type='constant')
        dv.taper(0.05)

        foru.taper(0.05)
        foru.detrend(type='linear')
        foru.detrend(type='constant')
        foru.taper(0.05)

        correlation_xarray = correlate(foru_xarray,dv_xarray)
        numpy_source_array = convert_xarray_to_np_array(correlation_xarray)

        correlation_target = sig.fftconvolve(foru[0].data,dv[0].data[::-1])

        plt.figure(figsize=(10,4))
        plt.plot(numpy_source_array)
        plt.plot(correlation_target)
        plt.show()

        np.testing.assert_allclose(numpy_source_array,correlation_target)

    def test_obspy_taper_demean_correlate_long_vector(self):
        dv   = get_dv()
        foru = get_FORU()
        #foru[0].data = np.cos(np.linspace(0, 200 * np.pi, num=10000))

        dv_xarray     = converter(dv.copy())
        foru_xarray   = converter(foru.copy())

        foru_xarray = taper(foru_xarray)
        foru_xarray = rmmean(foru_xarray)
        foru_xarray = taper(foru_xarray)

        dv_xarray   = taper(dv_xarray)
        dv_xarray   = rmmean(dv_xarray)
        dv_xarray   = taper(dv_xarray)

        dv.taper(0.05)
        dv.detrend(type='linear')
        dv.detrend(type='constant')
        dv.taper(0.05)

        foru.taper(0.05)
        foru.detrend(type='linear')
        foru.detrend(type='constant')
        foru.taper(0.05)

        correlation_xarray = correlate(foru_xarray,dv_xarray)
        numpy_source_array = convert_xarray_to_np_array(correlation_xarray)
        #numpy_source_array/= np.amax(np.abs(numpy_source_array))

        correlation_target = sig.fftconvolve(foru[0].data,dv[0].data[::-1],mode='full')
        #correlation_target/= np.amax(np.abs(correlation_target))

        plt.figure(figsize=(10,4))
        plt.plot(numpy_source_array)
        plt.plot(correlation_target)

        plt.xlim([4320400,4320600])
        plt.show()

        np.testing.assert_allclose(numpy_source_array,correlation_target)


    def test_obspy_trim_taper_demean_correlate_time_window(self):
        starttime = UTCDateTime("2017-10-01 06:00:00")
        endtime = UTCDateTime("2017-10-01 06:10:00")
        dv   = get_dv().trim(starttime,endtime)
        foru = get_FORU().trim(starttime,endtime)
        #foru[0].data = np.cos(np.linspace(0, 200 * np.pi, num=10000))

        dv_xarray     = converter(dv.copy())
        foru_xarray   = converter(foru.copy())

        foru_xarray = taper(foru_xarray)
        foru_xarray = rmmean(foru_xarray)
        foru_xarray = taper(foru_xarray)

        dv_xarray   = taper(dv_xarray)
        dv_xarray   = rmmean(dv_xarray)
        dv_xarray   = taper(dv_xarray)

        dv.taper(0.05)
        dv.detrend(type='linear')
        dv.detrend(type='constant')
        dv.taper(0.05)

        foru.taper(0.05)
        foru.detrend(type='linear')
        foru.detrend(type='constant')
        foru.taper(0.05)

        correlation_xarray = correlate(foru_xarray,dv_xarray)
        numpy_source_array = convert_xarray_to_np_array(correlation_xarray)
        #numpy_source_array/= np.amax(np.abs(numpy_source_array))

        correlation_target = sig.fftconvolve(foru[0].data,dv[0].data[::-1],mode='full')
        #correlation_target/= np.amax(np.abs(correlation_target))

        plt.figure(figsize=(10,4))
        plt.plot(numpy_source_array)
        plt.plot(correlation_target)
        plt.xlim([30000,40000])
        plt.show()

        np.testing.assert_allclose(numpy_source_array,correlation_target)


    def test_obspy_trim_taper_demean_correlate_with_process(self):
        starttime = UTCDateTime("2017-10-01 06:00:00")
        endtime = UTCDateTime("2017-10-01 06:10:00")
        dv   = get_dv().trim(starttime,endtime)
        foru = get_FORU().trim(starttime,endtime)
        #foru[0].data = np.cos(np.linspace(0, 200 * np.pi, num=10000))

        anxcor_main = build_anxcor(None)
        dv_xarray   = anxcor_main._station_window_operations(dv,starttime=starttime.timestamp)
        foru_xarray = anxcor_main._station_window_operations(foru, starttime=starttime.timestamp)

        dv.taper(0.05)
        dv.detrend(type='linear')
        dv.detrend(type='constant')
        dv.taper(0.05)

        foru.taper(0.05)
        foru.detrend(type='linear')
        foru.detrend(type='constant')
        foru.taper(0.05)

        correlation_xarray = correlate(foru_xarray,dv_xarray)
        numpy_source_array = convert_xarray_to_np_array(correlation_xarray)
        #numpy_source_array/= np.amax(np.abs(numpy_source_array))

        correlation_target = sig.fftconvolve(foru[0].data,dv[0].data[::-1],mode='full')
        #correlation_target/= np.amax(np.abs(correlation_target))

        plt.figure(figsize=(10,4))
        plt.plot(numpy_source_array,label='1')
        plt.plot(correlation_target,label='2',alpha=0.5)
        plt.legend()
        plt.xlim([30000,40000])
        plt.show()

        np.testing.assert_allclose(numpy_source_array,correlation_target)


    def test_autocorrelation_with_anxcor_data_scheme(self):
        tau=None
        starttime = UTCDateTime("2017-10-01 06:00:00").timestamp
        endtime   = UTCDateTime("2017-10-01 06:10:00").timestamp
        starttime_utc = UTCDateTime("2017-10-01 06:00:00")
        endtime_utc = UTCDateTime("2017-10-01 06:10:00")
        #test this with anxcor functions outside of anxcor
        anxcor_main = build_anxcor(tau)
        foru_data = anxcor_main._get_task('data')(starttime=starttime,
                                                  station='Nodal.1')
        foru_xarray = anxcor_main._station_window_operations(foru_data,starttime=starttime)

        foru_corr  = correlate(foru_xarray,foru_xarray)
        foru_numpy = convert_xarray_to_np_array(foru_corr)

        foru = get_dv().trim(starttime_utc, endtime_utc)
        foru.taper(0.05)
        foru.detrend(type='linear')
        foru.detrend(type='constant')
        foru.taper(0.05)
        correlation_target = sig.fftconvolve(foru[0].data, foru[0].data[::-1], mode='full')

        np.testing.assert_allclose(foru_data[0].data,foru[0].data)


    def test_correlation_with_anxcor_data_scheme(self):
        tau=None
        starttime = UTCDateTime("2017-10-01 06:00:00").timestamp
        endtime   = UTCDateTime("2017-10-01 06:10:00").timestamp
        starttime_utc = UTCDateTime("2017-10-01 06:00:00")
        endtime_utc = UTCDateTime("2017-10-01 06:10:00")
        #test this with anxcor functions outside of anxcor
        anxcor_main = build_anxcor(tau)
        foru_data = anxcor_main._get_task('data')(starttime=starttime,
                                                  station='UU.Broadband')
        dv_data = anxcor_main._get_task('data')(starttime=starttime,
                                                  station='Nodal.1')
        foru_xarray = anxcor_main._station_window_operations(foru_data,starttime=starttime)
        dv_xarray   = anxcor_main._station_window_operations(dv_data, starttime=starttime)

        corr  = correlate(foru_xarray,dv_xarray)
        foru_numpy = convert_xarray_to_np_array(corr)

        foru = get_FORU().trim(starttime_utc, endtime_utc)
        foru.taper(0.05)
        foru.detrend(type='linear')
        foru.detrend(type='constant')
        foru.taper(0.05)

        dv = get_dv().trim(starttime_utc, endtime_utc)
        dv.taper(0.05)
        dv.detrend(type='linear')
        dv.detrend(type='constant')
        dv.taper(0.05)
        correlation_target = sig.fftconvolve(foru[0].data, dv[0].data[::-1], mode='full')

        np.testing.assert_allclose(foru_data[0].data,foru[0].data)


    def test_passband_2_obspy_equivlanet(self):
        overlap = 0.0
        starttime = UTCDateTime("2017-10-01 06:00:00").timestamp
        endtime = UTCDateTime("2017-10-01 06:10:00").timestamp
        anxcor_main = build_anxcor()
        starttime_list = anxcor_main.get_starttimes(starttime, endtime, overlap)

        pass

    def test_passband_3_obspy_equivalent(self):
        overlap = 0.0
        starttime = UTCDateTime("2017-10-01 06:00:00").timestamp
        endtime = UTCDateTime("2017-10-01 06:10:00").timestamp
        anxcor_main = build_anxcor()
        starttime_list = anxcor_main.get_starttimes(starttime, endtime, overlap)
        pass







