import unittest
from anxcor.xarray_routines import  XArrayProcessor
import os
from anxcor.containers import  AnxcorDatabase
import glob
from anxcor.core import Anxcor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from obspy.core import  UTCDateTime, Trace, Stream, read

def get_dv():
    stream =read('tests/test_data/correlation_integration_testing/Nodal/1/20171001000022180.1.EHZ.DV.sac.d')
    stats = stream[0].stats
    stats.delta = 0.02
    stats.channel='z'

    return Stream(traces=[Trace(stream[0].data,header=stats)])

def get_FORU():
    stream = read('tests/test_data/correlation_integration_testing/Broadband/FORU/20171001.FORU.EHZ.sac')
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
                self.cast_to_simple_channel(trace)
                traces.append(trace)
        new_stream = Stream(traces=traces)
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


def build_anxcor(tau,interpolation_method='nearest'):
    broadband_data_dir               = 'tests/test_data/correlation_integration_testing/Broadband'
    broadband_station_location_file  = 'tests/test_data/correlation_integration_testing/broadband_stationlist.txt'
    nodal_data_dir               =     'tests/test_data/correlation_integration_testing/Nodal'
    nodal_station_location_file  =     'tests/test_data/correlation_integration_testing/nodal_stationlist.txt'

    broadband_database = DWellsDecimatedReader(broadband_data_dir, broadband_station_location_file)
    nodal_database     = DWellsDecimatedReader(nodal_data_dir,     nodal_station_location_file,extension='d')
    window_length = 10*60.0
    #include_stations = ['Nodal.{}'.format(x) for x in range(1,10)]
    include_stations = ['UU.FORU','DV.1']

    taper_ratio     = 0.05
    target_rate     = 50.0
    correlate_kwargs= dict(max_tau_shift=tau,taper=taper_ratio)
    resample_kwargs = dict(target_rate=target_rate,lowpass=False)

    anxcor_main = Anxcor(interp_method=interpolation_method)
    anxcor_main.set_window_length(window_length)
    anxcor_main.set_must_only_include_station_pairs(include_stations)
    anxcor_main.add_dataset(broadband_database,'BB')
    anxcor_main.add_dataset(nodal_database, 'Nodal')
    anxcor_main.set_task_kwargs('crosscorrelate',correlate_kwargs)
   # anxcor_main.set_task('post-correlate',XArrayCustomComponentNormalizer())
    return anxcor_main


class TestCorrelation(unittest.TestCase):


    def test_conversion_correlation(self):
        starttime = UTCDateTime("2017-10-01 06:00:00").timestamp

        starttime_utc = UTCDateTime("2017-10-01 06:00:00")
        endtime_utc = UTCDateTime("2017-10-01 06:10:00")
        anxcor_main = build_anxcor(None,interpolation_method='nearest')
        stream_source = anxcor_main._get_task('data')(starttime=starttime, station='UU.FORU')
        stream_target = get_FORU().trim(starttime_utc,endtime_utc)
        source_times = np.linspace(stream_source[0].stats.starttime.timestamp, stream_source[0].stats.endtime.timestamp,
                                   num=stream_source[0].stats.npts)
        target_times = np.linspace(stream_target[0].stats.starttime.timestamp, stream_target[0].stats.endtime.timestamp,
                                   num=stream_target[0].stats.npts)

        difference = stream_source[0].data - stream_target[0].data
        print('mean {} median {} max {} std: {}'.format(np.mean(np.abs(difference)),np.median(np.abs(difference)),
                                                np.amax(np.abs(difference)),np.std(difference)))
        print(difference)
        difference = 1e-7*difference/np.amax(np.abs(difference))
        plt.figure(figsize=(7, 5))
        plt.title('interpolation approach in data FORU')
        plt.plot(target_times, stream_source[0].data, label='anxcor data load, norm, & trim')
        plt.plot(target_times, stream_target[0].data, label='obspy data load, norm, & trim')
        plt.legend()
        plt.xlim([stream_source[0].stats.starttime.timestamp,stream_target[0].stats.endtime.timestamp])
        plt.xlabel('time')
        plt.show()
        np.testing.assert_allclose(stream_source[0].data,stream_target[0])


