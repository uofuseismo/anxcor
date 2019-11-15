import unittest
#from tests.synthetic_trace_factory import create_random_trace, create_sinsoidal_trace_w_decay, create_triangle_trace
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
    broadband_data_dir               = 'tests/test_data/correlation_integration_testing/FORU'
    broadband_station_location_file  = 'tests/test_data/correlation_integration_testing/bbstationlist.txt'
    nodal_data_dir               =     'tests/test_data/correlation_integration_testing/DV'
    nodal_station_location_file  =     'tests/test_data/correlation_integration_testing/DV_stationlst.lst'

    broadband_database = DWellsDecimatedReader(broadband_data_dir, broadband_station_location_file)
    nodal_database     = DWellsDecimatedReader(nodal_data_dir,     nodal_station_location_file,extension='d')
    window_length = 10*60.0
    #include_stations = ['DV.{}'.format(x) for x in range(1,10)]
    include_stations = ['UU.FORU','DV.1','DV.2']

    taper_ratio     = 0.05
    target_rate     = 50.0
    correlate_kwargs= dict(max_tau_shift=tau,taper=taper_ratio)
    resample_kwargs = dict(target_rate=target_rate,lowpass=False)

    anxcor_main = Anxcor()
    anxcor_main.set_window_length(window_length)
    anxcor_main.set_must_only_include_station_pairs(include_stations)
    anxcor_main.add_dataset(broadband_database,'BB')
    anxcor_main.add_dataset(nodal_database, 'DV')
    anxcor_main.add_process(XArrayTaper(taper=taper_ratio))
    anxcor_main.add_process(XArrayRemoveMeanTrend())
    anxcor_main.add_process(XArrayTaper(taper=taper_ratio))
    anxcor_main.set_task_kwargs('crosscorrelate',correlate_kwargs)
    anxcor_main.set_task('post-correlate',XArrayCustomComponentNormalizer())
    return anxcor_main


converter = XArrayConverter()
correlate = XArrayXCorrelate(max_tau_shift=None)
taper     = XArrayTaper(taper=0.05,type='hann')
def convert_xarray_to_np_array(xarray):
    xarray_sub = xarray.loc[dict(src_chan='z',rec_chan='z')].squeeze()
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
        stream[0].data = np.ones(stream[0].data.shape)
        target_stream = stream.copy()
        source_xarray = converter(stream)
        source_xarray = taper(source_xarray)
        target_stream.taper(0.05,type='hann')

        source = source_xarray.data.squeeze()
        source /=np.amax(np.abs(source))
        target = target_stream[0].data
        target /=np.amax(np.abs(target))

        differece = target-source
        differece/=np.amax(np.abs(differece))
        plt.figure()
        plt.plot(source,label='xarray')
        plt.plot(target,label='obspy')
        plt.plot(differece,label='obspy - xarray')
        #plt.xlim([0,200])
        #plt.ylim([-0.25,-.15])
        plt.legend()
        plt.show()
        np.testing.assert_allclose(source_xarray.data.squeeze(),target_stream[0].data)




