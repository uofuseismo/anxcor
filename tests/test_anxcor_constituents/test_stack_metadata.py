import unittest
from obsplus.bank import WaveBank
from obspy.core import Stream, Trace
from utils import _clean_files_in_dir, _how_many_fmt
from anxcor.core import Anxcor
from anxcor.xarray_routines import XArrayTemporalNorm, XArrayWhiten
import numpy as np
import xarray as xr

source_dir = '../test_data/test_ancor_bank/test_waveforms_multi_station'
target_dir = '../test_data/test_ancor_bank/test_save_output'

starttime_stamp = 1481761092.0 + 3600 * 24


class TestProcess:

    def __init__(self):
        pass

    def process(self, trace):
        pass


def get_ancor_set():
    bank = WaveBank(source_dir)
    return bank


class WavebankWrapper:

    def __init__(self, directory):
        self.bank = WaveBank(directory)
        import warnings
        warnings.filterwarnings("ignore")

    def get_waveforms(self, **kwargs):
        stream =  self.bank.get_waveforms(**kwargs)
        traces = []
        for trace in stream:
            data = trace.data[:-1]
            header = {'delta':np.floor(trace.stats.delta*1000)/1000.0,
                      'station': trace.stats.station,
                      'starttime':trace.stats.starttime,
                      'channel': trace.stats.channel,
                      'network': trace.stats.network}
            traces.append(Trace(data,header=header))
        return Stream(traces=traces)

    def get_stations(self):
        df = self.bank.get_uptime_df()

        def create_seed(row):
            network = row['network']
            station = row['station']
            return network + '.' + station

        df['seed'] = df.apply(lambda row: create_seed(row), axis=1)
        unique_stations = df['seed'].unique().tolist()
        return unique_stations

class WavebankWrapperWLatLons:

    def __init__(self, directory):
        self.bank = WaveBank(directory)
        import warnings
        warnings.filterwarnings("ignore")

    def get_waveforms(self, **kwargs):
        stream =  self.bank.get_waveforms(**kwargs)
        traces = []
        for trace in stream:
            data = trace.data[:-1]
            header = {'delta':np.floor(trace.stats.delta*1000)/1000.0,
                      'station': trace.stats.station,
                      'starttime':trace.stats.starttime,
                      'channel': trace.stats.channel,
                      'network': trace.stats.network,
                      'latitude': trace.stats.sac['stla'],
                      'longitude': -trace.stats.sac['stlo'],}
            traces.append(Trace(data,header=header))
        return Stream(traces=traces)

    def get_stations(self):
        df = self.bank.get_uptime_df()

        def create_seed(row):
            network = row['network']
            station = row['station']
            return network + '.' + station

        df['seed'] = df.apply(lambda row: create_seed(row), axis=1)
        unique_stations = df['seed'].unique().tolist()
        return unique_stations


class TestMetadataInCombine(unittest.TestCase):


    def test_stacking_preserves_pair_key(self):

        anxcor = Anxcor(3600, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_parameters('correlate', dict(dummy_task=True))
        result = anxcor.process(starttime=starttime_stamp, endtime=starttime_stamp + 2 * 3600)
        #
        self.assertTrue('src:FG.21rec:FG.22' in result.attrs.keys())

    def test_metadata_with_latlons(self):
        anxcor = Anxcor(3600, 0.5)
        bank = WavebankWrapperWLatLons(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_parameters('correlate', dict(dummy_task=True))
        result = anxcor.process(starttime=starttime_stamp, endtime=starttime_stamp + 2 * 3600)
        print(result.variables.values)
        #
        self.assertTrue('location' in result.attrs['src:FG.21rec:FG.22'].keys())

    def test_output_dataset_format(self):
        anxcor = Anxcor(3600, 0.5)
        bank = WavebankWrapperWLatLons(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_parameters('correlate', dict(dummy_task=True))
        result = anxcor.process(starttime=starttime_stamp, endtime=starttime_stamp + 2 * 3600)
        len_variables = len(list(result.data_vars))
        #
        self.assertEqual(len_variables,1,'too many variables added to dataset')

    def test_output_dimension_lengths(self):
        anxcor = Anxcor(3600, 0.5)
        bank = WavebankWrapperWLatLons(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_parameters('correlate', dict(dummy_task=True))
        result = anxcor.process(starttime=starttime_stamp, endtime=starttime_stamp + 2 * 3600)
        len_pair = len(list(result.coords['pair'].values))
        len_src = len(list(result.coords['src_chan'].values))
        len_rec = len(list(result.coords['rec_chan'].values))
        #
        self.assertEqual(len_pair,3,'not enough pairs retained')
        self.assertEqual(len_src, 3, 'not enough source channels retained')
        self.assertEqual(len_rec, 3, 'not enough receiver channels retained')




if __name__ == '__main__':
    unittest.main()