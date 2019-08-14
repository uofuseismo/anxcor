import unittest
from obsplus.bank import WaveBank
from obspy.core import Stream, Trace
from anxcor.core import Anxcor
from anxcor.containers import  AnxcorDatabase
import numpy as np

source_dir = 'tests/test_data/test_anxcor_database/test_waveforms_multi_station'
target_dir = 'test_data/test_anxcor_database/test_save_output'

starttime_stamp = 0+1
endtime_stamp   = 5*2*60+1

def get_ancor_set():
    bank = WaveBank(source_dir)
    return bank


class WavebankWrapper(AnxcorDatabase):

    def __init__(self, directory):
        super().__init__()
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
        df = self.bank.get_availability_df()

        def create_seed(row):
            network = row['network']
            station = row['station']
            return network + '.' + station

        df['seed'] = df.apply(lambda row: create_seed(row), axis=1)
        unique_stations = df['seed'].unique().tolist()
        return unique_stations

class WavebankWrapperWLatLons(AnxcorDatabase):

    def __init__(self, directory):
        super().__init__()
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
                      'network': trace.stats.network
                      }
            trace = Trace(data,header=header)
            trace.stats.coordinates={'longitude': -117, 'latitude': 35}
            traces.append(trace)
        return Stream(traces=traces)

    def get_stations(self):
        df = self.bank.get_availability_df()

        def create_seed(row):
            network = row['network']
            station = row['station']
            return network + '.' + station

        df['seed'] = df.apply(lambda row: create_seed(row), axis=1)
        unique_stations = df['seed'].unique().tolist()
        return unique_stations


class TestMetadataInCombine(unittest.TestCase):


    def test_stacking_preserves_pair_key(self):

        anxcor = Anxcor()
        anxcor.set_window_length(100)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 100, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        result = anxcor.process(times)
        assert 'src:AX.1rec:AX.2' in result.attrs.keys()

    def test_metadata_with_latlons(self):
        anxcor = Anxcor()
        anxcor.set_window_length(100)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 100, 0.5)
        bank = WavebankWrapperWLatLons(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        result = anxcor.process(times)
        pair_dict = result.attrs['src:AX.1rec:AX.1']
        assert 'location' in pair_dict.keys()

    def test_output_dataset_format(self):
        anxcor = Anxcor()
        anxcor.set_window_length(100)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 100, 0.5)
        bank = WavebankWrapperWLatLons(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        result = anxcor.process(times)
        len_variables = len(list(result.data_vars))
        #
        assert len_variables == 1,'too many variables added to dataset'

    def test_output_dimension_lengths(self):
        anxcor = Anxcor()
        anxcor.set_window_length(100)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 100, 0.5)
        bank = WavebankWrapperWLatLons(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        result = anxcor.process(times)
        len_pair = len(list(result.coords['pair'].values))
        len_src = len(list(result.coords['src_chan'].values))
        len_rec = len(list(result.coords['rec_chan'].values))
        #
        assert len_pair == 6,'not enough pairs retained'
        assert len_src == 3, 'not enough source channels retained'
        assert len_rec == 3, 'not enough receiver channels retained'




if __name__ == '__main__':
    unittest.main()