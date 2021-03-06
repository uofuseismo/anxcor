import unittest
from obsplus.bank import WaveBank
from obspy.core import Stream, Trace, read
from anxcor.core import Anxcor
from anxcor.containers import  AnxcorDatabase
from anxcor.xarray_routines import XArrayConverter
import numpy as np

import pytest
source_dir = 'tests/test_data/test_anxcor_database/test_waveforms_multi_station'
target_dir = 'test_data/test_anxcor_database/test_save_output'

starttime_stamp = 0


def get_ancor_set():
    bank = WaveBank(source_dir)
    return bank


class WavebankWrapper(AnxcorDatabase):

    def __init__(self, directory):
        super().__init__()
        self.bank = WaveBank(directory,name_structure='{network}.{station}.{channel}.{time}')
        self.bank.update_index()

    def get_waveforms(self, **kwargs):
        stream =  self.bank.get_waveforms(**kwargs)
        traces = []
        for trace in stream:
            data = trace.data[:-1]
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


class TestObspyUtilFunction(unittest.TestCase):


    def test_single_execution(self):
        # stations 21, & 22
        # 3 windows say
        #
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        result  = anxcor.process([starttime_stamp])
        streams = anxcor.xarray_to_obspy(result)
        assert len(streams) == 25,'not enough traces retained!'


    def test_basic_read_conversion_functionality(self):
        stream = read()
        xarray = XArrayConverter()(stream)
        assert True


    @pytest.mark.skip('to be implemented')
    def test_rotation(self):
        # stations 21, & 22
        # 3 windows say
        #
        anxcor = Anxcor(3600)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        result = anxcor.process([starttime_stamp])
        rotated_result = anxcor.align_station_pairs(result)
        streams = anxcor.xarray_to_obspy(result)
        assert len(streams) == 9,'not enough traces retained!'




if __name__ == '__main__':
    unittest.main()