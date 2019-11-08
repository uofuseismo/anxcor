

from dask.distributed import Future
import unittest
from obsplus.bank import WaveBank
from obspy.core import Stream, Trace
from anxcor.core import Anxcor
from anxcor.containers import AnxcorDatabase
from os import path
import numpy as np
import pytest
import os

source_dir = 'tests/test_data/test_anxcor_database/test_waveforms_multi_station'
target_dir = 'tests/test_data/test_anxcor_database/test_save_output'

if not path.exists(target_dir):
    print(os.getcwd())
    os.mkdir(target_dir)

class Process:

    def __init__(self):
        pass

    def process(self, trace):
        pass


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

        def create_seed(row):
            network = row['network']
            station = row['station']
            return network + '.' + station

        df['seed'] = df.apply(lambda row: create_seed(row), axis=1)
        unique_stations = df['seed'].unique().tolist()
        return unique_stations

class TestConfig(unittest.TestCase):


    def test_dask_execution(self):
        # created test after observing incorrect time array conversion. Will test again on cluster
        from distributed import Client, LocalCluster
        cluster = LocalCluster()
        c = Client(cluster)
        anxcor = Anxcor()
        anxcor.set_window_length(150)
        starttime = 0.0
        times = [starttime,starttime+60]
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        result = anxcor.process(times, dask_client=c)
        if isinstance(result,Future):
            print('future')
            result = result.result()
        rec_chan = list(result.coords['rec_chan'].values)
        assert 3 == len(rec_chan)