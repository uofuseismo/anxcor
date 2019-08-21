

from dask.distributed import Future
import unittest
from obsplus.bank import WaveBank
from obspy.core import Stream, Trace
from anxcor.utils import _clean_files_in_dir, _how_many_fmt
import anxcor.utils as utils
from anxcor.core import Anxcor
from anxcor.containers import AnxcorDatabase
from anxcor.xarray_routines import XArrayTemporalNorm, XArrayWhiten
import xarray as xr
from os import path
import numpy as np
import pytest
import os

source_dir = 'tests/test_data/test_anxcor_database/test_waveforms_obsplus_test_case/1'
target_dir = 'tests/test_data/test_anxcor_database/test_save_output'

if not path.exists(target_dir):
    print(os.getcwd())
    os.mkdir(target_dir)

class TestProcess:

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

    @pytest.mark.skip('not needed')
    def test_dask_execution(self):
        # created test after observing incorrect time array conversion. Will test again on cluster
        from distributed import Client, LocalCluster
        cluster = LocalCluster(n_workers=1, threads_per_worker=4)
        c = Client(cluster)
        anxcor = Anxcor()
        anxcor.set_window_length(60*15)
        starttime = 1482368213.0
        times = [starttime]
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        result = anxcor.process(times, dask_client=None)
        if isinstance(result,Future):
            result = result.result()
        rec_chan = list(result.coords['rec_chan'].values)
        assert 3 == len(rec_chan)