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

source_dir = 'test_data/test_anxcor_database/test_stacking_data/NOA_test_0.1_0.2'

class WavebankWrapper(AnxcorDatabase):

    def __init__(self, directory):
        super().__init__()
        self.bank = WaveBank(directory)
        self.bank.update_index()

    def get_waveforms(self, **kwargs):
        stream =  self.bank.get_waveforms(**kwargs)
        print(stream)
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

        def create_seed(row):
            network = row['network']
            station = row['station']
            return network + '.' + station

        df['seed'] = df.apply(lambda row: create_seed(row), axis=1)
        unique_stations = df['seed'].unique().tolist()
        return unique_stations


class TestIntegratedIOOps(unittest.TestCase):


    def test_single_execution(self):
        anxcorWavebank = WavebankWrapper(source_dir)
        min_starttime = anxcorWavebank.bank.get_availability_df()['starttime'].min()
        endtime = anxcorWavebank.bank.get_availability_df()['endtime'].max()
        starttime = int(min_starttime + 1)
        endtime = int(endtime)
        overlap = 0.5
        window_length = 3 * 60
        resample_kwargs = dict(taper=0.05, target_rate=20.0)
        correlate_kwargs = dict(taper=0.01, max_tau_shift=60)
        anxcor_main = Anxcor()
        anxcor_main.set_window_length(window_length)
        anxcor_main.add_dataset(anxcorWavebank, 'test')
        anxcor_main.set_task_kwargs('resample', resample_kwargs)
        anxcor_main.set_task_kwargs('crosscorrelate', correlate_kwargs)
        starttime_list = anxcor_main.get_starttimes(starttime, starttime + window_length, overlap)
        result = anxcor_main.process(starttime_list)
        assert len(result['src:test rec:test'].coords['rec_chan'].values)==3
        assert len(result['src:test rec:test'].coords['src_chan'].values)==3
        assert not np.isnan(result['src:test rec:test']).any()
