import unittest
import pytest
import os
from os import path
from anxcor.containers import AnxcorDatabase
from anxcor.utils import _clean_files_in_dir, _how_many_fmt
from anxcor.core import Anxcor
from anxcor.xarray_routines import XArrayBandpass
from obspy.core import Stream, Trace
import anxcor.utils as utils
from obsplus import WaveBank
import xarray as xr
import numpy as np
from anxcor.xarray_routines import XArrayTemporalNorm
import json


source_dir = 'tests/test_data/test_anxcor_database/test_waveforms_multi_station'
target_dir = 'tests/test_data/test_anxcor_database/test_save_output'

starttime_stamp = 0
endtime_stamp   = 5*2*60 # 10 minutes

if not path.exists(target_dir):
    print(os.getcwd())
    os.mkdir(target_dir)


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
        uptime = self.bank.get_uptime_df()

        def create_seed(row):
            network = row['network']
            station = row['station']
            return network + '.' + station

        df['seed'] = df.apply(lambda row: create_seed(row), axis=1)
        unique_stations = df['seed'].unique().tolist()
        return unique_stations



class TestConfig(unittest.TestCase):

    def tearDown(self):
        _clean_files_in_dir(target_dir)

    def test_bandpass_result(self):
        bp = XArrayBandpass(freqmin=0.1, freqmax=10.0)
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp, endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        result = anxcor.process(times)
        bp_result = bp(result)
        assert isinstance(bp_result,xr.Dataset)
