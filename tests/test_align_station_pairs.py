import unittest
from obsplus.bank import WaveBank
from obspy.core import Stream, Trace
from anxcor.utils import _clean_files_in_dir, _how_many_fmt
import anxcor.utils as utils
from anxcor.core import Anxcor
from anxcor.containers import AnxcorDatabase
from anxcor.xarray_routines import XArrayTemporalNorm, XArrayWhiten, XArrayResample
import xarray as xr
from os import path
import numpy as np
import pytest
import os
import pandas as pd

source_dir = 'tests/test_data/test_anxcor_database/test_waveforms_multi_station'
target_dir = 'tests/test_data/test_anxcor_database/test_save_output'

starttime_stamp = 0
endtime_stamp   = 5*2*60 # 10 minutes

if not path.exists(target_dir):
    print(os.getcwd())
    os.mkdir(target_dir)


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
            if 'z' in trace.stats.channel.lower():
                channel = 'z'
            elif 'n' in trace.stats.channel.lower():
                channel='n'
            elif 'e' in trace.stats.channel.lower():
                channel='e'
            header = {'delta':   trace.stats.delta,
                      'station': trace.stats.station,
                      'starttime':trace.stats.starttime,
                      'channel': channel,
                      'network': trace.stats.network}
            trace = Trace(data,header=header)
            coordinate_dict={'latitude':0,'longitude':1,'elevation':0}
            if trace.stats.station=='1':
                coordinate_dict['latitude']  = 50
                coordinate_dict['longitude'] = 50
            elif trace.stats.station=='2':
                coordinate_dict['latitude']  = 0
                coordinate_dict['longitude'] = 50
            else:
                coordinate_dict['latitude']  = 50
                coordinate_dict['longitude'] = 0
            trace.stats.coordinates=coordinate_dict
            traces.append(trace)
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


class TestAlignOps(unittest.TestCase):

    def tearDown(self):
        _clean_files_in_dir(target_dir)


    def test_single_execution(self):
        # stations 21, & 22
        # 3 windows say
        #
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank  = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        result = anxcor.process(times)
        pairs  = list(result.coords['rec'].values) + list(result.coords['src'].values)
        anxcor.align_station_pairs(result)
        assert len(pairs) == 6

    def test_align_algo(self):
        # z, R, T
        se_one_trace = np.asarray([1e-10, -0.5,np.sqrt(3)/2]).T
        nw_one_trace = np.asarray([1e-10, 0.5,-np.sqrt(3)/2])
        zne_matrix = np.outer(se_one_trace,nw_one_trace)
        result = np.stack([zne_matrix]*100,axis=2)
        result = np.reshape(result,(3,3,100,1,1))
        anxcor = Anxcor()
        dataarray = xr.DataArray(result,coords={
            'rec' : ['test_rec'],
            'src' : ['test_src'],
            'src_chan':['Z','N','E'],
            'rec_chan':['Z','N','E'],
            'time':np.arange(0,100)},
            dims=['src_chan','rec_chan','time','rec','src'],
            name='test_array')
        data = dataarray.data
        dataset  = dataarray.to_dataset()
        df = pd.DataFrame([{'src':'test_src', 'rec':'test_rec',
                                             'src_latitude':0,
                                             'src_longitude':0,
                                             'rec_latitude':-1/200,
                                             'rec_longitude': np.sqrt(3)/200}])
        dataset.attrs['df'] = df

        alined_set = anxcor.align_station_pairs(dataset)
        assert False


    def test_dask_align(self):
        assert False