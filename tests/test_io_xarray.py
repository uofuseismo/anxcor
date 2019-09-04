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


class TestIntegratedIOOps(unittest.TestCase):

    def tearDown(self):
        _clean_files_in_dir(target_dir)


    def test_single_execution(self):
        # stations 21, & 22
        # 3 windows say
        #
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        result = anxcor.process(times)
        pairs = list(result.coords['pair'].values)
        assert len(pairs) == 6

    #@pytest.mark.skip('skipping dask debugging')
    def test_dask_execution(self):

        from distributed import Client, LocalCluster
        cluster = LocalCluster(n_workers=1, threads_per_worker=1)
        c = Client(cluster)
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        result = anxcor.process(times,dask_client=c)
        result = result.result()
        pairs  = list(result.coords['pair'].values)
        c.close()
        cluster.close()
        assert 6 ==len(pairs)

    def test_pair_preservation(self):
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        result = anxcor.process(times)
        pairs  = list(result.coords['pair'].values)
        assert 6 ==len(pairs)

    #@pytest.mark.skip('skipping dask debugging')
    def test_dask_read_combine(self):

        from distributed import Client, LocalCluster
        from dask.distributed import wait
        cluster = LocalCluster(n_workers=1, threads_per_worker=1)
        c = Client(cluster)
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp, endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')

        anxcor.save_at_task(target_dir, 'combine')
        result = anxcor.process(times,dask_client=c)
        wait(result)
        result.result()
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.load_at_task(target_dir, 'combine')
        result = anxcor.process(times,dask_client=c)
        result.result()
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        c.close()
        cluster.close()
        assert 48 == how_many_nc

    def test_dask_read_combine_instastack(self):

        from distributed import Client, LocalCluster
        from dask.distributed import wait
        cluster = LocalCluster(n_workers=1, threads_per_worker=1)
        c = Client(cluster)
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp, endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')

        anxcor.save_at_task(target_dir, 'combine')
        result = anxcor.process(times,dask_client=c,stack_immediately=True)

        result.result()
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.load_at_task(target_dir, 'combine')
        result = anxcor.process(times,dask_client=c,stack_immediately=True)
        result.result()
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        c.close()
        cluster.close()
        assert 48 == how_many_nc

    def test_write_tempnorm(self):
        from distributed import Client, LocalCluster
        from dask.distributed import wait
        cluster = LocalCluster(n_workers=1, threads_per_worker=1)
        c = Client(cluster)
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.add_process(XArrayTemporalNorm(time_window=5.0, lower_frequency=0.02))
        anxcor.save_at_process(target_dir,'temp_norm')
        result = anxcor.process(times,dask_client=c)
        result.result()
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 20 == how_many_nc


    def test_read_xconvert(self):
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.save_at_task(target_dir,'xconvert')
        result = anxcor.process(times)
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.load_at_task(target_dir, 'xconvert')
        result = anxcor.process(times)
        _clean_files_in_dir(target_dir)
        assert isinstance(result,xr.Dataset)


    def test_write_xconvert(self):
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.save_at_task(target_dir, 'xconvert')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 20 == how_many_nc


    def test_write_resample(self):
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.save_at_task(target_dir, 'resample')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 20 == how_many_nc

    def test_read_resample(self):
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.save_at_task(target_dir,'resample')
        result = anxcor.process(times)
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.load_at_task(target_dir, 'resample')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 20 == how_many_nc

    def test_write_correlate(self):
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.save_at_task(target_dir,'crosscorrelate')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 36 == how_many_nc

    def test_read_correlate(self):
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.save_at_task(target_dir,'crosscorrelate')
        result = anxcor.process(times)

        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.load_at_task(target_dir, 'crosscorrelate')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 36 == how_many_nc


    def test_write_stack(self):
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.save_at_task(target_dir, 'stack')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 9 == how_many_nc

    def test_read_stack(self):
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.save_at_task(target_dir, 'stack')
        result = anxcor.process(times)
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.load_at_task(target_dir,'stack')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 9 == how_many_nc

    def test_read_every_stack(self):
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.save_at_task(target_dir, 'stack')
        result = anxcor.process(times)
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.load_at_task(target_dir,'stack')
        result = anxcor.process(times,stack_immediately=True)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 9 == how_many_nc


    def test_write_combine(self):
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.save_at_task(target_dir, 'combine')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 48 == how_many_nc


    def test_read_combine(self):
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')

        anxcor.save_at_task(target_dir, 'combine')
        result = anxcor.process(times)
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.load_at_task(target_dir,'combine')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 48 == how_many_nc


    def test_write_tempnorm(self):
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.add_process(XArrayTemporalNorm(time_window=5.0, lower_frequency=0.02))
        anxcor.save_at_process(target_dir,'temp_norm')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 20 == how_many_nc


    def test_read_tempnorm(self):
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.add_process(XArrayTemporalNorm(time_window=5.0, lower_frequency=0.02))
        anxcor.save_at_process(target_dir,'temp_norm')
        result = anxcor.process(times)
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.add_process(XArrayTemporalNorm(time_window=5.0, lower_frequency=0.02))
        anxcor.load_at_process(target_dir,'temp_norm')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 20 == how_many_nc


    def test_write_whitening(self):
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.add_process(XArrayWhiten())
        anxcor.save_at_process(target_dir,'whiten')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 20 == how_many_nc


    def test_read_whitening(self):
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        times = anxcor.get_starttimes(starttime_stamp,endtime_stamp, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.add_process(XArrayWhiten())
        anxcor.save_at_process(target_dir, 'whiten')
        result = anxcor.process(times)
        anxcor = Anxcor()
        anxcor.set_window_length(120.0)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.add_process(XArrayWhiten())
        anxcor.load_at_process(target_dir, 'whiten')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 20 == how_many_nc


if __name__ == '__main__':
    unittest.main()