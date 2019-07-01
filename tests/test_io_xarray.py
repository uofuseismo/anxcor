import unittest
from obsplus.bank import WaveBank
from obspy.core import Stream, Trace
from utils import _clean_files_in_dir, _how_many_fmt
from anxcor.core import Anxcor, AnxcorDatabase
from anxcor.xarray_routines import XArrayTemporalNorm, XArrayWhiten
import numpy as np
import xarray as xr

source_dir = 'test_data/test_ancor_bank/test_waveforms_multi_station'
target_dir = 'test_data/test_ancor_bank/test_save_output'

starttime_stamp = 1481761092.0 + 3600 * 24


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


class TestIntegratedIOOps(unittest.TestCase):


    def test_single_execution(self):
        # stations 21, & 22
        # 3 windows say
        #
        anxcor = Anxcor(window_length=3600)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 3600, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        result = anxcor.process(times)

        assert isinstance(result, xr.Dataset)


    def test_read_xconvert(self):
        anxcor = Anxcor(window_length=3600)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 3600, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        anxcor.save_at_task(target_dir,'xconvert')
        result = anxcor.process(times)
        anxcor = Anxcor(3600)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        anxcor.load_at_task(target_dir, 'xconvert')
        result = anxcor.process(times)
        _clean_files_in_dir(target_dir)
        assert isinstance(result,xr.Dataset)

    def test_write_xconvert(self):
        anxcor = Anxcor(window_length=3600)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 3600, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        anxcor.save_at_task(target_dir, 'xconvert')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 7 == how_many_nc

    def test_write_resample(self):
        anxcor = Anxcor(window_length=3600)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 3600, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscrosscorrelate', dict(dummy_task=True))
        anxcor.save_at_task(target_dir, 'resample')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 7 == how_many_nc

    def test_read_resample(self):
        anxcor = Anxcor(window_length=3600)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 3600, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        anxcor.save_at_task(target_dir,'resample')
        result = anxcor.process(times)
        anxcor = Anxcor(window_length=3600)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        anxcor.load_at_task(target_dir, 'resample')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 7 == how_many_nc

    def test_write_correlate(self):
        anxcor = Anxcor(window_length=3600)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 3600, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        anxcor.save_at_task(target_dir,'crosscorrelate')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 10 == how_many_nc

    def test_read_correlate(self):
        anxcor = Anxcor(window_length=3600)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 3600, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        anxcor.save_at_task(target_dir,'crosscorrelate')
        result = anxcor.process(times)
        anxcor = Anxcor(3600)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        anxcor.load_at_task(target_dir, 'crosscorrelate')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 10 == how_many_nc

    def test_write_stack(self):
        anxcor = Anxcor(window_length=3600)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 3600, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        anxcor.save_at_task(target_dir, 'stack')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 9 == how_many_nc

    def test_read_stack(self):
        anxcor = Anxcor(window_length=3600)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 3600, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        anxcor.save_at_task(target_dir, 'stack')
        result = anxcor.process(times)
        anxcor = Anxcor(window_length=3600)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 3600, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        anxcor.load_at_task(target_dir,'stack')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 9 == how_many_nc

    def test_write_combine(self):
        anxcor = Anxcor(window_length=3600)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 3600, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate',dict(dummy_task=True))
        anxcor.save_at_task(target_dir, 'combine')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 2 == how_many_nc

    def test_read_combine(self):
        anxcor = Anxcor(window_length=3600)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 4 * 3600, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))

        anxcor.save_at_task(target_dir, 'combine')
        result = anxcor.process(times)
        anxcor = Anxcor(window_length=3600)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate',dict(dummy_task=True))
        anxcor.load_at_task(target_dir,'combine')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 2 == how_many_nc

    def test_write_tempnorm(self):
        anxcor = Anxcor(window_length=3600)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 3600, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        anxcor.add_process(XArrayTemporalNorm(time_window=5.0, lower_frequency=0.02))
        anxcor.save_at_process(target_dir,'temp_norm')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 7 == how_many_nc

    def test_read_tempnorm(self):
        anxcor = Anxcor(window_length=3600)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 3600, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        anxcor.add_process(XArrayTemporalNorm(time_window=5.0, lower_frequency=0.02))
        anxcor.save_at_process(target_dir,'temp_norm')
        result = anxcor.process(times)
        anxcor = Anxcor(3600)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        anxcor.add_process(XArrayTemporalNorm(time_window=5.0, lower_frequency=0.02))
        anxcor.load_at_process(target_dir,'temp_norm')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 7 == how_many_nc

    def test_write_whitening(self):
        anxcor = Anxcor(window_length=3600)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 3600, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        anxcor.add_process(XArrayWhiten())
        anxcor.save_at_process(target_dir,'whiten')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 7 == how_many_nc


    def test_read_whitening(self):
        anxcor = Anxcor(window_length=3600)
        times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 2 * 3600, 0.5)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        anxcor.add_process(XArrayWhiten())
        anxcor.save_at_process(target_dir, 'whiten')
        result = anxcor.process(times)
        anxcor = Anxcor(3600)
        bank = WavebankWrapper(source_dir)
        anxcor.add_dataset(bank, 'nodals')
        anxcor.set_task_kwargs('crosscorrelate', dict(dummy_task=True))
        anxcor.add_process(XArrayWhiten())
        anxcor.load_at_process(target_dir, 'whiten')
        result = anxcor.process(times)
        how_many_nc = _how_many_fmt(target_dir, format='.nc')
        _clean_files_in_dir(target_dir)
        assert 7 == how_many_nc


if __name__ == '__main__':
    unittest.main()