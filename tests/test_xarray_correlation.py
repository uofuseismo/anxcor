import unittest
#from tests.synthetic_trace_factory import create_random_trace, create_sinsoidal_trace_w_decay, create_triangle_trace
from synthetic_trace_factory import create_random_trace, create_sinsoidal_trace_w_decay, create_triangle_trace
from anxcor.xarray_routines import XArrayXCorrelate, XArrayConverter, XArrayRemoveMeanTrend
from anxcor.containers import XArrayStack, AnxcorDatabase
from anxcor.numpyfftfilter import _multiply_in_mat, xarray_crosscorrelate
from anxcor.core import Anxcor
import numpy as np
import xarray as xr
from obspy.core import  UTCDateTime, Trace, Stream, read
from obsplus.bank import WaveBank
import pytest


class WavebankWrapper(AnxcorDatabase):

    def __init__(self, directory):
        super().__init__()
        self.bank = WaveBank(directory)
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
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title('after wavebank')
        for trace in traces:
            plt.plot(trace.data,label=trace.stats.channel)
        plt.legend()
        plt.show()
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

def shift_trace(data,time=1.0,delta=0.1):
    sampling_rate = int(1.0/delta)
    random_data   = np.zeros((data.shape[0],data.shape[1],int(sampling_rate*time)))
    new_data      = np.concatenate((random_data,data),axis=2)[:,:,:data.shape[2]]
    return new_data

def create_example_xarrays():
    e1 = create_sinsoidal_trace_w_decay(decay=0.9, station='h', network='v', channel='e', duration=20)
    n1 = create_sinsoidal_trace_w_decay(decay=0.3, station='h', network='v', channel='n', duration=20)
    z1 = create_sinsoidal_trace_w_decay(decay=0.4, station='h', network='v', channel='z', duration=20)

    e2 = create_sinsoidal_trace_w_decay(decay=0.8, station='k', network='v', channel='e', duration=20)
    n2 = create_sinsoidal_trace_w_decay(decay=0.7, station='k', network='v', channel='n', duration=20)
    z2 = create_sinsoidal_trace_w_decay(decay=0.6, station='k', network='v', channel='z', duration=20)

    syth_trace1 = converter(e1.copy() + z1.copy() + n1.copy())
    syth_trace2 = converter(e2.copy() + z2.copy() + n2.copy())
    return syth_trace1, syth_trace2


def create_example_xarrays_missing_channel():
    e1 = create_sinsoidal_trace_w_decay(decay=0.9, station='h', network='v', channel='e', duration=20)
    z1 = create_sinsoidal_trace_w_decay(decay=0.4, station='h', network='v', channel='z', duration=20)

    e2 = create_sinsoidal_trace_w_decay(decay=0.8, station='k', network='v', channel='e', duration=20)
    n2 = create_sinsoidal_trace_w_decay(decay=0.7, station='k', network='v', channel='n', duration=20)
    z2 = create_sinsoidal_trace_w_decay(decay=0.6, station='k', network='v', channel='z', duration=20)

    syth_trace1 = converter(e1.copy() + z1.copy())
    syth_trace2 = converter(e2.copy() + z2.copy() + n2.copy())
    return syth_trace1, syth_trace2


stacker   = XArrayStack()
converter = XArrayConverter()
class TestCorrelation(unittest.TestCase):


    def test_autocorrelation(self):
        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        synth_trace1 = converter(create_random_trace(station='1',network='v',duration=20))
        correlation = correlator(synth_trace1,synth_trace1)
        zero_target_index = correlation.data.shape[3]//2
        zero_source_index = np.argmax(correlation.data[0,0,0,:])
        assert zero_source_index == zero_target_index,'autocorrelation failed'

    def test_stacking_preserves_metadata(self):
        correlator = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1', network='v', duration=20))

        result_1 = correlator(syth_trace1, syth_trace1)
        result_2 = correlator(syth_trace1, syth_trace1)

        stack = stacker(result_1,result_2)

        attrs = stack.attrs

        assert 'starttime' in attrs.keys(), 'starttime did not persist through stacking'
        assert 'endtime' in attrs.keys(), 'endtime did not persist through stacking'
        assert attrs['stacks'] == 2,'unexpected stack length'
        assert 'operations' in attrs.keys(), 'operations did not persist through stacking'
        assert 'delta' in attrs.keys(), 'delta did not persist through stacking'


    def test_has_starttime(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1',network='v',duration=20))

        keys = correlator(syth_trace1,syth_trace1).attrs.keys()

        assert 'starttime' in keys, 'starttime not preserved through correlation'

    def test_has_endtime(self):

        correlator = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1', network='v', duration=20))

        keys = correlator(syth_trace1, syth_trace1).attrs.keys()

        assert 'endtime' in keys, 'endtime not preserved through correlation'

    def test_has_delta(self):

        correlator = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1', network='v', duration=20))

        keys = correlator(syth_trace1, syth_trace1).attrs.keys()

        assert 'delta' in keys, 'delta not preserved through correlation'

    def test_has_stacks(self):

        correlator = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1', network='v', duration=20))

        keys = correlator(syth_trace1, syth_trace1).attrs.keys()

        assert 'stacks' in keys, 'stacks not preserved through correlation'

    def test_one_stack(self):

        correlator = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1', network='v', duration=20))

        attr = correlator(syth_trace1, syth_trace1).attrs

        assert attr['stacks'] == 1,'stacks assigned improper value'

    def test_autocorrelation_delta_attr(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1',network='v',duration=20))

        correlation = correlator(syth_trace1,syth_trace1)

        assert 'delta' in correlation.attrs.keys(),'did not propagate the delta value'

    def test_autocorrelation_starttime_attr(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1',network='v',duration=20))

        correlation = correlator(syth_trace1,syth_trace1)

        assert 'starttime' in correlation.attrs.keys(),'did not propagate the delta value'

    def test_name(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1',network='v',duration=20))

        correlation = correlator(syth_trace1,syth_trace1)

        assert isinstance(correlation.name,str),'improper name type'

    def test_array_is_real(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1',network='v',duration=20))

        correlation = correlator(syth_trace1,syth_trace1)

        assert correlation.data.dtype == np.float64,'improper data type'

    def test_correlation_length(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter(create_random_trace(station='1',network='v'))
        syth_trace2 = converter(create_random_trace(station='2',network='v'))
        try:
            correlation = correlator(syth_trace1,syth_trace2)
            assert False,'failed to catch exception'
        except Exception:
            assert True, 'failed to catch exception'


    def test_shift_trace(self):
        max_tau_shift = 19.9
        correlator  = XArrayXCorrelate(max_tau_shift=max_tau_shift)
        time_shift  = 4
        synth_trace1 = converter(create_sinsoidal_trace_w_decay(decay=0.6,station='h',network='v',channel='z',duration=20))
        synth_trace2 = converter(create_sinsoidal_trace_w_decay(decay=0.4,station='h',network='w',channel='z',duration=20))
        synth_trace2 = xr.apply_ufunc(shift_trace,synth_trace2,input_core_dims=[['time']],
                                                             output_core_dims=[['time']],
                                                             kwargs={'time': time_shift,
                                                                     'delta':synth_trace2.attrs['delta']},
                                                             keep_attrs=True)
        correlation = correlator(synth_trace1,synth_trace2)
        correlation/= correlation.max()
        time_array  = correlation.coords['time'].values
        max_index   = np.argmax(correlation.data)
        tau_shift   = (time_array[max_index]- np.datetime64(UTCDateTime(0.0).datetime) ) / np.timedelta64(1, 's')
        assert tau_shift == -time_shift,'failed to correctly identify tau shift'

    def test_shift_trace_time_slice(self):
        max_tau_shift = 10.0
        correlator  = XArrayXCorrelate(max_tau_shift=max_tau_shift)
        time_shift  = 4
        synth_trace1 = converter(create_sinsoidal_trace_w_decay(decay=0.6,station='h',network='v',channel='z',duration=20))
        synth_trace2 = converter(create_sinsoidal_trace_w_decay(decay=0.4,station='h',network='w',channel='z',duration=20))
        synth_trace2 = xr.apply_ufunc(shift_trace,synth_trace2,input_core_dims=[['time']],
                                                             output_core_dims=[['time']],
                                                             kwargs={'time': time_shift,
                                                                     'delta':synth_trace2.attrs['delta']},
                                                             keep_attrs=True)
        correlation = correlator(synth_trace1,synth_trace2)
        correlation/= correlation.max()
        time_array  = correlation.coords['time'].values
        max_index   = np.argmax(correlation.data)
        tau_shift   = (time_array[max_index]- np.datetime64(UTCDateTime(0.0).datetime) ) / np.timedelta64(1, 's')
        assert tau_shift == -time_shift,'failed to correctly identify tau shift'

    def test_correct_pair_ee(self):
        max_tau_shift = 19
        correlator = XArrayXCorrelate(max_tau_shift=max_tau_shift)

        synth_trace_1, synth_trace_2 = create_example_xarrays()
        src_chan='e'
        rec_chan='e'

        correlation_source = correlator(synth_trace_1.copy(), synth_trace_2.copy())

        test_correlation   = correlator(synth_trace_1.sel(dict(channel=src_chan)).expand_dims('channel'),
                                        synth_trace_2.sel(dict(channel=rec_chan)).expand_dims('channel'))
        result_1     = correlation_source.loc[src_chan, rec_chan, :, :] - test_correlation
        assert 0 == np.sum(result_1.data)


    def test_correct_pair_zz(self):
        max_tau_shift = 19
        correlator = XArrayXCorrelate(max_tau_shift=max_tau_shift)

        synth_trace_1, synth_trace_2 = create_example_xarrays()
        src_chan = 'z'
        rec_chan = 'z'

        correlation_source = correlator(synth_trace_1.copy(), synth_trace_2.copy())

        test_correlation = correlator(synth_trace_1.sel(dict(channel=src_chan)).expand_dims('channel'),
                                      synth_trace_2.sel(dict(channel=rec_chan)).expand_dims('channel'))
        result_1 = correlation_source.loc[src_chan, rec_chan, :, :] - test_correlation
        assert 0 == np.sum(result_1.data)

    def test_correct_pair_ez(self):
        max_tau_shift = 8
        correlator = XArrayXCorrelate(max_tau_shift=max_tau_shift)

        synth_trace_1, synth_trace_2 = create_example_xarrays()
        src_chan = 'e'
        rec_chan = 'z'

        correlation_source = correlator(synth_trace_1.copy(), synth_trace_2.copy())

        test_correlation = correlator(synth_trace_1.sel(dict(channel=src_chan)).expand_dims('channel'),
                                      synth_trace_2.sel(dict(channel=rec_chan)).expand_dims('channel'))
        result_1 = correlation_source.loc[src_chan, rec_chan, :, :] - test_correlation
        assert 0 == np.sum(result_1.data)

    def test_correct_pair_ze(self):
        max_tau_shift = 8
        correlator = XArrayXCorrelate(max_tau_shift=max_tau_shift)

        synth_trace_1, synth_trace_2 = create_example_xarrays()
        src_chan = 'z'
        rec_chan = 'e'

        correlation_source = correlator(synth_trace_1.copy(), synth_trace_2.copy())

        test_correlation = correlator(synth_trace_1.sel(dict(channel=src_chan)).expand_dims('channel'),
                                      synth_trace_2.sel(dict(channel=rec_chan)).expand_dims('channel'))
        result_1 = correlation_source.loc[src_chan, rec_chan, :, :] - test_correlation
        assert 0 == np.sum(result_1.data)



    def test_correct_pair_ez24(self):
        correlator = XArrayXCorrelate(max_tau_shift=None)
        synth_trace_1, synth_trace_2 = create_example_xarrays_missing_channel()
        src_chan = 'z'
        rec_chan = 'e'

        correlation_source = correlator(synth_trace_1.copy(), synth_trace_2.copy())

        test_correlation = correlator(synth_trace_1.sel(dict(channel=src_chan)).expand_dims('channel'),
                                      synth_trace_2.sel(dict(channel=rec_chan)).expand_dims('channel'))
        result_1 = correlation_source.loc[src_chan, rec_chan, :, :] - test_correlation
        assert 0 == np.sum(result_1.data)

    def test_proper_matrix_order(self):
        one = np.arange(0,4).reshape((4,1))
        two = np.arange(1,4).reshape((3,1))

        source = _multiply_in_mat(one,two,dtype=np.float32)
        target = np.asarray([[0, 0, 0],
                             [1, 2, 3],
                             [2, 4, 6],
                             [3, 6, 9]])
        assert np.sum(source[:,:,0]-target) == 0



    def test_nonetype_in_out(self):
        correlator = XArrayXCorrelate()
        result = correlator(None,None, starttime=0, station=0)
        assert result == None

    def test_autocorrelation_combined(self):
        source_dir = 'tests/test_data/test_anxcor_database/test_auto_correlation/src_auto'
        target_dir = 'tests/test_data/test_anxcor_database/test_auto_correlation/expected_result_corr/target_auto_corr.sac'
        anxcor = Anxcor()
        anxcor.set_window_length(60 * 9.9995)
        anxcor.set_task_kwargs('crosscorrelate',{'max_tau_shift':None})
        anxcor.set_task_kwargs('resample',{'target_rate':20})
        anxcor.add_process(XArrayRemoveMeanTrend())
        bank = WavebankWrapper(source_dir)
        df = bank.bank.get_uptime_df()
        starttime=df['starttime'].min()
        anxcor.add_dataset(bank, 'nodals')
        result      = anxcor.process([starttime])['src:nodals rec:nodals']
        full_time   = result.coords['time'].values
        result_data = result.data.ravel()
        result_data/=max(result_data)

        trace=read(target_dir)[0]
        trace.filter('lowpass', freq=10.0, zerophase=True)
        trace.resample(20.0)
        trace.data/=max(trace.data)



        anxcor.set_task_kwargs('crosscorrelate',{'max_tau_shift':20.0})
        result_slice = anxcor.process([starttime])['src:nodals rec:nodals']
        slice_time   = result_slice.coords['time'].values
        slice_data = result_slice.data.ravel()
        slice_data /= max(slice_data)
        assert np.cumsum(trace.data - result_data.ravel())[-1]==pytest.approx(0,abs=1e-2)




if __name__ == '__main__':
    unittest.main()
