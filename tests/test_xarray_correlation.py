import unittest
from tests.synthetic_trace_factory import create_random_trace, create_sinsoidal_trace_w_decay, create_triangle_trace
#from synthetic_trace_factory import create_random_trace, create_sinsoidal_trace_w_decay, create_triangle_trace
from anxcor.xarray_routines import XArrayXCorrelate, XArrayConverter
from anxcor.containers import XArrayStack
from anxcor.filters import _multiply_in_mat, xarray_crosscorrelate
import numpy as np
import xarray as xr
from obspy.core import  UTCDateTime
import matplotlib.pyplot as plt

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
        time_shift  = 2
        synth_trace2 = converter(create_sinsoidal_trace_w_decay(decay=0.6,station='h',network='v',channel='z',duration=20))
        synth_trace1 = converter(create_sinsoidal_trace_w_decay(decay=0.4,station='h',network='w',channel='z',duration=20))
        synth_trace2 = xr.apply_ufunc(shift_trace,synth_trace2,input_core_dims=[['time']],
                                                             output_core_dims=[['time']],
                                                             kwargs={'time': time_shift,'delta':synth_trace2.attrs['delta']},keep_attrs=True)
        synth_trace1.plot()
        synth_trace2.plot()
        correlation = correlator(synth_trace1,synth_trace2)
        correlation/= correlation.max()
        time_array  = correlation.coords['time'].values
        max_index   = np.argmax(correlation.data)
        tau_shift   = (time_array[max_index]- np.datetime64(UTCDateTime(0.0).datetime) ) / np.timedelta64(1, 's')
        assert tau_shift == time_shift,'failed to correctly identify tau shift'

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


if __name__ == '__main__':
    unittest.main()
