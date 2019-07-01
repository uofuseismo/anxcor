import unittest
from tests.synthetic_trace_factory import create_random_trace, create_sinsoidal_trace_w_decay, create_triangle_trace
from anxcor.xarray_routines import XArrayXCorrelate, XArrayConverter
from anxcor.containers import XArrayStack
from anxcor.filters import _multiply_in_mat, xarray_crosscorrelate
import numpy as np
import xarray as xr


def shift_trace(data,time=1.0,delta=0.1):
    sampling_rate = int(1.0/delta)
    random_data   = np.random.uniform(-1,1,(data.shape[0],data.shape[1],int(sampling_rate*time)))
    new_data      = np.concatenate((random_data,data),axis=2)[:,:,:data.shape[2]]
    return new_data

stacker   = XArrayStack()
converter = XArrayConverter()
class TestCorrelation(unittest.TestCase):


    def test_autocorrelation(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter([create_random_trace(station='1',network='v',duration=20)],starttime=0,station=0)

        correlation = correlator(syth_trace1,syth_trace1,starttime=0,station=0)
        zero_target_index = correlation.data.shape[3]//2

        zero_source_index = np.argmax(correlation.data[0,0,0,:])
        assert zero_source_index == zero_target_index,'autocorrelation failed'

    def test_stacking_preserves_metadata(self):
        correlator = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter([create_random_trace(station='1', network='v', duration=20)],starttime=0,station=0)

        result_1 = correlator(syth_trace1, syth_trace1,starttime=0,station=0)
        result_2 = correlator(syth_trace1, syth_trace1,starttime=0,station=0)

        stack = stacker(result_1,result_2,starttime=0,station=0)

        attrs = stack.attrs

        assert 'starttime' in attrs.keys(), 'starttime did not persist through stacking'
        assert 'endtime' in attrs.keys(), 'endtime did not persist through stacking'
        assert attrs['stacks'] == 2,'unexpected stack length'
        assert 'operations' in attrs.keys(), 'operations did not persist through stacking'
        assert 'delta' in attrs.keys(), 'delta did not persist through stacking'


    def test_has_starttime(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter([create_random_trace(station='1',network='v',duration=20)],starttime=0,station=0)

        keys = correlator(syth_trace1,syth_trace1,starttime=0,station=0).attrs.keys()

        assert 'starttime' in keys, 'starttime not preserved through correlation'

    def test_has_endtime(self):

        correlator = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter([create_random_trace(station='1', network='v', duration=20)],starttime=0,station=0)

        keys = correlator(syth_trace1, syth_trace1,starttime=0,station=0).attrs.keys()

        assert 'endtime' in keys, 'endtime not preserved through correlation'

    def test_has_delta(self):

        correlator = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter([create_random_trace(station='1', network='v', duration=20)],starttime=0,station=0)

        keys = correlator(syth_trace1, syth_trace1,starttime=0,station=0).attrs.keys()

        assert 'delta' in keys, 'delta not preserved through correlation'

    def test_has_stacks(self):

        correlator = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter([create_random_trace(station='1', network='v', duration=20)],starttime=0,station=0)

        keys = correlator(syth_trace1, syth_trace1,starttime=0,station=0).attrs.keys()

        assert 'stacks' in keys, 'stacks not preserved through correlation'

    def test_one_stack(self):

        correlator = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter([create_random_trace(station='1', network='v', duration=20)],starttime=0,station=0)

        attr = correlator(syth_trace1, syth_trace1,starttime=0,station=0).attrs

        assert attr['stacks'] == 1,'stacks assigned improper value'

    def test_autocorrelation_delta_attr(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter([create_random_trace(station='1',network='v',duration=20)],starttime=0,station=0)

        correlation = correlator(syth_trace1,syth_trace1,starttime=0,station=0)

        assert 'delta' in correlation.attrs.keys(),'did not propagate the delta value'

    def test_autocorrelation_starttime_attr(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter([create_random_trace(station='1',network='v',duration=20)],starttime=0,station=0)

        correlation = correlator(syth_trace1,syth_trace1,starttime=0,station=0)

        assert 'starttime' in correlation.attrs.keys(),'did not propagate the delta value'

    def test_name(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter([create_random_trace(station='1',network='v',duration=20)],starttime=0,station=0)

        correlation = correlator(syth_trace1,syth_trace1,starttime=0,station=0)

        assert isinstance(correlation.name,str),'improper name type'

    def test_array_is_real(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter([create_random_trace(station='1',network='v',duration=20)],starttime=0,station=0)

        correlation = correlator(syth_trace1,syth_trace1,starttime=0,station=0)

        assert correlation.data.dtype == np.float64,'improper data type'

    def test_correlation_length(self):

        correlator  = XArrayXCorrelate(max_tau_shift=5.0)
        syth_trace1 = converter([create_random_trace(station='1',network='v')],starttime=0,station=0)
        syth_trace2 = converter([create_random_trace(station='2',network='v')],starttime=0,station=0)
        try:
            correlation = correlator(syth_trace1,syth_trace2,starttime=0,station=0)
            assert False,'failed to catch exception'
        except Exception:
            assert True, 'failed to catch exception'


    def test_shift_trace(self):
        max_tau_shift = 8
        correlator  = XArrayXCorrelate(max_tau_shift=max_tau_shift)
        time_shift  = 2
        syth_trace2 = converter([create_sinsoidal_trace_w_decay(decay=0.6,station='h',network='v',channel='z',duration=20)],starttime=0,station=0)
        syth_trace1 = converter([create_sinsoidal_trace_w_decay(decay=0.4,station='h',network='w',channel='z',duration=20)],starttime=0,station=0)
        syth_trace2 = xr.apply_ufunc(shift_trace,syth_trace2,input_core_dims=[['time']],
                                                             output_core_dims=[['time']],
                                                             kwargs={'time': time_shift,'delta':syth_trace2.attrs['delta']},keep_attrs=True)

        correlation = correlator(syth_trace1,syth_trace2,starttime=0,station=0)
        zero_index = max_tau_shift*40
        tau_shift  = time_shift*40# added two because of noise effects
        max_index  = np.argmax(correlation.data)
        assert max_index == zero_index+tau_shift,'failed to correctly identify tau shift'

    def test_correct_pair_ee(self):
        max_tau_shift = 8
        correlator = XArrayXCorrelate(max_tau_shift=max_tau_shift)
        time_shift = 2

        e1 = create_sinsoidal_trace_w_decay(decay=0.9, station='h', network='v', channel='e', duration=20)
        n1 = create_sinsoidal_trace_w_decay(decay=0.3, station='h', network='v', channel='n', duration=20)
        z1 = create_sinsoidal_trace_w_decay(decay=0.4, station='h', network='v', channel='z', duration=20)

        e2 = create_sinsoidal_trace_w_decay(decay=0.8, station='k', network='v', channel='e', duration=20)
        n2 = create_sinsoidal_trace_w_decay(decay=0.7, station='k', network='v', channel='n', duration=20)
        z2 = create_sinsoidal_trace_w_decay(decay=0.6, station='k', network='v', channel='z', duration=20)
        syth_trace1 = converter([e1.copy(), z1.copy()],starttime=0,station=0)
        syth_trace2 = converter([e2.copy(), z2.copy()],starttime=0,station=0)

        correlation_source = correlator(syth_trace1, syth_trace2,starttime=0,station=0)
        correlation_ee     = correlator(converter([e1.copy()],starttime=0,station=0),
                                        converter([e2.copy()],starttime=0,station=0),starttime=0,station=0)
        result_1     = correlation_source.loc['e', 'e', :, :] - correlation_ee
        assert 0 == np.sum(result_1.data)


    def test_correct_pair_zz(self):
        max_tau_shift = 8
        correlator = XArrayXCorrelate(max_tau_shift=max_tau_shift)
        time_shift = 2

        e1 = create_sinsoidal_trace_w_decay(decay=0.9, station='h', network='v', channel='e', duration=20)
        n1 = create_sinsoidal_trace_w_decay(decay=0.3, station='h', network='v', channel='n', duration=20)
        z1 = create_sinsoidal_trace_w_decay(decay=0.4, station='h', network='v', channel='z', duration=20)

        e2 = create_sinsoidal_trace_w_decay(decay=0.8, station='k', network='v', channel='e', duration=20)
        n2 = create_sinsoidal_trace_w_decay(decay=0.7, station='k', network='v', channel='n', duration=20)
        z2 = create_sinsoidal_trace_w_decay(decay=0.6, station='k', network='v', channel='z', duration=20)
        syth_trace1 = converter([e1, z1],starttime=0,station=0)
        syth_trace2 = converter([e2, z2],starttime=0,station=0)

        correlation_source = correlator(syth_trace1, syth_trace2,starttime=0,station=0)
        correlation_zz = correlator(converter([z1.copy()],starttime=0,station=0),
                                    converter([z2.copy()],starttime=0,station=0),starttime=0,station=0)
        result_1 = correlation_source.loc['z', 'z', :, :] - correlation_zz
        assert 0 == np.sum(result_1.data)

    def test_correct_pair_ez(self):
        max_tau_shift = 8
        correlator = XArrayXCorrelate(max_tau_shift=max_tau_shift)
        time_shift = 2

        e1 = create_sinsoidal_trace_w_decay(decay=0.9, station='h', network='v', channel='e', duration=20)
        n1 = create_sinsoidal_trace_w_decay(decay=0.3, station='h', network='v', channel='n', duration=20)
        z1 = create_sinsoidal_trace_w_decay(decay=0.4, station='h', network='v', channel='z', duration=20)

        e2 = create_sinsoidal_trace_w_decay(decay=0.8, station='k', network='v', channel='e', duration=20)
        n2 = create_sinsoidal_trace_w_decay(decay=0.7, station='k', network='v', channel='n', duration=20)
        z2 = create_sinsoidal_trace_w_decay(decay=0.6, station='k', network='v', channel='z', duration=20)
        syth_trace1 = converter([e1, z1],starttime=0,station=0)
        syth_trace2 = converter([e2, z2],starttime=0,station=0)

        correlation_source = correlator(syth_trace1, syth_trace2,starttime=0,station=0)
        correlation_zz = correlator(converter([e1.copy()],starttime=0,station=0),
                                    converter([z2.copy()],starttime=0,station=0),starttime=0,station=0)
        result_1 = correlation_source.loc['e', 'z', :, :] - correlation_zz
        assert 0 == np.sum(result_1.data)

    def test_correct_pair_ze(self):
        max_tau_shift = 8
        correlator = XArrayXCorrelate(max_tau_shift=max_tau_shift)
        time_shift = 2

        e1 = create_sinsoidal_trace_w_decay(decay=0.9, station='h', network='v', channel='e', duration=20)
        n1 = create_sinsoidal_trace_w_decay(decay=0.3, station='h', network='v', channel='n', duration=20)
        z1 = create_sinsoidal_trace_w_decay(decay=0.4, station='h', network='v', channel='z', duration=20)

        e2 = create_sinsoidal_trace_w_decay(decay=0.8, station='k', network='v', channel='e', duration=20)
        n2 = create_sinsoidal_trace_w_decay(decay=0.7, station='k', network='v', channel='n', duration=20)
        z2 = create_sinsoidal_trace_w_decay(decay=0.6, station='k', network='v', channel='z', duration=20)
        syth_trace1 = converter([e1, z1],starttime=0,station=0)
        syth_trace2 = converter([e2, z2],starttime=0,station=0)

        correlation_source = correlator(syth_trace1, syth_trace2,starttime=0,station=0)

        correlation_ze = correlator(converter([z1],starttime=0,station=0),
                                    converter([e2],starttime=0,station=0),starttime=0,station=0)

        result_1 = correlation_source.loc['z', 'e', :, :] - correlation_ze[0, 0, :, :]

        assert 0 == np.sum(result_1.data)



    def test_correct_pair_ez24(self):
        max_tau_shift = None
        correlator = XArrayXCorrelate(max_tau_shift=max_tau_shift)
        time_shift = 2

        e1 = create_sinsoidal_trace_w_decay(decay=0.9, station='h', network='v', channel='e', duration=20)
        z1 = create_triangle_trace(                    station='h', network='v', channel='z', duration=20)

        e2 = create_sinsoidal_trace_w_decay(decay=0.8, station='k', network='v', channel='e', duration=20)
        n2 = create_random_trace(                      station='k', network='v', channel='n', duration=20)
        z2 = create_sinsoidal_trace_w_decay(decay=0.3, station='k', network='v', channel='z', duration=20)
        b2 = create_random_trace(station='k', network='v', channel='b', duration=20)
        syth_trace1 = converter([e1.copy(), z1.copy()],starttime=0,station=0)
        syth_trace2 = converter([e2.copy(), n2.copy(), z2.copy(), b2.copy()],starttime=0,station=0)

        correlation_source = correlator(syth_trace1, syth_trace2,starttime=0,station=0)

        correlation_ez24 = correlator(converter([e1.copy()],starttime=0,station=0),
                                      converter([z2.copy()],starttime=0,station=0),starttime=0,station=0)

        result_1 = correlation_source.loc['e', 'z', :, :] - correlation_ez24[0, 0, :, :]

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


    def test_xcorr_dummy_execution(self):
        times = np.arange(0,5)
        one_orig   = np.arange(0, 4).reshape((4,1))
        two_orig   = np.arange(1, 4).reshape((3,1))
        one        = one_orig.copy()
        two        = two_orig.copy()
        for i in range(0,4):
            one = np.concatenate((one,one_orig),axis=1)
            two = np.concatenate((two,two_orig),axis=1)

        two =two.reshape((3, 1, 5))
        one =one.reshape((4, 1, 5))

        xarray_source = xr.DataArray(one,coords=(('channel',['a','b','c','d']),
                                                 ('station_id',['0']),
                                                 ('time', times)))

        xarray_receiver = xr.DataArray(two, coords=(('channel', ['x','y','z']),
                                                  ('station_id', ['1']),
                                                  ('time', times)))

        attrs = {'starttime':0,'delta':1}
        xarray_receiver.attrs = attrs
        xarray_source.attrs   = attrs
        result = xarray_crosscorrelate(xarray_source,xarray_receiver,dummy_task=True,starttime=0,station=0)

        assert result.loc['d','z',:,0]==9
        assert result.loc['c', 'z', :, 0] == 6

    def test_nonetype_in_out(self):
        correlator = XArrayXCorrelate()
        result = correlator(None,None, starttime=0, station=0)
        assert result == None


if __name__ == '__main__':
    unittest.main()
