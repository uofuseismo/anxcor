import unittest
from anxcor.xarray_routines import XArrayConverter
# for travis build
import tests.synthetic_trace_factory as synthfactory

from scipy.signal import correlate
import pytest
import xarray as xr
import anxcor.filters as filt_ops

import numpy as np

convert = XArrayConverter()
class TestSimpleRelations(unittest.TestCase):

    def test_zero_ended_taper(self):
        data = np.random.uniform(0,1,1000)
        result = filt_ops.taper_func(data, taper=0.1)
        assert result[0]==0
        assert result[-1]==0

    def test_zero_ended_taper_odd(self):
        data = np.random.uniform(0,1,1000)
        result = filt_ops.taper_func(data, taper=0.1)
        assert result[0]==0
        assert result[-1]==0


class TestImpulseDecays(unittest.TestCase):

    def test_lowpass_farfield_impulse(self):
        stream   = synthfactory.create_impulse_stream(sampling_rate=40.0, duration = 1000.0)
        xarray    = convert(stream)
        filtered_array = xr.apply_ufunc(filt_ops.lowpass_filter, xarray,
                                       input_core_dims=[['time']],
                                       output_core_dims=[['time']],
                                       kwargs={'sample_rate': 40.0,
                                               'upper_frequency': 5.0},
                                       keep_attrs=True)
        assert filtered_array.data[0, 0, -1] == pytest.approx(0, abs=1e-18)

    def test_bandpass_farfield_impulse(self):
        stream = synthfactory.create_impulse_stream(sampling_rate=40.0, duration=1000.0)
        xarray = convert(stream)
        filtered_array = xr.apply_ufunc(filt_ops.bandpass_in_time_domain, xarray,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={'sample_rate': 40.0,
                                                'upper_frequency': 5.0,
                                                'lower_frequency':0.01},
                                        keep_attrs=True)
        assert filtered_array.data[0, 0, -1] == pytest.approx(0, abs=1e-18)


    def test_taper_farfield_impulse(self):
        stream = synthfactory.create_impulse_stream(sampling_rate=40.0, duration=1000.0)
        xarray = convert(stream)
        filtered_array = xr.apply_ufunc(filt_ops.taper_func, xarray,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={'taper': 0.1},
                                        keep_attrs=True)
        assert filtered_array.data[0, 0, -1] == pytest.approx(0, abs=1e-18)

    def test_into_freq_and_back(self):
        stream = synthfactory.create_impulse_stream(sampling_rate=40.0, duration=1000.0)
        xarray = convert(stream)
        xarray_freq = filt_ops.xarray_time_2_freq(xarray)
        xarray_time = filt_ops.xarray_freq_2_time(xarray_freq,xarray)

        assert np.allclose(xarray_time.data,xarray.data)


    def test_crosscorrelate_farfield_impulse(self):
        stream1 = synthfactory.create_impulse_stream(sampling_rate=40.0, duration=1000.0)
        stream2 = synthfactory.create_impulse_stream(sampling_rate=40.0, duration=1000.0)
        xarray_src = convert(stream1)
        xarray_rec = convert(stream2)
        xarray_corr = filt_ops.xarray_crosscorrelate(xarray_src,xarray_rec)

        assert xarray_corr.data.shape[-1]==xarray_src.data.shape[-1]*2 -1


class TestZeroPhaseFilter(unittest.TestCase):

    def test_lowpass_zero_phase(self):
        stream   = synthfactory.create_sinsoidal_trace_w_decay(sampling_rate=40.0, duration = 1000.0,
                                                               period=0.5)
        xarray    = convert(stream)
        filtered_array = xr.apply_ufunc(filt_ops.lowpass_filter, xarray,
                                       input_core_dims=[['time']],
                                       output_core_dims=[['time']],
                                       kwargs={'sample_rate': 40.0,
                                               'upper_frequency': 5.0},
                                       keep_attrs=True)
        a =  xarray.data[0,0,:]
        b = filtered_array.data[0,0,:]
        xcorr = correlate(a, b)

        # delta time array to match xcorr
        dt = np.arange(1 - a.shape[-1], a.shape[-1])

        recovered_time_shift = dt[xcorr.argmax()]

        assert recovered_time_shift==0

    def test_bandpass_zero_phase(self):
        stream = synthfactory.create_sinsoidal_trace(sampling_rate=40.0, duration=1000.0,
                                                             period=0.3)
        xarray = convert(stream)
        filtered_array = xr.apply_ufunc(filt_ops.taper_func, xarray,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={'taper':0.01},
                                        keep_attrs=True)
        filtered_array = xr.apply_ufunc(filt_ops.bandpass_in_time_domain, filtered_array,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={'sample_rate': 40.0,
                                                'upper_frequency': 5.0,
                                                'lower_frequency': 0.01},
                                        keep_attrs=True)
        a = xarray.data[0, 0, :]
        b = filtered_array.data[0, 0, :]
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(a)
        plt.plot(b)
        plt.show()
        xcorr = correlate(a, b)

        # delta time array to match xcorr
        dt = np.arange(1 - a.shape[-1], a.shape[-1])

        recovered_time_shift = dt[xcorr.argmax()]

        assert recovered_time_shift == 0


    def test_taper_phase_shift(self):
        stream = synthfactory.create_sinsoidal_trace_w_decay(sampling_rate=40.0, duration=1000.0,
                                                             period=0.5)
        xarray = convert(stream)
        filtered_array = xr.apply_ufunc(filt_ops.taper_func, xarray,
                                        input_core_dims=[['time']],
                                        output_core_dims=[['time']],
                                        kwargs={'taper': 0.1},
                                        keep_attrs=True)
        a = xarray.data[0, 0, :]
        b = filtered_array.data[0, 0, :]
        xcorr = correlate(a, b)

        # delta time array to match xcorr
        dt = np.arange(1 - a.shape[-1], a.shape[-1])

        recovered_time_shift = dt[xcorr.argmax()]

        assert recovered_time_shift == 0

    def test_into_freq_and_back_phase_shift(self):
        stream = synthfactory.create_sinsoidal_trace_w_decay(sampling_rate=40.0, duration=1000.0,
                                                             period=0.5)
        xarray = convert(stream)
        xarray_freq = filt_ops.xarray_time_2_freq(xarray)
        xarray_time = filt_ops.xarray_freq_2_time(xarray_freq,xarray)

        a = xarray.data[0, 0, :]
        b = xarray_time.data[0, 0, :]
        xcorr = correlate(a, b)

        # delta time array to match xcorr
        dt = np.arange(1 - a.shape[-1], a.shape[-1])

        recovered_time_shift = dt[xcorr.argmax()]

        assert recovered_time_shift == 0


    def test_crosscorrelate_phase_shift(self):
        stream1 = synthfactory.create_sinsoidal_trace(sampling_rate=10.0, duration=60.1,
                                                             period=15)
        stream2 = synthfactory.create_sinsoidal_trace(sampling_rate=10.0, duration=60.1,
                                                             period=15)
        xarray_src = convert(stream1)
        xarray_rec = convert(stream2)
        xarray_corr = filt_ops.xarray_crosscorrelate(xarray_src,xarray_rec)
        xcorr_data = xarray_corr.data[0,0,:].squeeze()
        dt = np.arange(1 -  xarray_src.data.shape[-1],  xarray_rec.data.shape[-1])

        recovered_time_shift = dt[xcorr_data.argmax()]


        assert recovered_time_shift == 0


if __name__ == '__main__':
    unittest.main()