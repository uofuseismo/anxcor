import unittest
import os
from anxcor.utils import _clean_files_in_dir, _how_many_fmt
from anxcor.core import Anxcor
from anxcor.xarray_routines import XArrayTemporalNorm
import json

save_dir = 'test_data/test_ancor_bank/test_save_config'
class TestConfig(unittest.TestCase):

    def tearDown(self):
        _clean_files_in_dir(save_dir)

    def test_single_execution(self):
        anxcor = Anxcor(window_length=3600)
        anxcor.save_config(save_dir + os.sep + 'config.ini')
        amnt = _how_many_fmt(save_dir,'.ini')
        assert amnt == 1

    def test_tau_is_correct(self):
        target_shift = 15.0
        anxcor = Anxcor(window_length=3600)
        anxcor.set_task_kwargs('crosscorrelate', dict(max_tau_shift=target_shift))
        anxcor.save_config(save_dir + os.sep + 'config.ini')
        with open(save_dir+os.sep+'config.ini') as conf:
            config = json.load(conf)

        source_tau_shift = config['crosscorrelate']['max_tau_shift']

        assert target_shift == source_tau_shift


    def test_load_config_wout_routine(self):
        anxcor = Anxcor(window_length=3600)
        anxcor.add_process(XArrayTemporalNorm(time_window=15, rolling_procedure='max'))
        anxcor.save_config(save_dir + os.sep + 'config.ini')

        anxcor2 = Anxcor(window_length=3600)
        anxcor2.load_config(save_dir + os.sep + 'config.ini')
        assert True


    def test_load_config_with_routine(self):
        time_mean_target = 15
        anxcor = Anxcor(window_length=3600)
        anxcor.add_process(XArrayTemporalNorm(time_window=time_mean_target, rolling_procedure='max'))
        anxcor.save_config(save_dir + os.sep + 'config.ini')

        anxcor2 = Anxcor(window_length=3600)
        anxcor2.add_process(XArrayTemporalNorm())
        anxcor2.load_config(save_dir + os.sep + 'config.ini')
        tnorm=anxcor2.get_task('process')['temp_norm']
        time_mean_src = tnorm._kwargs['time_window']
        assert time_mean_target == time_mean_src


if __name__ == '__main__':
    unittest.main()