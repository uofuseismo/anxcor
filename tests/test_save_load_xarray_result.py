import unittest
import os
from anxcor.utils import _clean_files_in_dir, _how_many_fmt
from anxcor.core import Anxcor
import anxcor.utils as utils
from anxcor.xarray_routines import XArrayTemporalNorm
import json

save_dir = 'tests/test_data/test_anxcor_database/test_save_config'
if not utils.folder_exists(save_dir):
    os.mkdir(save_dir)

class TestConfig(unittest.TestCase):

    def tearDown(self):
        _clean_files_in_dir(save_dir)

    def test_save_result(self):
        assert False

    def test_load_result(self):
        assert False



if __name__ == '__main__':
    unittest.main()