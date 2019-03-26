import unittest
from ancor_windowmap import WindowMap
from ancor_file_indexing import FileIndex
from datetime import  datetime, timedelta
from ancor.ancor import os_utils

save_dir = 'test_data/test_file_connect_windowmap/test_save_directory'
class MockFileIndex(FileIndex):

    def __init__(self):
        super().__init__()
        self.map = {}
        self.map['0']=[]
        self.map['08242009'] =['test_data/test_ancor_bank_generator/test_date_keys/date_key_dir/20090824.sac']
        self.map['08262009'] =['test_data/test_ancor_bank_generator/test_date_keys/date_key_dir/20090826.sac']
        self.map['08282009'] =['test_data/test_ancor_bank_generator/test_date_keys/date_key_dir/20090828.sac']


    def get_file_map(self,**kwargs):
        return self.map


class TestWindowMap(unittest.TestCase):

    def setUp(self):
        self.windowmap = WindowMap(file_index=MockFileIndex())

    def test_date_extraction(self):
        min_date_target = datetime.strptime('08242009','%m%d%Y')
        max_date_target = datetime.strptime('08292009', '%m%d%Y')
        max_date_source, min_date_source = self.windowmap._extract_minmax_dates()
        self.assertEqual(min_date_target, min_date_source)
        self.assertEqual(max_date_target, max_date_source)

    def test_window(self):
        target_window = timedelta(seconds=1000)
        target_increment = timedelta(seconds=0.0)
        self.windowmap.window_length = 1000
        self.windowmap.overlap_percent = 0.0
        source_window, source_increment = self.windowmap._get_time_iterables()
        self.assertEqual(target_window, source_window)

    def test_iteration_length1(self):
        target_increment = timedelta(seconds=250)
        self.windowmap.window_length = 1000
        self.windowmap.overlap_percent = 0.75
        source_window, source_increment = self.windowmap._get_time_iterables()
        self.assertEqual(source_increment, target_increment)

    def test_iteration_length2(self):
        target_increment = timedelta(seconds=750)
        self.windowmap.window_length = 1000
        self.windowmap.overlap_percent = 0.25
        source_window, source_increment = self.windowmap._get_time_iterables()
        self.assertEqual(source_increment, target_increment)


    def test_window_1_jobs(self):
        self.windowmap.window_length   = 10000
        self.windowmap.overlap_percent = 0.0
        map = self.windowmap.map_jobs_to_files()
        target = 8
        source = len(map[8]['windows'])
        self.assertEqual(target,source)

    def test_window_2_jobs(self):
        self.windowmap.window_length = 20000
        self.windowmap.overlap_percent = 0.0
        map = self.windowmap.map_jobs_to_files()
        target = 3
        source = len(map[8]['windows'])
        self.assertEqual(target, source)

    def test_overlap_1_jobs(self):
        self.windowmap.window_length = 10000
        self.windowmap.overlap_percent = 0.25
        map = self.windowmap.map_jobs_to_files()
        target = 10
        source = len(map[8]['windows'])
        self.assertEqual(target, source)

    def test_overlap_2_jobs(self):
        self.windowmap.window_length = 10000
        self.windowmap.overlap_percent = 0.75
        map = self.windowmap.map_jobs_to_files()
        target = 30
        source = len(map[8]['windows'])
        self.assertEqual(target, source)

    def test_save_file(self):
        test_save_file = save_dir + os_utils.sep + 'job_file.json'
        self.windowmap.window_length = 10000
        self.windowmap.overlap_percent = 0.0
        self.windowmap.save_readable_job(test_save_file)
        self.assertTrue(os_utils.file_exists(test_save_file))
        os_utils.delete_file(test_save_file)

if __name__ == '__main__':
    unittest.main()
