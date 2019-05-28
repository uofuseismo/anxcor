from ancor.ancor_bank import AncorBank
from ancor.ancor_windowmap import WindowMap
from ancor.ancor_file_indexing import FileIndex
import unittest

source_data = 'test_data/test_ancor_bank/test_jobs/date_key_dir'
test_dir    = 'test_data/test_ancor_bank/test_jobs/test_directory'

class TestWindowMap(unittest.TestCase):

    def setUp(self):
        f_index = FileIndex(directory=source_data,format='%Y%m%d')
        w_map = WindowMap(file_index=f_index,window_length=10000)
        self.jobs = w_map.map_jobs_to_files()


    def test_single_job_bank(self):
        bank=AncorBank(self.jobs[0],working_directory=test_dir,
                       processor_data="single test processor")
        bank.execute()
        success = bank._processor.success
        self.assertTrue(success)

    def test_all_jobs(self):
        bank = AncorBank(self.jobs, working_directory=test_dir, processor_data="multi test processor")
        bank.execute()
        success = bank._processor.successes
        total = all(item for item in success)
        self.assertTrue(total)

if __name__ == '__main__':
    unittest.main()
