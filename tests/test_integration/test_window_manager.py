from ancor   import WindowManager, IRISBank
from ancor.worker_processes import Worker, OneBit, Downsample, SpectralWhiten
from obspy.core import UTCDateTime
import unittest
import os_utils

file_process='test_data/test_ancor_bank/test_save_windows'
christmas_time = '2018-12-25T00:00:00.0'

class TestWindowManager(unittest.TestCase):

    def _build_db_worker_winman(self):
        worker = Worker()
        worker.append_step(Downsample(target_rate=5.0))
        worker.append_step(SpectralWhiten(frequency_smoothing_interval=0.125, taper_percent=0.1))
        worker.append_step(OneBit())

        database = IRISBank(latitude=38.44, longitude=-112.7, minradius=0, maxradius=0.51)

        window_manager = WindowManager(window_length=15 * 60.0)
        return worker, database, window_manager

    def _process_for_correlations(self):
        worker, database, window_manager = self._build_db_worker_winman()
        window_manager.add_database(database, worker)
        start_window = [UTCDateTime(christmas_time).timestamp]
        window_manager.process_windows(file_process, start_window)

    def test_add_database_worker_pair(self):
        worker, database, window_manager = self._build_db_worker_winman()

        try:
            window_manager.add_database(database,worker)
            self.assertTrue(True)
        except TypeError:
            self.assertTrue(False)


    def test_process_windows_worker_dir(self):
        worker, database, window_manager = self._build_db_worker_winman()
        window_manager.add_database(database,worker)
        start_window = [UTCDateTime(christmas_time).timestamp]
        window_manager.process_windows(file_process, start_window)
        target_dir = file_process+'/0'
        file_exists = os_utils.folder_exists(target_dir)
        os_utils.delete_dirs(target_dir)
        self.assertTrue(file_exists,'did not create sub worker directory')

    def test_process_windows_trace_numbers(self):
        worker, database, window_manager = self._build_db_worker_winman()
        window_manager.add_database(database,worker)
        start_window = [UTCDateTime(christmas_time).timestamp]
        window_manager.process_windows(file_process, start_window)
        target_dir = file_process+'/0'

        file_list   = os_utils.get_filelist(target_dir)
        sac_list    = os_utils.get_files_with_extensions(file_list,'.sac')

        os_utils.delete_dirs(target_dir)
        self.assertEqual(len(sac_list),19,'not enough traces processed')

    def test_correlate_windows_tofile(self):
        window_manager = WindowManager(60*15)
        window_manager.correlate_windows(file_process,max_tau_shift=20.0)
        self.assertFalse(True)

    def test_process_and_correlate_windows(self):
        worker, database, window_manager = self._build_db_worker_winman()
        self.assertFalse(True)

    def test_correlate_and_stack_windows(self):
        worker, database, window_manager = self._build_db_worker_winman()
        self.assertFalse(True)

    def test_process_correlate_and_stack_windows(self):
        worker, database, window_manager = self._build_db_worker_winman()
        self.assertFalse(True)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestWindowManager())
    return suite