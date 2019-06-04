import unittest
from ancor.os_utils import delete_file, delete_dirs, file_exists
from obsplus import WaveBank
from obspy.core import UTCDateTime
from ancor import worker_factory, window_manager, IRISBank
import os


temp_repo = '../test_data/test_ancor_bank/test_waveforms'
temp_directory = '../test_data/test_ancor_bank/test_save_windows'

class TestMassProcessWavebank(unittest.TestCase):


    def tearDown(self):
        file = temp_repo + '/.index.h5'
        if file_exists(file):
            delete_file(temp_repo+'/.index.h5')

    def test_insuficcient_args(self):
        database_1   = WaveBank(temp_repo)
        database_1.update_index()
        stream = database_1.get_waveforms(station='22')
        df=database_1.get_availability_df()

        worker = worker_factory.build_worker()
        manager = window_manager.WindowManager()

        manager.add_database(database_1,worker)
        assertion=False
        try:
            manager.process_windows(temp_directory)
        except SyntaxError:
            assertion=True

        self.assertTrue(assertion, 'didn\'t catch syntax error')


    def test_windows_amount(self):

        target = 4
        database_1   = WaveBank(temp_repo)
        database_1.update_index()
        df=database_1.get_availability_df()

        min_time = UTCDateTime(df['starttime'].min())
        worker = worker_factory.build_worker()
        manager = window_manager.WindowManager()

        manager.add_database(database_1, worker)
        windows = manager._gen_window_array(target,min_time, None)
        source = len(windows)

        self.assertEqual(source,target,'incorrect generated time windows')

    def test_windows_time(self):
        database_1   = WaveBank(temp_repo)
        database_1.update_index()
        df=database_1.get_availability_df()
        target = 99
        min_time = UTCDateTime(df['starttime'].min())
        max_time = UTCDateTime(df['endtime'].max())
        worker = worker_factory.build_worker()

        manager = window_manager.WindowManager(window_length=60*60)
        manager.add_database(database_1, worker)
        windows = manager._gen_window_array(None,min_time,max_time)
        source  = len(windows)

        self.assertEqual(source,target,'incorrect generated time windows')

    def test_process_save(self):
        database_1   = WaveBank(temp_repo)
        database_1.update_index()
        df=database_1.get_availability_df()
        target = 99
        min_time = UTCDateTime(df['starttime'].min())
        max_time = UTCDateTime(df['endtime'].max())
        worker = worker_factory.build_worker()

        manager = window_manager.WindowManager(window_length=60*60)
        manager.add_database(database_1, worker)
        manager.process_windows(temp_directory,starttime=min_time,max_windows=5)


        self.assertTrue(os.path.exists(temp_directory + '/0'),'incorrect saved time windows')
        self.assertTrue(os.path.exists(temp_directory + '/1'), 'incorrect saved time windows')
        self.assertTrue(os.path.exists(temp_directory + '/2'), 'incorrect saved time windows')
        self.assertTrue(os.path.exists(temp_directory + '/3'), 'incorrect saved time windows')
        self.assertTrue(os.path.exists(temp_directory + '/4'), 'incorrect saved time windows')
        for i in range(0,5):
            delete_dirs(temp_directory + '/' + str(i))


    def test_onebit_worker(self):
        database_1   = WaveBank(temp_repo)
        database_1.update_index()
        df=database_1.get_availability_df()
        target = 99
        min_time = UTCDateTime(df['starttime'].min())
        max_time = UTCDateTime(df['endtime'].max())
        worker = worker_factory._shapiro()

        manager = window_manager.WindowManager(window_length=60*60)
        manager.add_database(database_1, worker)
        manager.process_windows(temp_directory,starttime=min_time,max_windows=5)

        self.assertTrue(os.path.exists(temp_directory + '/0'),'incorrect saved time windows')
        self.assertTrue(os.path.exists(temp_directory + '/1'), 'incorrect saved time windows')
        self.assertTrue(os.path.exists(temp_directory + '/2'), 'incorrect saved time windows')
        self.assertTrue(os.path.exists(temp_directory + '/3'), 'incorrect saved time windows')
        self.assertTrue(os.path.exists(temp_directory + '/4'), 'incorrect saved time windows')
        for i in range(0,5):
            delete_dirs(temp_directory + '/' + str(i))



class TestMassProcessIRISRouter(unittest.TestCase):


    def tearDown(self):
        file = temp_repo + '/.index.h5'
        if file_exists(file):
            delete_file(temp_repo + '/.index.h5')

    def test_insuficcient_args(self):
        database_1   = IRISBank(minlongitude=-111.5,maxlongitude=-111,minlatitude=40,maxlatitude=41)
        worker = worker_factory.build_worker()
        manager = window_manager.WindowManager()

        manager.add_database(database_1,worker)
        assertion=False
        try:
            manager.process_windows(temp_directory)
        except SyntaxError:
            assertion=True

        self.assertTrue(assertion, 'didn\'t catch syntax error')


    def test_windows_amount(self):

        target = 4
        database_1   = IRISBank(minlongitude=-111.5,maxlongitude=-111,minlatitude=40,maxlatitude=41)

        min_time = UTCDateTime(1240561632.5)
        worker = worker_factory.build_worker()
        manager = window_manager.WindowManager()

        manager.add_database(database_1, worker)
        windows = manager._gen_window_array(target,min_time, None)
        source = len(windows)

        self.assertEqual(source,target,'incorrect generated time windows')

    def test_windows_time(self):
        database_1    = IRISBank(minlongitude=-111.5,maxlongitude=-111,minlatitude=40,maxlatitude=41)
        target = 5
        min_time = UTCDateTime(1240561632.5)
        max_time = UTCDateTime(1240561632.5 + 3600*target)
        worker = worker_factory.build_worker()

        manager = window_manager.WindowManager(window_length=60*60)
        manager.add_database(database_1, worker)
        windows = manager._gen_window_array(None,min_time,max_time)
        source  = len(windows)

        self.assertEqual(target,source,'incorrect generated time windows')

    def test_process_save(self):
        database_1    = IRISBank(minlongitude=-111.5,maxlongitude=-111,minlatitude=40,maxlatitude=41)
        target = 99
        min_time = UTCDateTime(1240561632.5)
        max_time = UTCDateTime(1240564632.5)
        worker = worker_factory.build_worker()

        manager = window_manager.WindowManager(window_length=60*60)
        manager.add_database(database_1, worker)
        manager.process_windows(temp_directory,starttime=min_time,max_windows=5)


        self.assertTrue(os.path.exists(temp_directory + '/0'),'incorrect saved time windows')
        self.assertTrue(os.path.exists(temp_directory + '/1'), 'incorrect saved time windows')
        self.assertTrue(os.path.exists(temp_directory + '/2'), 'incorrect saved time windows')
        self.assertTrue(os.path.exists(temp_directory + '/3'), 'incorrect saved time windows')
        self.assertTrue(os.path.exists(temp_directory + '/4'), 'incorrect saved time windows')
        for i in range(0,5):
            delete_dirs(temp_directory + '/' + str(i))


    def test_onebit_worker(self):
        database_1    = IRISBank(minlongitude=-111.5,maxlongitude=-111,minlatitude=40,maxlatitude=41)

        target = 99
        min_time = UTCDateTime(1240561632.5)
        max_time = UTCDateTime(1240564632.5)
        worker = worker_factory._shapiro()

        manager = window_manager.WindowManager(window_length=60*60)
        manager.add_database(database_1, worker)
        manager.process_windows(temp_directory,starttime=min_time,max_windows=5,single_thread=True)

        self.assertTrue(os.path.exists(temp_directory + '/0'),'incorrect saved time windows')
        self.assertTrue(os.path.exists(temp_directory + '/1'), 'incorrect saved time windows')
        self.assertTrue(os.path.exists(temp_directory + '/2'), 'incorrect saved time windows')
        self.assertTrue(os.path.exists(temp_directory + '/3'), 'incorrect saved time windows')
        self.assertTrue(os.path.exists(temp_directory + '/4'), 'incorrect saved time windows')
        for i in range(0,5):
            delete_dirs(temp_directory + '/' + str(i))




if __name__ == '__main__':
    unittest.main()