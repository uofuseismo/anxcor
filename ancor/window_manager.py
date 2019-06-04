from obspy.core.utcdatetime import UTCDateTime
from multiprocessing import Pool
from collections.abc import Iterable
from worker_factory import Worker
import psutil
import os_utils
import copy
import numpy as np
import os
from stream_jobs import process_and_save_to_file, window_worker, write_worker



class WindowManager:

    def __init__(self, window_length):
        """
        Parameters
        ----------
        window_length: float
            a float representing the window time length in seconds
        """
        self.window_length = window_length
        self.database_worker_pairs = []

    def add_database(self, database, worker: Worker):
        """
            Add a database-worker pair to the WindowManager workload

        Parameters
        ----------
        database:
            a database. So far, only Obsplus Wavebanks and ANCOR IRISBanks are supported
        worker:
            An ANCOR worker object

        """
        if self._verify_db_worker_pair(database, worker):
            self.database_worker_pairs.append({'database' : database,'worker' : worker})

    def process_windows(self,output_directory,window_starttimes, mode='cpu_limited'):
        if self._verify_mode(mode) and self._verify_starttimes(window_starttimes) and self._valid_dir(output_directory):
            compute_object = get_compute(mode)
            compute_object.set_args(starttimes      =window_starttimes, window_length=self.window_length,
                                    output_directory=output_directory,  database=self.database_worker_pairs)
            compute_object.process()


    def correlate_windows(self,source_directory, output_directory=None, max_tau_shift=None):
        if self._valid_dir(output_directory) and self._valid_dir(source_directory):
            pass

    def process_and_correlate_windows(self, window_starttimes, mode='cpu_limited',
                                            output_directory=None, max_tau_shift=None):
        if self._verify_mode(mode) and self._verify_starttimes(window_starttimes) and self._valid_dir(output_directory):
            compute_object = get_compute(mode)
            compute_object.set_args(starttimes=window_starttimes, window_length=self.window_length,
                                    output_directory=output_directory, database=self.database_worker_pairs)


    def _verify_db_worker_pair(self, database, worker):
        db_verify = hasattr(database,'get_waveforms')
        worker    = isinstance(worker, Worker)

        if not worker:
            print('invalid worker! please supply a valid worker object')

        if not db_verify:
            print('invalid database object! please supply a valid database object')

        return worker and db_verify

    def _verify_mode(self, mode):
        cpu =  mode=='cpu_limited'
        ram =  mode=='ram_limited'
        test=  mode=='test'

        if not (cpu or ram or test):
            print('invalid compute mode specified. please specify a valid compute mode.')

        return cpu or ram or test

    def _verify_starttimes(self, window_starttimes):
        if not isinstance(window_starttimes,Iterable):
            print('improper starttime object supplied!!!')
            return False
        return True

    def _valid_dir(self,output_directory):
        result = os.path.isdir(output_directory)
        if not result:
            print('{}   is not a directory!!'.format(output_directory))
        return result


class ComputeInterface:

    def __init__(self):
        pass

    def set_args(self, starttimes, window_length, output_directory, database):
        self.starttimes         = starttimes
        self.window_length      = window_length
        self.output_directory   = output_directory
        self.databases          = database

    def process(self):
        for window_number, start_of_window in enumerate(self.starttimes):

            starttime = start_of_window
            filepath = self.output_directory + '/' + str(window_number) + '/'
            os_utils.create_workingdir(filepath,fail_if_exists=False)
            for db_worker_pair in self.databases:

                worker   = db_worker_pair['worker']
                database = db_worker_pair['database']

                traces = window_worker(starttime,self.window_length,worker,database)
                write_worker(filepath, traces, format='sac')



class SingleThreadCompute(ComputeInterface):


    def __init__(self):
        super().__init__()

class LimitedRamCompute(ComputeInterface):

    def __init__(self):
        super().__init__()

class LimitedCPUCompute(ComputeInterface):

    def __init__(self):
        super().__init__()

def get_compute(mode)-> ComputeInterface:
    if mode=='test':
        return SingleThreadCompute()
    elif mode=='cpu_limited':
        return LimitedCPUCompute()
    else:
        return LimitedRamCompute()

class F:

    def __init__(self, window_length=300, overlap_percent=0):
        self.window_length   = window_length
        self.overlap_percent = overlap_percent
        self.databases=[]

    def add_database(self, database, worker: Worker):
        self.databases.append((database,worker))

    def correlate_windows(self, max_stacks=None,starttime=None,endtime=None):
        """
        Correlate waveforms stored in the given databases using the assigned workers.
        Either max_stacks & starttime must be specified, or starttime and endtime must be specified

        Parameters
        ----------
        max_stacks : int
            the maximum number of processed crosscorrelation waveforms to stack

        starttime: UTCDateTime
            the beginning time of the correlation windows

        endtime: UTCDateTime
            the ending time of the correlation windows

        Returns: dict
        -------
            a dictionary of crosscorrelated pair waveforms. returned dict has the structure:
            { 'source-receiver' : [Correlated Traces] ...}

        """
        self._check_params(max_stacks,starttime,endtime)


    def process_windows(self,directory, max_windows=None,
                        starttime=None, endtime=None, format='sac',
                        single_thread=False, physical_cores_only=True):
        """
        Performs a preprocessing operation only on the indicated windows, saving the result to file.
        Files are stored in the given directory, with subdirectory names corresponding to the indicated window.

        Parameters
        ----------
        directory: str
            directory to save the windows to
        max_windows: int
            maximum number of windows to worker_processes.py
        starttime:
            starting time of windowing
        endtime:
            ending time of windowing

        """
        self._check_params(max_windows, starttime, endtime)
        window_array = self._gen_window_array(max_windows, starttime, endtime)
        self._process_and_save_window_array(directory, format, window_array,
                                            single_thread=single_thread,physical_cores_only=physical_cores_only)

    def _process_and_save_window_array(self, directory, format, window_array, single_thread=False,physical_cores_only=True):
        usable_cpus = psutil.cpu_count(logical=physical_cores_only)
        worker_arguments = self.create_worker_args(directory, format, window_array)
        if not single_thread:
            with Pool(usable_cpus) as p:
                result = p.map(process_and_save_to_file, worker_arguments)

        else:
            for arg in worker_arguments:
                process_and_save_to_file(arg)




    def _gen_window_array(self, windows, starttime: UTCDateTime, endtime: UTCDateTime):
        delta_percent = (1.0 - self.overlap_percent / 100.0)
        if windows is not None:
            delta   = windows * delta_percent * self.window_length
            endtime_timestamp = starttime.timestamp + delta
            endtime = UTCDateTime(endtime_timestamp)
            var=endtime.timestamp - starttime.timestamp

        utc_start_times = np.arange(starttime.timestamp,
                                    endtime.timestamp,
                                    self.window_length * delta_percent).tolist()

        var = endtime.timestamp - starttime.timestamp

        utc_start_times = [ (x, x+self.window_length) for x in utc_start_times]

        return utc_start_times



    def _check_params(self,max_stacks,starttime,endtime):
        if max_stacks is None and starttime is None and endtime is None:
            raise SyntaxError('Too many Nones! Please specify a max_stacks integer'+\
                              'value and a valid starttime or a valid starttime and endtime')


    def create_worker_args(self,directory, format, start_stop_array):
        worker_arguments = []
        window_number=0
        for starttime, endtime in start_stop_array:
            database_worker_list = []
            for database, worker in self.databases:
                database_worker_list.append([copy.deepcopy(worker), copy.deepcopy(database)])

            worker_arguments.append( (window_number, directory, format, starttime, endtime,database_worker_list))
            window_number+=1

        return worker_arguments
