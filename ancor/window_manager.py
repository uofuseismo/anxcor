
from obspy.core import  read, Stream
from multiprocessing import Pool
from collections.abc import Iterable
from ancor.worker_processes import Worker
import psutil
import os_utils
import copy
import numpy as np
import os
from stream_jobs import process_and_save_to_file, window_worker, write_worker, window_correlator



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

    def process_windows(self,output_directory, window_starttimes, mode='test'):
        """
            Performs preprocessing prior to crosscorrelations with the intended result being lots of treated files on disk
        Parameters
        ----------
        output_directory
        window_starttimes
        mode

        Returns
        -------

        """
        if self._verify_mode(mode) and self._verify_starttimes(window_starttimes) and self._valid_dir(output_directory):
            compute_object = get_compute(mode)
            compute_object.set_args(starttimes      =window_starttimes, window_length=self.window_length,
                                    output_directory=output_directory,  database=self.database_worker_pairs)
            compute_object.process()


    def correlate_windows(self,source_directory_list, corr_worker,output_directory=None,mode='test'):
        """
            takes files on disk and performs component-wise crosscorrelations
        Parameters
        ----------
        source_directory_list: List[str]
            list object containing source directories
        corr_worker:
            correlation worker of choice
        output_directory:
            directory to save the correlation output. if none, will return a list of correlated traces
        mode: str
            computation mode to execute the correlation

        Returns
        -------

        """
        if self._valid_dir(output_directory) and self._valid_dirs(source_directory_list):
            compute_object = get_compute(mode)
            result         = compute_object.correlate(source_directory_list, corr_worker, output_directory)

    def correlate_and_stack_windows(self,corr_worker):
        pass

    def process_and_correlate_windows(self, window_starttimes, mode='test',
                                            output_directory=None, max_tau_shift=None):
        """
            Performs processing and correlation in one go with the intended result being crosscorrelations on disk
        Parameters
        ----------
        window_starttimes
        mode
        output_directory
        max_tau_shift

        Returns
        -------

        """
        if self._verify_mode(mode) and self._verify_starttimes(window_starttimes) and self._valid_dir(output_directory):
            compute_object = get_compute(mode)
            compute_object.set_args(starttimes=window_starttimes, window_length=self.window_length,
                                    output_directory=output_directory, database=self.database_worker_pairs)

    def process_correlate_and_stack_windows(self, window_starttimes, mode='cpu_limited',
                                            output_directory=None, max_tau_shift=None):
        """
            Performs processing, correlation, and stacking in one go with the intended result being either xcorr on disk or
            a returned dictionary of source-receiver-component corrs.
        ----------
        window_starttimes
        mode
        output_directory
        max_tau_shift

        Returns
        -------

        """
        if self._verify_mode(mode) and self._verify_starttimes(window_starttimes) and self._valid_dir(output_directory):
            compute_object = get_compute(mode)
            compute_object.set_args(starttimes=window_starttimes, window_length=self.window_length,
                                    output_directory=output_directory, database=self.database_worker_pairs)


    def _verify_db_worker_pair(self, database, worker):
        db_verify = hasattr(database,'get_waveforms')
        worker    = isinstance(worker, Worker)

        if not worker:
            raise TypeError('invalid worker! please supply a valid worker object')

        if not db_verify:
            raise TypeError('invalid database object! please supply a valid database object')

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
        print(os.getcwd())
        if not result:
            print('{}   is not a directory!!'.format(output_directory))
        return result

    def _valid_dirs(self, source_directory_list):
        bad_results = []
        for dir in source_directory_list:
            if not self._valid_dir(dir):
                bad_results.append(1)

        return not bad_results


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

    def correlate(self, source_directory_list, corr_worker, output_directory,format='sac'):
        total_corrs = []
        for source_dir in source_directory_list:

            stream       = read(source_dir)
            correlations = window_correlator(stream, corr_worker)

            if output_directory is not None:
                write_worker(output_directory,correlations,format=format)
            else:
                total_corrs = total_corrs + correlations

        return total_corrs



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


def _process_and_save_window_array(self, directory, format, window_array, single_thread=False,physical_cores_only=True):
        usable_cpus = psutil.cpu_count(logical=physical_cores_only)
        worker_arguments = self.create_worker_args(directory, format, window_array)
        if not single_thread:
            with Pool(usable_cpus) as p:
                result = p.map(process_and_save_to_file, worker_arguments)

        else:
            for arg in worker_arguments:
                process_and_save_to_file(arg)

