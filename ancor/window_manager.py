from obspy.core.utcdatetime import UTCDateTime
from multiprocessing import Pool
import os
from worker_factory import Worker
import psutil
import copy
import numpy as np
from stream_jobs import process_and_save_to_file




class WindowManager:

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
            maximum number of windows to process.py
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
            for database, worker in self.databases:
                worker_arguments.append(
                    (window_number, directory, format, starttime, endtime, copy.deepcopy(worker), copy.deepcopy(database)))

            window_number+=1

        return worker_arguments
