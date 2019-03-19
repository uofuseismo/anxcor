from ancor_file_indexing import FileIndex
import os_utils
from datetime import datetime
from datetime import timedelta

class WindowMap:


    def __init__(self, file_index = None, window_length=10, overlap_percent=0.0):
        self.file_index      = file_index
        self.window_length   = window_length
        self.overlap_percent = overlap_percent

    @property
    def window_length(self):
        return self.__window_length

    @property
    def overlap_percent(self):
        return self.__overlap_percent

    @window_length.setter
    def window_length(self,window_length):
        self.__window_length = window_length

    @window_length.getter
    def window_length(self):
        return timedelta(seconds=self.__window_length)

    @overlap_percent.setter
    def overlap_percent(self,overlap_percent):
        self.__overlap_percent = overlap_percent

    @overlap_percent.getter
    def overlap_percent(self):
        return self.__overlap_percent


    def get_timespan(self):
        return self._extract_minmax_dates()

    def map_jobs_to_files(self):
        datekey_map = self._get_datekey_map()
        wavebank_sourcefile_list = []

        for key, windows in datekey_map.items():
            wavebank_sourcefile  = []

            for keydate in key:
                files_for_date      = self.file_index.get_file_list(keydate,create_new_map=False)
                wavebank_sourcefile = wavebank_sourcefile + files_for_date
            unraveled_window = [val for sublist in windows for val in sublist]
            wavebank_sourcefile_list.append({
                'files'    : wavebank_sourcefile_list,
                'windows'  : windows,
                'max time'     : max(unraveled_window),
                'min time'     : min(unraveled_window)
            })
        return wavebank_sourcefile_list

    def _get_datekey_map(self):
        window_list = self._window_list()
        key_map = {}
        format  = self.file_index.get_format()
        for starttime, endtime in window_list:
            start_key = starttime.strftime(format)
            end_key   = endtime.strftime(format)

            if start_key==end_key:
                key = (start_key,)
            else:
                key = (start_key, end_key)

            if key not in key_map.keys():
                key_map[key]=[]
            key_map[key].append((starttime, endtime))

        return key_map


    def _window_list(self):
        max_date, min_date     = self._extract_minmax_dates()
        delta, delta_increment = self._get_time_iterables()

        start_time = min_date
        date_list = []
        while start_time < max_date:
            time_range = (start_time, start_time+delta)
            date_list.append(time_range)
            start_time+=delta_increment

        return date_list

    def _get_time_iterables(self):
        delta = self.window_length
        delta_increment = delta * (1-self.overlap_percent)
        delta_increment = timedelta(seconds=int(delta_increment.total_seconds()))
        delta = timedelta(seconds=int(delta.total_seconds()))
        return delta, delta_increment

    def _extract_minmax_dates(self):
        keys   = list(self.file_index.get_keys())
        if '0' in keys: keys.remove('0')
        format = self.file_index.get_format()
        datetime_list = list(map(lambda x: datetime.strptime(str(x), format), keys))
        max_date = max(datetime_list) + timedelta(days=1)
        min_date = min(datetime_list)
        return max_date, min_date


class AncorBankGenerator:

    def __init__(self,working_directory,**kwargs):
        self.working_directory = working_directory
        self.window_map = WindowMap(**kwargs)

    def get_jobs(self):
        mapped_job = self.window_map.map_jobs()
        return mapped_job

    def set_window_length(self,window_length):
        self.window_map.window_length = window_length

    def set_overlap_percent(self, overlap_percent):
        self.window_map.overlap_percent = overlap_percent

    def generate_job(self,**kwargs):
        jobs = self.get_jobs()
        for job in jobs:
            ancorbank = AncorBank(job,self.working_directory,**kwargs)
            yield ancorbank


class AncorBank:

    def __init__(self,job_data,working_directory, clean_after_use=True):
        window_number  = job_data['window_number']
        self.directory = working_directory + os_utils.sep + str(working_directory)
        self.windows   = job_data['window_list']
        self.files     = job_data['file_list']
        self.clean_after_use = clean_after_use