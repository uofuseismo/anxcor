
from obspy import  read
import re
from datetime import datetime
import os

def join(one,two):
    return os.path.join(one,two)

class FileIndex:

    valid_extension = ['.sac']
    def __init__(self):
        self.source_directory = None
        self.file_map  = {}
        self.method   = 'file_datestring'
        self.fmt = '%m%d%Y'

    @property
    def source_directory(self):
        return self.__directory

    @property
    def fmt(self):
        return self.__fmt

    @fmt.setter
    def fmt(self, fmt):
        self.__fmt = fmt

    @source_directory.setter
    def source_directory(self, directory):
        self.__directory = directory

    @source_directory.getter
    def source_directory(self):
        return self.__directory

    def _create_map(self):
        list_of_files = self.get_list_of_files()
        list_of_valid_files = self.get_valid_files(list_of_files)
        self.parse_files(list_of_valid_files)

    def parse_files(self, list_of_valid_files):
        for potential_file in list_of_valid_files:
            date_key = self.get_date_key(potential_file)
            self._add_file(date_key, potential_file)

    def _add_file(self,key,file_path):
        if key not in self.file_map.keys():
            self.file_map[key]=[]
        self.file_map[key].append(file_path)

    def get_valid_files(self, list_of_files):
        list_of_valid_files = []
        for potential_file in list_of_files:
            if potential_file.lower().endswith(tuple(self.valid_extension)):
                list_of_valid_files.append(potential_file)
        return list_of_valid_files

    def get_list_of_files(self):
        listOfFiles = []
        for (dirpath, dirnames, filenames) in os.walk(self.source_directory):
            listOfFiles += [os.path.join(dirpath, file) for file in filenames]
        return listOfFiles

    def get_file_map(self):
        self.file_map[0] = []
        self._create_map()
        return {**self.file_map}

    def get_date_key(self, potential_file):
        if self.method is 'file_datestring':
            return self.file_string_method(potential_file)
        else:
            return self._obspy_datestring_method(potential_file)

    def file_string_method(self, potential_file):
        delimiters = os.sep , '.'
        regexPattern = '|'.join(map(re.escape, delimiters))
        split_file = re.split(regexPattern, potential_file)
        for portion in split_file:
            if portion.isdigit():
                try:
                    dt = datetime.strptime(portion, self.fmt)
                    return dt.strftime(self.fmt)
                except ValueError:
                    pass
        return 0

    def _obspy_datestring_method(self, potential_file):
        stream = read(potential_file)
        date = stream.traces[0].stats.starttime.strftime(self.fmt)
        return date


class FileIndexTest(FileIndex):

    def __init__(self):
        super().__init__()
