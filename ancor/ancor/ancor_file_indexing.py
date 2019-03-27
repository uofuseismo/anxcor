"""
Let it be known that this piece of code was written while standing vigil for Emma and Joe Sánchez.
Both lived extremely hard lives in simple poverty, and performed extraordinary feats of service to their four
children and three grandchildren.
"""
from obspy import  read
import regex_utils
from datetime import datetime
import os_utils

class _FileIndexData:
    valid_extension = ['.sac']
    _valid_methods=['file_datestring','obspy_read']
    def __init__(self,directory=None,method='file_datestring',format='%m%d%y'):
        """

        Parameters
        ----------
        directory: str
            directory of the source files

        method: {'file_datestring','obspy_read}, optional
            one of the valid date parsing methods

        format: str, optional
            a datetime format string used to parse the files
        """
        self.file_map = {}
        self._directory = directory
        self._should_reset = True
        self._method = method
        self._fmt    = format

    def get_directory(self):
        return self._directory

    def set_directory(self, directory):
        self._directory = directory

    def get_method(self):
        return self._method

    def set_method(self, method):
        if method in self._valid_methods:
            self._method = method
        else:
            print('Method {} does not exist!! Please refer to the documentation'.format(method))

    def set_format(self, format):
        self._fmt = format

    def get_format(self):
        return self._fmt



class FileIndex(_FileIndexData):
    method_docstring = "Method 1: \n"+ \
               "\t 'file_datestring'\n"+\
               "-------------------------"+\
               "A method that uses a sequence of integers in the \n"+\
               "file name to parse the possible seismic files by date."+\
               "the integer date must be in the format %x%x%x (only year, month and date allowed) and separated \n"+\
               "from other strings in the file name by either '\' or '\.' \n\n"+\
                "Method 2: \n" + \
                "\t 'obspy_load'\n" + \
                "-------------------------" + \
                "A method that attemps to use obspy to get the year, month and date of each readable seismic file. \n" + \
                "This methos is incredibly slow, but can use non %x%x%x string formatted files."

    valid_extension = ['.sac']

    def __init__(self, directory: str = None, method: str = 'file_datestring', format: str = '%m%d%y'):
        """

        Parameters
        ----------
        directory: str
            directory of the source files

        method: str {'file_datestring','obspy_read}, optional
            one of the valid date parsing methods. can be either 'file_datestring' if the date
            is contained inside the file name, or 'obspy_read' if you want to read the metadata in the file

        format: str, optional
            a datetime format string used to parse the files
        """
        super().__init__(directory=directory,method=method,format=format)
        self._has_indexed = False


    def _create_map(self):
        directory = self.get_directory()
        list_of_files       = os_utils.get_filelist(directory)
        list_of_valid_files = os_utils.get_files_with_extensions(list_of_files, self.valid_extension)
        self._parse_files(list_of_valid_files)

    def _parse_files(self, list_of_valid_files):
        for potential_file in list_of_valid_files:
            date_key = self._get_date_key(potential_file)
            self._add_file(date_key, potential_file)

    def _add_file(self,key,file_path):
        if key not in self.file_map.keys():
            self.file_map[key]=[]
        self.file_map[key].append(file_path)


    def get_file_map(self,create_new_map=True,**kwargs):
        """

        Parameters
        ----------
            create_new_map:

        Returns
        -------

        """
        if create_new_map:
            self.file_map['0'] = []
            self._create_map()
        elif create_new_map is False and self._has_indexed is False:
            self.file_map['0'] = []
            self._create_map()
            self._has_indexed=True

        return {**self.file_map}

    def get_keys(self,create_new_map=False):
        """
        get a list of integers which act as the file keys. Method #get_file_map() specifies kwargs

        Returns
        -------
            key_list
                a set of keys corresponding to

        """
        file_map = self.get_file_map(create_new_map=create_new_map)
        return file_map.keys()

    def _get_date_key(self, potential_file):
        if self.get_method() is self._valid_methods[0]:
            return self._file_string_method(potential_file)
        elif self.get_method() is self._valid_methods[1]:
            return self._obspy_datestring_method(potential_file)
        else:
            raise NotImplementedError('the selected date parsing method does not exist')

    def _file_string_method(self, potential_file):
        delimiters = os_utils.sep , '.'
        pattern    = regex_utils.create_pattern(delimiters)
        split_file = regex_utils.split_string_by_substrings(potential_file, pattern)
        for portion in split_file:
            if portion.isdigit():
                try:
                    dt = datetime.strptime(portion, self.get_format())
                    return dt.strftime(self.get_format())
                except ValueError:
                    pass
        return 0

    def _obspy_datestring_method(self, potential_file):
        stream = read(potential_file)
        date = stream.traces[0].stats.starttime.strftime(self.get_format())
        return date

    def get_file_list(self, keydate, **kwargs):
        map = self.get_file_map(**kwargs)
        if keydate in map.keys():
            return map[keydate]
        else:
            return []


class FileIndexTest(FileIndex):

    def __init__(self):
        super().__init__()

    def get_file_map(self,**kwargs):
        return {
            0: read().traces[0],
            1: read().traces[1],
            2: read().traces[2]


        }
