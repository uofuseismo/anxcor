import unittest
import os
from datetime import datetime
from obspy import read

from ancor.ancor_file_indexing import FileIndex

good_strings_directory = 'test_data/test_file_connect_windowmap/test_date_strings/test_good_date_strings'
bad_strings_directory  = 'test_data/test_file_connect_windowmap/test_date_strings/test_bad_date_strings'
obspy_dates_directory  = 'test_data/test_file_connect_windowmap/test_obspy_dates/obspy_dates'

def join(one,two):
    return os.path.join(one,two)

def p(target,source):
    string= 'source:\n'
    for key, entry in source.items():
        string = string + '{}:{}\n'.format(key,entry)
    string = string + 'target:\n'
    for key, entry in target.items():
        string = string + '{}:{}\n'.format(key,entry)
    return string

class TestDateStringFMT(unittest.TestCase):

    def source_directory(self,fmt,target_dir):
        datetime_file_directory = {}
        zero_key = []
        for file in os.listdir(target_dir):
            split = file.split('.')
            try:
                dt = datetime.strptime(split[0], fmt)
                if dt not in datetime_file_directory.keys():
                    datetime_file_directory[dt.strftime(fmt)] = []
                datetime_file_directory[dt.strftime(fmt)].append(join(target_dir,file))
            except ValueError:
                zero_key.append(join(target_dir,file))
        datetime_file_directory[0]=zero_key
        return datetime_file_directory

    def test_datestring_fmt2(self):
        target_dir = good_strings_directory + '/date_string_2'
        datetime_format = '%m%d%Y'
        target = self.source_directory(datetime_format,target_dir)
        fileIndex = FileIndex()
        fileIndex.set_directory(target_dir)
        source = fileIndex.get_file_map()
        self.assertEqual(target, source,p(target,source))

    def test_datestring_fmt1(self):
        target_dir = good_strings_directory + '/date_string_1'
        datetime_format = '%Y%m%d'
        target = self.source_directory(datetime_format,target_dir)
        fileIndex = FileIndex()
        fileIndex.set_directory(target_dir)
        fileIndex.set_format(datetime_format)
        source = fileIndex.get_file_map()
        self.assertEqual(target, source, p(target, source))

    def test_bad_datestrings_fmt1(self):
        target_dir = bad_strings_directory + '/bad_date_strings'
        datetime_format = '%Y%m%d'
        target = self.source_directory(datetime_format,target_dir)
        fileIndex = FileIndex()
        fileIndex.set_format(datetime_format)
        fileIndex.method = 'file_datestring'
        fileIndex.set_directory(target_dir)
        source = fileIndex.get_file_map()
        self.assertEqual(target, source, p(target, source))


class TestObspyDates(unittest.TestCase):

    def source_directory(self,fmt,target_dir):
        datetime_file_directory = {}
        for file in os.listdir(target_dir):
            file_total = join(target_dir,file)
            obspy_stream = read(file_total)
            datetime_file_directory[obspy_stream[0].stats.starttime.strftime(fmt)] = [file_total]
        datetime_file_directory[0]=[]
        return datetime_file_directory

    def test_obspy_read(self):
        target_dir = obspy_dates_directory
        datetime_format = '%Y%m%d'
        target = self.source_directory(datetime_format,target_dir)
        fileIndex = FileIndex()
        fileIndex.set_format(datetime_format)
        fileIndex.set_method('obspy_read')
        fileIndex.set_directory(target_dir)
        source = fileIndex.get_file_map()
        self.assertEqual(target, source,p(source,target))




if __name__ == '__main__':
    unittest.main()
