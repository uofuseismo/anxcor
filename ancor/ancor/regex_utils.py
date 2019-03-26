import re
import os_utils
import datetime

_standard_datetime_pattern = '%d-%m-%Y_%H:%M:%S'

def create_pattern(delimiters):
    return '|'.join(map(re.escape, delimiters))

def split_string_by_substrings(file, pattern):
    return re.split(pattern, file)

def create_str_from_dt(dt: datetime):
    return dt.strftime(_standard_datetime_pattern)

def create_str_from_dt_float(dt: float):
    return datetime.datetime.fromtimestamp(dt).strftime(_standard_datetime_pattern)

def create_dt_from_str(st):
    return None