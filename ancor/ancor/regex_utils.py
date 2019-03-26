import re
import os_utils

def create_pattern(delimiters):
    return '|'.join(map(re.escape, delimiters))

def split_string_by_substrings(file, pattern):
    return re.split(pattern, file)