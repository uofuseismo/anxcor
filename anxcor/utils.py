import os
import shutil
try:
    from pympler import asizeof
    objsize = asizeof.asizeof
except Exception:
    def objsize(obj):
        return 0
import sys
from numbers import Number
from collections import Set, Mapping, deque

zero_depth_bases = (str, bytes, Number, range, bytearray)
iteritems = 'items'
sep = os.sep
time_format = '"D%d_M%m_Y%Y__H%H:M%M:S%S"'

def join(one,two):
    return os.path.join(one,two)


def make_path_from_list(path_list):
    path = os.path.join(*path_list)
    make_dir(path)
    return path

def make_dir(path):
    if not folder_exists(path):
        os.makedirs(path)

def get_filelist(directory):
    listOfFiles = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    return listOfFiles


def get_files_with_extensions(list_of_files,extensions):
    list_of_valid_files = []
    for potential_file in list_of_files:
        if potential_file.lower().endswith(extensions.lower()):
            list_of_valid_files.append(potential_file)
    return list_of_valid_files

def create_workingdir(*args,fail_if_exists=True):
    args = list(args)
    dir = args.pop(0)
    for arg in args:
        dir = dir + sep + arg
    try:
        os.makedirs(dir,exist_ok=fail_if_exists)
    except FileExistsError:
        if fail_if_exists:
            raise FileExistsError
    return dir

def delete_dirs(dirs):
    shutil.rmtree(dirs)

def delete_file(file):
    os.remove(file)

def file_exists(file):
    from os import path
    return path.exists(file) and path.isfile(file)

def folder_exists(folder):
    return os.path.isdir(folder)

def _clean_dirs_of_index(dirlist):
    for dir in dirlist:
        try:
            os.remove(dir + './index.h5')
        except Exception:
            pass

def _clean_files_in_dir(dir):
    filelist = get_filelist(dir)
    for dir in filelist:
        try:
            os.remove(dir)
        except Exception:
            pass

def _clean_dirs_and_files(dirlist):
    for dir in dirlist:
        try:
            os.remove(dir)
        except Exception:
            pass
        try:
            shutil.rmtree(dir)
        except Exception:
            pass

def _how_many_fmt(directory,format='.sac'):
    filelist = get_filelist(directory)
    saclist  = get_files_with_extensions(filelist, format)
    return len(saclist)

def get_folderpath(filepath):

    for i in range(len(filepath)-1,-1,-1):
        if filepath[i]==os.sep:
            folder = filepath[:i]
            return folder

def getsize(obj_0):
    """Recursively iterate to sum size of object & members."""
    size = objsize(obj_0)
    if size < 1e9:
        mb = size / 1e6
        string = '{} MB'.format(mb)
    else:
        gb = size/1e9
        string = '{} GB'.format(gb)

    return string