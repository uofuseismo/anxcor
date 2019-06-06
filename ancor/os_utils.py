import os
import shutil

sep = os.sep
time_format = '"D%d_M%m_Y%Y__H%H:M%M:S%S"'

def join(one,two):
    return os.path.join(one,two)


def get_filelist(directory):
    listOfFiles = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    return listOfFiles


def get_files_with_extensions(list_of_files,extensions):
    list_of_valid_files = []
    for potential_file in list_of_files:
        if potential_file.lower().endswith(tuple(extensions)):
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