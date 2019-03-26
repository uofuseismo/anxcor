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

def create_workingdir(working_dir,extension):
    dir = working_dir + sep + extension
    os.mkdir(dir)
    return dir

def delete_dirs(dir):
    shutil.rmtree(dir)