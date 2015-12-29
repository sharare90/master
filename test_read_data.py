__author__ = 'sharare'

from os import listdir
import test_settings



def get_all_files_in_directory(dir_address):
    files = listdir(dir_address)
    return files

def get_image(image_number, column_format=True):
    trains = [x for x in listdir(test_settings.TRAIN_ADDRESS) if x.endswith(test_settings.FILES_SUFFIX)]
    for file_name in trains:
        train_file = open(test_settings.TRAIN_ADDRESS + file_name)
        



