from os import listdir
import numpy as np

import settings


def get_all_files_in_directory(dir_address):
    files = listdir(dir_address)
    return files


def get_file(image_number, column_format=True):
    """
    Returns an image with its label based on 'image_number'
    """
    headers = [x for x in listdir(settings.LABELS_ADDRESS) if x.endswith(settings.HEADER_FILES_SUFFIX)]
    for file_name in headers:
        header_file = open(settings.LABELS_ADDRESS + file_name)
        number_of_slices = int(header_file.readline().split()[2])
        if image_number < number_of_slices:
            header_file.close()
            buchar_file_name = settings.LABELS_ADDRESS + file_name.replace(settings.HEADER_FILES_SUFFIX,
                                                                           settings.BUCHAR_FILES_SUFFIX)
            label = read_buchar_file(buchar_file_name, image_number, number_of_slices)
            offsets = read_offsets()
            img_name = buchar_file_name[buchar_file_name.index('/') + 1:buchar_file_name.index('.')]
            folder_name = img_name + '/'
            file_name = img_name + '_' + str(image_number + offsets[img_name] + 2) + settings.IMAGE_FILES_SUFFIX
            img_address = settings.IMAGES_ADDRESS + folder_name + file_name
            img = read_image_file(img_address, column_format=column_format)
            return img, label
        else:
            image_number -= number_of_slices
            header_file.close()


def read_buchar_file(input_filename, slice_number, number_of_slices):
    """
    The image labels in IBSR are buchar files with 256 * 256 size
    this function reads a buchar file.
    """
    dtype = np.dtype('>u1')  # big-endian unsigned integer (8bit)

    # Reading.
    fid = open(input_filename, 'rb')
    data = np.fromfile(fid, dtype)
    image = data.reshape(number_of_slices, 256 * 256)
    image = image[slice_number, :]
    return image


def read_image_file(input_filename, column_format=False):
    """
    Reads the image data from file
    """
    dtype = np.dtype('>u2')  # big-endian unsigned integer (16bit)
    # Reading.
    fid = open(input_filename, 'rb')
    image = np.fromfile(fid, dtype)
    if not column_format:
        image = image.reshape(256, 256)
        return image
    return image


def read_offsets():
    offsets = {}
    f = open(settings.OFFSETS_ADDRESS)
    for line in f:
        words = line.split()
        key = (words[0])
        offset = int(words[1])
        offsets[key] = offset
    return offsets


def get_number_of_images():
    headers = [x for x in listdir(settings.LABELS_ADDRESS) if x.endswith(settings.HEADER_FILES_SUFFIX)]
    number_of_images = 0
    for header_file in headers:
        f = open(settings.LABELS_ADDRESS + header_file)
        number_of_images += int(f.readline().split()[2])
    return number_of_images


