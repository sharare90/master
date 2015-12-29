from os import listdir
from random import sample

import numpy as np

import settings
from settings import DECIMAL_POINT_ROUND


def get_all_files_in_directory(dir_address):
    files = listdir(dir_address)
    return files


def get_file(image_number, column_format=True):
    headers = [x for x in listdir(settings.LABELS_ADDRESS) if x.endswith(settings.HEADER_FILES_SUFFIX)]
    for file_name in headers:
        header_file = open(settings.LABELS_ADDRESS + file_name)
        number_of_slices = int(header_file.readline().split()[2])
        if image_number < number_of_slices:
            header_file.close()
            buchar_file_name = settings.LABELS_ADDRESS + file_name.replace(settings.HEADER_FILES_SUFFIX,
                                                                           settings.BUCHAR_FILES_SUFFIX)
            label = read_buchar_file(buchar_file_name, image_number, number_of_slices, column_format=column_format)
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


def read_buchar_file(input_filename, slice_number, number_of_slices, column_format=False):
    dtype = np.dtype('>u1')  # big-endian unsigned integer (8bit)

    # Reading.
    fid = open(input_filename, 'rb')
    data = np.fromfile(fid, dtype)
    if column_format:
        # This reshaping is the original image in column format
        image = data.reshape(number_of_slices, 256 * 256)
        image = image[slice_number, :]
        image = discretization(image)
        image = np.matrix(image)
    else:
        image = data.reshape(number_of_slices, 256, 256)
        image = image[slice_number, :, :]
    return image


def read_image_file(input_filename, column_format=False):
    dtype = np.dtype('>u2')  # big-endian unsigned integer (16bit)
    # Reading.
    fid = open(input_filename, 'rb')
    image = np.fromfile(fid, dtype)
    if not column_format:
        image = image.reshape(256, 256)
        return image
    image = normalization(image)
    image = np.matrix(image).transpose()
    return image


def normalization(image):
    max_image = image.max()
    min_image = image.min()
    image = (image - min_image) / float(max_image - min_image)
    # image = np.round(image, DECIMAL_POINT_ROUND)
    return image


def discretization(label):
    discretized_label = -1 * np.ones(shape=(len(label) * 4, 1))
    for i in xrange(len(label)):
        if label[i] == 0:
            discretized_label[4 * i] = 1
        elif label[i] == 128:
            discretized_label[4 * i + 1] = 1
        elif label[i] == 192:
            discretized_label[4 * i + 2] = 1
        else:
            discretized_label[4 * i + 3] = 1
    return discretized_label


def polarization(label):
    label = label.astype(float)
    for i in xrange(len(label)):
        if label[i] == 254:
            label[i] = 1
        elif label[i] == 192:
            label[i] = -1
        elif label[i] == 128:
            label[i] = 0.5
    return label


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


def random_select(ratio):
    """Selects a part of data for train and another part for test randomly
    then return 4 objects. train_set, test_set, train_labels, test_labels"""
    train_set = []
    test_set = []
    train_labels = []
    test_labels = []
    number_of_images = get_number_of_images()
    test_image_numbers = range(number_of_images)
    train_image_numbers = sample(test_image_numbers, int(ratio * number_of_images))
    for i in train_image_numbers:
        img, label = get_file(i)
        train_set.append(img)
        train_labels.append(label)
        test_image_numbers.remove(i)

    for i in test_image_numbers:
        test_img, test_lbl = get_file(i)
        test_set.append(test_img)
        test_labels.append(test_lbl)

    return train_set, test_set, train_labels, test_labels, train_image_numbers, test_image_numbers
