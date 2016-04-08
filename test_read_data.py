import numpy as np
from os import listdir
from display import display
import test_settings
from models import DeepBeliefNetwork, NeuralNetwork
from evaluations import SimpleEvaluation

__author__ = 'sharare'

DISPLAY_NEURAL_NETWORK_SEGMENTATION = True


def get_all_files_in_directory(dir_address):
    files = listdir(dir_address)
    return files


def get_text(text_number, address, column_format=True):
    files = [x for x in listdir(address) if x.endswith(test_settings.FILES_SUFFIX)]
    img = np.zeros((test_settings.NUMBER_OF_LINES, test_settings.NUMBER_OF_CHARACTERS_IN_A_LINE))
    lbl = -1 * np.ones((test_settings.LABEL_SIZE, 1))
    if text_number < len(files):
        train_file = open(address + files[text_number])
        for i in xrange(test_settings.NUMBER_OF_LINES):
            line = train_file.readline()
            for j in xrange(test_settings.NUMBER_OF_CHARACTERS_IN_A_LINE):
                character = line[j]
                if character == ".":
                    img[i, j] = -1
                elif character == "#":
                    img[i, j] = 1
        train_file.close()
        lbl_index = text_number / 3
        lbl[lbl_index] = 1
        img = img.reshape(63, 1)
        return img, lbl
    raise Exception()


def number_of_text_files():
    trains = [x for x in listdir(test_settings.TRAIN_ADDRESS) if x.endswith(test_settings.FILES_SUFFIX)]
    number_of_train_files = len(trains)
    return number_of_train_files


if __name__ == "__main__":
    train_set = []
    train_labels = []
    train_image_numbers = []
    test_set = []
    test_labels = []
    test_image_numbers = []
    layers = [9 * 7, 6, 7]
    learning_rate = 0.03
    iteration_number = 100
    neural_network = NeuralNetwork(layers, learning_rate)
    for i in xrange(test_settings.NUMBER_OF_TRAIN_IMAGES):
        img, lbl = get_text(i, test_settings.TRAIN_ADDRESS)
        test_img, test_lbl = get_text(i, test_settings.TEST_ADDRESS)
        train_set.append(img)
        train_labels.append(lbl)
        train_image_numbers.append(i)
        test_set.append(test_img)
        test_labels.append(test_lbl)
        test_image_numbers.append(i)
    neural_network.train(train_set, train_labels, train_image_numbers, iteration_number)
    simple_evaluation = SimpleEvaluation(neural_network)
    accuracy = simple_evaluation.evaluate(test_set, test_labels, test_image_numbers)
    print accuracy
    train_accuracy = simple_evaluation.evaluate(train_set, train_labels, train_image_numbers)
    print train_accuracy
