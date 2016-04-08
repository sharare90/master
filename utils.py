from math import e

import tensorflow as tf

import numpy as np

from read_data import post_process

from settings import DECIMAL_POINT_ROUND, THRESHOLD_128, THRESHOLD_192, THRESHOLD_0

__author__ = 'sharare'


def sigmoid(num, output_polarization=True):
    num = np.asarray(num)
    output = np.matrix(1 / (1 + e ** (-num)))
    if output_polarization:
        output = 2 * output - 1
    return output


def sigmoid_derivative(num, output_polarization=True):
    f_x = sigmoid(num)
    f_x = np.asarray(f_x)
    if output_polarization:
        return np.matrix(0.5 * (1 + f_x) * (1 - f_x))
    return np.matrix(f_x * (1 - f_x))


def create_random_weights(n, m):
    if m == 1:
        return tf.Variable(tf.zeros([n]))
    return tf.Variable(tf.zeros([n, m]))

def similarity(guess_label, label, consider_black_points=True):
    counter = 0
    guess_label = post_process(guess_label)
    c = guess_label == label
    if consider_black_points:
        num_all_pixels = float(guess_label.size)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                if c[i, j]:
                    counter += 1
    else:
        num_all_pixels = 0
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                if label[i, j] != 0:
                    if c[i, j]:
                        counter += 1
                    num_all_pixels += 1
                elif not c[i, j]:
                    # TODO consider what happens if model guesses incorrectly about a pixel which is black in label
                    pass
    return float(counter) / num_all_pixels


def maximization(output):
    for i in xrange(len(output) / 4):
        max_val = -2
        max_index = -1
        for j in [0, 1, 2, 3]:
            if output[4 * i + j][0, 0] > max_val:
                max_val = output[4 * i + j][0, 0]
                max_index = j
            output[4 * i + j] = -1
        output[4 * i + max_index] = 1
    return output

# def maximization(output):
#     max_val = -2
#     max_index = -1
#     for i in xrange(len(output)):
#         if output[i][0, 0] > max_val:
#             max_val = output[i][0, 0]
#             max_index = i
#         output[i] = -1
#     output[max_index] = 1
#     return output
