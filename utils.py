from math import e

import numpy as np
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
    return np.matrix(np.random.rand(n, m))


def similarity(a, b):
    counter = 0
    c = a == b
    for i in range(c.size):
        if c[i, 0]:
            counter += 1
    return float(counter) / float(a.size)


def maximization(output):
    for i in xrange(len(output) / 4):
        max_val = -2
        max_index = -1
        for j in [0, 1, 2, 3]:
            if output[4 * i + j] > max_val:
                max_val = output[4 * i + j]
                max_index = j
                output[4 * i + j] = -1
        output[4 * i + max_index] = 1
    return output
