from abc import ABCMeta, abstractmethod

from settings import THRESHOLD_0, THRESHOLD_254, THRESHOLD_192, THRESHOLD_128

import tensorflow as tf


class Evaluation(object):
    __metaclass__ = ABCMeta

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def evaluate(self, test_set, test_labels, test_image_numbers):
        pass


class SimpleEvaluation(Evaluation):
    def evaluate(self, dataset):
        sess = tf.Session()
        images, labels = dataset.next_batch()
        output = self.model.test(images)
        guess = get_index_of_thresholds(output)
        _labels = tf.placeholder(tf.float32, [None])
        tf_labels = get_index_of_thresholds(_labels)
        correct_prediction = tf.equal(guess, tf_labels)
        accuracy = tf.reduce_mean(correct_prediction)
        print(sess.run(accuracy, feed_dict={_labels: labels}))


def get_index_of_thresholds(output):
    a = output - THRESHOLD_0
    b = output - THRESHOLD_128
    c = output - THRESHOLD_192
    d = output - THRESHOLD_254
    distances = tf.concat(0, [a, b, c, d])
    size_of_output = 256 * 256
    distances = tf.reshape(distances, [4, size_of_output])

    return tf.argmin(distances, 0)
