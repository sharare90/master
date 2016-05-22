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
        outputs = self.model.test(images)
        for i in outputs:
            for j in i:
                if j != 0:
                    print j
        _outputs = tf.placeholder(tf.float32, [None, 256 * 256])
        _labels = tf.placeholder(tf.float32, [None, 256 * 256])
        guess = get_index_of_thresholds(_outputs)
        tf_labels = get_index_of_thresholds(_labels)
        correct_prediction = tf.equal(guess, tf_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return sess.run(accuracy, feed_dict={_labels: labels, _outputs: outputs})


def get_index_of_thresholds(output):
    a = output - THRESHOLD_0
    b = output - THRESHOLD_128
    c = output - THRESHOLD_192
    d = output - THRESHOLD_254
    distances = tf.concat(0, [a, b, c, d])
    distances = tf.reshape(distances, [4, 20, 256 * 256])

    return tf.argmin(distances, 1)
