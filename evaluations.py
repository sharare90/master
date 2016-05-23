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
        _outputs = tf.placeholder(tf.float32, [None, 256 * 256])
        _labels = tf.placeholder(tf.float32, [None, 256 * 256])
        guess = get_index_of_thresholds(_outputs)
        tf_labels = get_index_of_thresholds(_labels)
        correct_prediction = tf.equal(guess, tf_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        threshold_0 = tf.cast(tf.equal(tf_labels, 0), tf.int32)
        threshold_128 = tf.cast(tf.equal(tf_labels, 1), tf.int32)
        threshold_192 = tf.cast(tf.equal(tf_labels, 2), tf.int32)
        threshold_254 = tf.cast(tf.equal(tf_labels, 3), tf.int32)

        guess_0 = tf.cast(tf.equal(guess, 0), tf.int32)
        guess_128 = tf.cast(tf.equal(guess, 1), tf.int32)
        guess_192 = tf.cast(tf.equal(guess, 2), tf.int32)
        guess_254 = tf.cast(tf.equal(guess, 3), tf.int32)

        accuracy_0 = tf.reduce_sum(tf.mul(guess_0, threshold_0)) / tf.reduce_sum(threshold_0)
        accuracy_128 = tf.reduce_sum(tf.mul(guess_128, threshold_128)) / tf.reduce_sum(threshold_128)
        accuracy_192 = tf.reduce_sum(tf.mul(guess_192, threshold_192)) / tf.reduce_sum(threshold_192)
        accuracy_254 = tf.reduce_sum(tf.mul(guess_254, threshold_254)) / tf.reduce_sum(threshold_254)

        print "0 accuracy: %0.4f" % sess.run(accuracy_0, feed_dict={_labels: labels, _outputs: outputs})
        print "128 accuracy: %0.4f" % sess.run(accuracy_128, feed_dict={_labels: labels, _outputs: outputs})
        print "192 accuracy: %0.4f" % sess.run(accuracy_192, feed_dict={_labels: labels, _outputs: outputs})
        print "254 accuracy: %0.4f" % sess.run(accuracy_254, feed_dict={_labels: labels, _outputs: outputs})
        return sess.run(accuracy, feed_dict={_labels: labels, _outputs: outputs})


def get_index_of_thresholds(output):
    DATASET_SIZE = 125
    a = tf.abs(output - THRESHOLD_0)
    b = tf.abs(output - THRESHOLD_128)
    c = tf.abs(output - THRESHOLD_192)
    d = tf.abs(output - THRESHOLD_254)
    distances = tf.concat(0, [a, b, c, d])
    distances = tf.reshape(distances, [4, DATASET_SIZE, 256 * 256])

    return tf.argmin(distances, 0)
