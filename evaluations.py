from abc import ABCMeta, abstractmethod

from settings import THRESHOLD_0, THRESHOLD_254, THRESHOLD_192, THRESHOLD_128, height, width, window_height, \
    window_width
import settings

import tensorflow as tf


class Evaluation(object):
    __metaclass__ = ABCMeta

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def evaluate(self, dataset):
        pass


class SimpleEvaluation(Evaluation):
    def __init__(self, model):
        super(SimpleEvaluation, self).__init__(model)
        self.tf_labels = tf.placeholder(tf.float32, [None, 1 if settings.SUPER_PIXEL else window_height * window_width])
        self.guess = tf.placeholder(tf.float32, [None, 1 if settings.SUPER_PIXEL else window_height * window_width])

    def true_positive(self, klass):
        trues = tf.cast(tf.equal(self.tf_labels, klass), tf.float32)
        positives = tf.cast(tf.equal(self.guess, klass), tf.float32)
        true_positives = tf.reduce_sum(tf.mul(positives, trues))
        return true_positives

    def false_positive(self, klass):
        falses = tf.cast(tf.not_equal(self.tf_labels, klass), tf.float32)
        positives = tf.cast(tf.equal(self.guess, klass), tf.float32)
        false_positives = tf.reduce_sum(tf.mul(positives, falses))
        return false_positives

    def false_negative(self, klass):
        trues = tf.cast(tf.equal(self.tf_labels, klass), tf.float32)
        negative = tf.cast(tf.not_equal(self.guess, klass), tf.float32)
        false_negative = tf.reduce_sum(tf.mul(negative, trues))
        return false_negative

    def dice_similarity_coefficient(self, klass):
        tp = self.true_positive(klass)
        fp = self.false_positive(klass)
        fn = self.false_negative(klass)
        return (2 * tp) / (2 * tp + fp + fn)

    def evauluate_dice_similarity(self, dataset):
        sess = tf.Session()
        images, labels = dataset.next_batch()
        if settings.SUPER_PIXEL:
            labels = labels.reshape([labels.size, 1])
        outputs = self.model.test(images)
        if settings.SUPER_PIXEL:
            outputs = outputs.reshape(outputs.size, 1)
        for klass in [0, 1, 2, 3]:
            tf_accuracy = self.dice_similarity_coefficient(klass)
            accuracy = sess.run(tf_accuracy, feed_dict={self.tf_labels: labels, self.guess: outputs})
            print 'accuracy for class %d: %0.4f' % (klass, accuracy)

    def evaluate(self, dataset):
        sess = tf.Session()
        images, labels = dataset.next_batch()
        if settings.SUPER_PIXEL:
            labels = labels.reshape([labels.size, 1])
        outputs = self.model.test(images)
        if settings.SUPER_PIXEL:
            outputs = outputs.reshape(outputs.size, 1)

        # guess = get_index_of_thresholds(_outputs)
        # tf_labels = get_index_of_thresholds(_labels)

        threshold_0 = tf.cast(tf.equal(self.tf_labels, 0), tf.float32)
        threshold_128 = tf.cast(tf.equal(self.tf_labels, 1), tf.float32)
        threshold_192 = tf.cast(tf.equal(self.tf_labels, 2), tf.float32)
        threshold_254 = tf.cast(tf.equal(self.tf_labels, 3), tf.float32)

        guess_0 = tf.cast(tf.equal(self.guess, 0), tf.float32)
        guess_128 = tf.cast(tf.equal(self.guess, 1), tf.float32)
        guess_192 = tf.cast(tf.equal(self.guess, 2), tf.float32)
        guess_254 = tf.cast(tf.equal(self.guess, 3), tf.float32)

        correct_0 = tf.reduce_sum(tf.mul(guess_0, threshold_0))
        correct_128 = tf.reduce_sum(tf.mul(guess_128, threshold_128))
        correct_192 = tf.reduce_sum(tf.mul(guess_192, threshold_192))
        correct_254 = tf.reduce_sum(tf.mul(guess_254, threshold_254))

        accuracy_0 = correct_0 / tf.reduce_sum(threshold_0)
        accuracy_128 = correct_128 / tf.reduce_sum(threshold_128)
        accuracy_192 = correct_192 / tf.reduce_sum(threshold_192)
        accuracy_254 = correct_254 / tf.reduce_sum(threshold_254)

        accuracy = (correct_128 + correct_192 + correct_254) / (tf.reduce_sum(threshold_128) + tf.reduce_sum(
            threshold_192) + tf.reduce_sum(threshold_254))

        print "0 accuracy: %0.4f" % sess.run(accuracy_0, feed_dict={self.tf_labels: labels, self.guess: outputs})
        print "128 accuracy: %0.4f" % sess.run(accuracy_128, feed_dict={self.tf_labels: labels, self.guess: outputs})
        print "192 accuracy: %0.4f" % sess.run(accuracy_192, feed_dict={self.tf_labels: labels, self.guess: outputs})
        print "254 accuracy: %0.4f" % sess.run(accuracy_254, feed_dict={self.tf_labels: labels, self.guess: outputs})
        # accuracy = tf.reduce_mean(tf.cast(tf.equal(guess, tf_labels), tf.float32))
        return sess.run(accuracy, feed_dict={self.tf_labels: labels, self.guess: outputs})


def get_index_of_thresholds(output):
    DATASET_SIZE = 125
    a = tf.abs(output - THRESHOLD_0)
    b = tf.abs(output - THRESHOLD_128)
    c = tf.abs(output - THRESHOLD_192)
    d = tf.abs(output - THRESHOLD_254)
    distances = tf.concat(0, [a, b, c, d])
    distances = tf.reshape(distances, [4, 256 * 256, DATASET_SIZE])

    return tf.argmin(distances, 0)
