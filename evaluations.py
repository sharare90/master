from abc import ABCMeta, abstractmethod


__author__ = 'sharare'


class Evaluation(object):
    __metaclass__ = ABCMeta

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def evaluate(self, test_set, test_labels, test_image_numbers):
        pass


class SimpleEvaluation(Evaluation):
    def evaluate(self, test_set, test_labels, test_image_numbers):
        import tensorflow as tf
        from read_data import discretization
        # sess = tf.Session()
        # sess.run(tf.initialize_all_variables())
        # y_ = tf.placeholder(tf.float32, [1, self.model.layers_size[-1]])
        # sum_evaluation = 0
        for i in range(len(test_set)):
            guess = self.model.test(test_set[i])
            # correct_prediction = tf.equal(tf.argmax(tf.cast(self.model.test(test_set[i].transpose()), tf.float32), 1), tf.argmax(y_, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # similarity = sess.run(accuracy, feed_dict={y_: discretization(test_labels[i].reshape(256 * 256, 1)).transpose()})
            # sum_evaluation += similarity(self.model.test(test_set[i]), test_labels[i], consider_black_points=False)
            # sum_evaluation += similarity
        # acc = sum_evaluation / float(len(test_set))
        # return acc
