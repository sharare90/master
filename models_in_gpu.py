import pycuda.driver as cuda
import tensorflow as tf
import math

from abc import ABCMeta, abstractmethod

import numpy as np

from utils import sigmoid, sigmoid_derivative, create_random_weights, maximization

__author__ = 'sharare'


class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, dataset, iteration_number):
        pass

    @abstractmethod
    def test(self, img):
        pass


class Layer(object):
    def __init__(self, number_of_neurons):
        self.input = None
        self.output = None
        self.n = number_of_neurons


class NeuralNetwork(Model):
    def __init__(self, layers_size, learning_rate, initial_weights=None, initial_bias=None, layers=None):
        self.layers_size = layers_size
        self.layers = []
        for i in xrange(len(layers_size)):
            l = Layer(layers_size[i])
            self.layers.append(l)

        self.learning_rate = learning_rate
        if initial_weights is None:
            self.weights = self.create_initial_weights()
        else:
            self.weights = initial_weights

        if initial_bias is None:
            self.bias = self.create_initial_bias()
        else:
            self.bias = initial_bias
        self.sess = None

    def train(self, dataset, iteration_number):
        y_ = tf.placeholder(tf.float32, [None, self.layers_size[-1]])
        self.layers[0].input = tf.placeholder(tf.float32, [None, self.layers_size[0]])
        self.layers[0].output = tf.nn.softmax(self.layers[0].input)
        for i in xrange(len(self.layers) - 1):
            self.layers[i + 1].input = tf.matmul(self.layers[i].output, self.weights[i]) + self.bias[i]
            self.layers[i + 1].output = tf.nn.softmax(self.layers[i + 1].input)
        # loss_function = -tf.reduce_sum(y_ * tf.log(self.layers[-1].output))
        loss_function = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(self.layers[-1].output, 1e-10, 1.0)))
        train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss_function)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        for iteration in xrange(iteration_number):
            images, labels = dataset.next_batch()
            self.sess.run(train_step, feed_dict={self.layers[0].input: images, y_: labels})

    def test(self, imgs):
        output = self.layers[-1].output
        return self.sess.run(output, feed_dict={self.layers[0].input: imgs})

    def create_initial_weights(self):
        weights = []
        for i in xrange(len(self.layers_size) - 1):
            layer_nodes = self.layers_size[i]
            next_layer_nodes = self.layers_size[i + 1]
            weights.append(create_random_weights(layer_nodes, next_layer_nodes))
        return weights

    def create_initial_bias(self):
        bias_weights = []
        for i in xrange(len(self.layers_size) - 1):
            weights = create_random_weights(self.layers_size[i + 1], 1)
            bias_weights.append(weights)
        return bias_weights


class RestrictedBoltzmanMachine(NeuralNetwork):
    """ represents a sigmoidal rbm """

    def __init__(self, name, input_size, output_size):
        with tf.name_scope("rbm_" + name):
            self.weights = tf.Variable(
                tf.truncated_normal([input_size, output_size],
                                    stddev=1.0 / math.sqrt(float(input_size))), name="weights")
            self.v_bias = tf.Variable(tf.zeros([input_size]), name="v_bias")
            self.h_bias = tf.Variable(tf.zeros([output_size]), name="h_bias")

    def propup(self, visible):
        """ P(h|v) """
        return tf.nn.sigmoid(tf.matmul(visible, self.weights) + self.h_bias)

    def propdown(self, hidden):
        """ P(v|h) """
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.weights)) + self.v_bias)

    def sample_h_given_v(self, v_sample):
        """ Generate a sample from the hidden layer """
        return sample_prob(self.propup(v_sample))

    def sample_v_given_h(self, h_sample):
        """ Generate a sample from the visible layer """
        return sample_prob(self.propdown(h_sample))

    def gibbs_hvh(self, h0_sample):
        """ A gibbs step starting from the hidden layer """
        v_sample = self.sample_v_given_h(h0_sample)
        h_sample = self.sample_h_given_v(v_sample)
        return [v_sample, h_sample]

    def gibbs_vhv(self, v0_sample):
        """ A gibbs step starting from the visible layer """
        h_sample = self.sample_h_given_v(v0_sample)
        v_sample = self.sample_v_given_h(h_sample)
        return [h_sample, v_sample]

    def cd1(self, visibles, learning_rate=0.1):
        " One step of contrastive divergence, with Rao-Blackwellization "
        h_start = self.propup(visibles)
        v_end = self.propdown(h_start)
        h_end = self.propup(v_end)
        w_positive_grad = tf.matmul(tf.transpose(visibles), h_start)
        w_negative_grad = tf.matmul(tf.transpose(v_end), h_end)

        update_w = self.weights.assign_add(learning_rate * (w_positive_grad - w_negative_grad))

        update_vb = self.v_bias.assign_add(learning_rate * tf.reduce_mean(visibles - v_end, 0))

        update_hb = self.h_bias.assign_add(learning_rate * tf.reduce_mean(h_start - h_end, 0))

        return [update_w, update_vb, update_hb]

    def reconstruction_error(self, dataset):
        """ The reconstruction cost for the whole dataset """
        err = tf.stop_gradient(dataset - self.gibbs_vhv(dataset)[1])
        return tf.reduce_sum(err * err)


def sample_prob(probs):
    """Takes a tensor of probabilities (as from a sigmoidal activation)
       and samples from all the distributions"""
    return tf.nn.relu(
        tf.sign(
            probs - tf.random_uniform(probs.get_shape())))


class DeepBeliefNetwork(NeuralNetwork):
    def train(self, dataset, iteration_number):
        batch_size = dataset.batch_size
        dataset.batch_size = dataset.count()

        input_images = dataset.next_batch()[0]
        input_images = tf.cast(input_images, tf.float32)
        for i in range(len(self.layers) - 1):
            rbm = RestrictedBoltzmanMachine('rbm', self.layers_size[i], self.layers_size[i + 1])
            update_w, update_vb, update_hb = rbm.cd1(input_images, self.learning_rate)
            sess = tf.Session()
            sess.run(tf.initialize_all_variables())
            self.weights[i] = tf.Variable(sess.run(update_w))
            self.bias[i] = tf.Variable(sess.run(update_hb))
            input_images = sess.run(rbm.propup(input_images))

        dataset.batch_size = batch_size
        super(DeepBeliefNetwork, self).train(dataset, iteration_number)
