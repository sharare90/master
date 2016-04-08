import pycuda.driver as cuda
import tensorflow as tf

from abc import ABCMeta, abstractmethod

import numpy as np

from utils import sigmoid, sigmoid_derivative, create_random_weights, maximization

__author__ = 'sharare'


class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, train_set, labels, train_image_numbers):
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
    def __init__(self, layers_size, learning_rate, initial_weights=None, initial_bias=None):
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

    def train(self, train_set, labels, train_image_numbers, iteration_number):
        y_ = tf.placeholder(tf.float32, [None, self.layers_size[-1]])
        self.layers[0].input = tf.placeholder(tf.float32, [None, self.layers_size[0]])
        self.layers[0].output = tf.nn.softmax(self.layers[0].input)
        for i in xrange(len(self.layers) - 1):
            self.layers[i + 1].input = tf.matmul(self.layers[i].output, self.weights[i]) + self.bias[i]
            self.layers[i + 1].output = tf.nn.softmax(self.layers[i + 1].input)
        loss_function = -tf.reduce_sum(y_ * tf.log(self.layers[-1].output))
        train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss_function)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        for iteration in xrange(iteration_number):
            # input, labels = train_set.next_batch(self.batch_size)
            for i in xrange(len(train_set)):
                input_imgs = train_set[i].transpose()
                lbls = labels[i].transpose()
                self.sess.run(train_step, feed_dict={self.layers[0].input: input_imgs, y_: lbls})

    def test(self, img):
        output = self.layers[-1].output
        return self.sess.run(output, feed_dict={self.layers[0].input: img.transpose()})

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
    def train(self, train_set, labels, train_image_numbers, iteration_number):
        for iteration in xrange(iteration_number):
            for i in xrange(len(train_set)):
                print 'rbm:' + str(i)
                train_input = train_set[i]
                self.layers[0].input = train_input
                self.layers[0].output = train_input
                v = self.layers[0].output
                h = self.create_h_given_v(v)
                v2 = self.create_v_given_h(h)
                h2 = self.create_h_given_v(v2)
                delta_w = self.learning_rate * (
                    (v * h.transpose()) - (v2 * h2.transpose()))
                self.update_weights([delta_w, ], [np.matrix(np.zeros([self.layers_size[1], 1])), ])

    def create_h_given_v(self, v):
        return sigmoid(self.weights[0].transpose() * v + self.bias[0])

    def create_v_given_h(self, h):
        return sigmoid(self.weights[0]) * h


class DeepBeliefNetwork(NeuralNetwork):
    def train(self, train_set, labels, train_image_numbers, iteration_number, rbm_iteration_number=None):
        if rbm_iteration_number is None:
            rbm_iteration_number = iteration_number
        for i in range(len(self.layers) - 1):
            print i
            l_size = [self.layers_size[i], self.layers_size[i + 1]]
            w = [self.weights[i], ]
            b = [self.bias[i], ]
            l = [self.layers[i], self.layers[i + 1]]
            rbm = RestrictedBoltzmanMachine(l_size, self.learning_rate, initial_weights=w, initial_bias=b,
                                            layers=l)
            tr_set = []
            for t in train_set:
                t_in = t
                for x in range(i):
                    t_in = (self.weights[x].transpose() * t_in) + self.bias[x]
                tr_set.append(t_in)
            rbm.train(tr_set, None, train_image_numbers, rbm_iteration_number)
        super(DeepBeliefNetwork, self).train(train_set, labels, train_image_numbers, iteration_number)
