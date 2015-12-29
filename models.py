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
        self.input = []
        self.output = []
        self.n = number_of_neurons


class NeuralNetwork(Model):
    def __init__(self, layers_size, learning_rate, initial_weights=None, initial_bias=None, layers=None):
        self.layers_size = layers_size
        if layers:
            self.layers = layers
        else:
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

    def train(self, train_set, labels, train_image_numbers, iteration_number):
        for iteration in xrange(iteration_number):
            for i in xrange(len(train_set)):
                print i
                train_input = train_set[i]
                output = self.feed_forward(train_input)
                self.back_propagate(output, labels[i])

    def feed_forward(self, train_input):
        self.layers[0].input = train_input
        self.layers[0].output = train_input
        for i in xrange(len(self.layers) - 1):
            self.layers[i + 1].input = (self.weights[i].transpose() * self.layers[i].output) + self.bias[i]
            self.layers[i + 1].output = sigmoid(self.layers[i + 1].input)
        return self.layers[-1].output

    def back_propagate(self, output, label):
        # normalize_output()
        error = label - output
        delta = np.array(error) * np.array(sigmoid_derivative(self.layers[-1].input))
        delta = np.matrix(delta)
        delta_weights = []
        delta_bias = []
        for i in xrange(len(self.weights), 0, -1):
            delta_w = (np.array(self.layers[i - 1].output) * np.array(delta).transpose() * self.learning_rate)
            delta_w = np.matrix(delta_w)
            delta_b = np.matrix(self.learning_rate * delta)
            delta_weights.append(delta_w)
            delta_bias.append(delta_b)
            error = self.weights[i-1] * delta
            error = np.matrix(error)
            delta = np.array(error) * np.array(sigmoid_derivative(self.layers[i-1].input))
            delta = np.matrix(delta)
        self.update_weights(delta_weights, delta_bias)

    def update_weights(self, delta_weights, delta_bias):
        for i in range(len(self.weights)):
            self.weights[i] += delta_weights[-i-1]
            self.bias[i] += delta_bias[-i-1]

    def test(self, img, output_polarization=True):
        if output_polarization:
            return maximization(self.feed_forward(img))
        return self.feed_forward(img)

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
