from models_in_gpu import DeepBeliefNetwork, NeuralNetwork

__author__ = 'sharare'

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# x = tf.placeholder(tf.float32, [None, 784])
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))

# y = tf.nn.softmax(tf.matmul(x, W) + b)
# y_ = tf.placeholder(tf.float32, [None, 10])
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# squared_sum = tf.reduce_sum(tf.square(y_-y))


# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(squared_sum)
# init = tf.initialize_all_variables()

# sess = tf.Session()
# sess.run(init)

layers = [784, 10]
learning_rate = 0.01
iteration_number = 1000

# nn = NeuralNetwork(layers, learning_rate)
nn = NeuralNetwork(layers, learning_rate)
nn.train(mnist.train, iteration_number)
guess = nn.test(mnist.test.images)


print 'labels'
import numpy as np
labels = np.argmax(mnist.test.labels, 1)

print np.mean(np.equal(guess, labels))
print np.sum(guess[np.where(guess == 3.0)]) / 3
print guess.shape


# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
