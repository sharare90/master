import tensorflow as tf
# import Image
import numpy as np
# import sys
# sys.path.append('/home/hanhong/Projects/python27/DeepLearningTutorials/code/')
# from utils import tile_raster_images
from tensorflow.python.ops import control_flow_ops


#### we do the first test on the minst data again
#
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, \
                     mnist.test.images, mnist.test.labels



# helper function

def sample(probs):
    return tf.to_float(tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1)))


def sampleInt(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))


class RBM(object):
    def __init__(self, size_x, size_h, size_bt):
        # size_x is the size of the visiable layer
        # size_h is the size of the hidden layer
        # TODO what is side_h??
        side_h = 10
        self.size_x = size_x
        self.size_h = size_h
        self.size_bt = size_bt  # batch size

        self.k = tf.constant(1)

        # variables and place holder

        self.b = tf.Variable(tf.random_uniform([self.size_h, 1], -0.005, 0.005))
        self.W = tf.Variable(tf.random_uniform([self.size_x, self.size_h], -0.005, 0.005))
        self.c = tf.Variable(tf.random_uniform([self.size_x, 1], -0.005, 0.005))
        self.x = tf.placeholder(tf.float32, [self.size_x, self.size_bt])
        self.a = tf.placeholder(tf.float32)

        # define graph/algorithm

        # sample h x1 h1 ..
        self.h = sample(tf.sigmoid(tf.matmul(tf.transpose(self.W), self.x) + tf.tile(self.b, [1, self.size_bt])))

        self.ct = tf.constant(1)

        [self.xk1, self.hk1, _, _] = control_flow_ops.While(self.less_than_k, self.rbm_gibbs,
                                                            [self.x, self.h, self.ct, self.k], 1, False)

        # update rule
        [self.W_, self.b_, self.c_] = [tf.mul(self.a / float(self.size_bt),
                                              tf.sub(tf.matmul(self.x, tf.transpose(self.h)),
                                                     tf.matmul(self.xk1, tf.transpose(self.hk1)))), \
                                       tf.mul(self.a / float(self.size_bt),
                                              tf.reduce_sum(tf.sub(self.h, self.hk1), 1, True)), \
                                       tf.mul(self.a / float(self.size_bt),
                                              tf.reduce_sum(tf.sub(self.x, self.xk1), 1, True))]

        # wrap session
        self.updt = [self.W.assign_add(self.W_), self.b.assign_add(self.b_), self.c.assign_add(self.c_)]

        # stop gradient to save time and mem
        tf.stop_gradient(self.h)
        tf.stop_gradient(self.xk1)
        tf.stop_gradient(self.hk1)
        tf.stop_gradient(self.W_)
        tf.stop_gradient(self.b_)
        tf.stop_gradient(self.c_)

        self.sess = tf.Session()

    def train(self, train_set):
        # run session
        init = tf.initialize_all_variables()
        self.sess.run(init)

        # loop with batch
        for i in range(1, 10002):
            tr_x, tr_y = train_set.next_batch(self.size_bt)
            tr_x = np.transpose(tr_x)
            tr_y = np.transpose(tr_y)
            alpha = min(0.05, 100 / float(i))
            self.sess.run(self.updt, feed_dict={self.x: tr_x, self.a: alpha})
            print i, ' step size ', alpha

    def rbm_gibbs(self, xx, hh, count, k):
        xk = sampleInt(tf.sigmoid(tf.matmul(self.W, hh) + tf.tile(self.c, [1, self.size_bt])))
        hk = sampleInt(tf.sigmoid(tf.matmul(tf.transpose(self.W), xk) + tf.tile(self.b, [1, self.size_bt])))
        # assh_in1 = h_in.assign(hk)
        return xk, hk, count + 1, k

    def less_than_k(self, xx, hk, count, k):
        return count <= k

# rbm = RBM(28 * 28, 10, 100)
# rbm.train(mnist.train)
