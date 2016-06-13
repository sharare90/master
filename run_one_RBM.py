__author__ = 'sharare'
import ibsr
import numpy as np
from settings import USE_PCA, PCA_COMPONENTS_COUNT, NUMBER_OF_CLASSES, window_height, window_width
# from models_in_gpu import RestrictedBoltzmanMachine
from rbm2 import RBM
import tensorflow as tf

layer_size = PCA_COMPONENTS_COUNT if USE_PCA else window_height * window_width + NUMBER_OF_CLASSES

learning_rate = 0.001

rbm = RBM(layer_size, layer_size, 10)

ibsr.train_set.batch_size = ibsr.train_set.count()
imgs, labels = ibsr.train_set.next_batch()
test_imgs, test_lbls = ibsr.test_set.next_batch()
one_hot_presentation = np.eye(NUMBER_OF_CLASSES)[[labels]].reshape(labels.shape[0], NUMBER_OF_CLASSES)
data = np.concatenate((imgs, one_hot_presentation), 1)
data = data.astype(np.float32)

train_set = ibsr.DataSet(data, labels, 10)
rbm.train(train_set)

labels_for_testset = np.zeros([test_imgs.shape[0], 4])
test_data = np.concatenate((test_imgs, labels_for_testset), 1)
test_data = test_data.astype(np.float32)

import tensorflow as tf

input_data = tf.placeholder(tf.float32, shape=[113400, 24])
bias_matrix = rbm.sess.run(rbm.b)
bias_matrix = np.multiply([bias_matrix,] * 113400, 1)
bias_matrix = bias_matrix.reshape(113400, 24)
tf_bias_matrix = tf.placeholder(tf.float32, shape=[113400, 24])

output = tf.matmul(input_data, rbm.W) + tf_bias_matrix
result = rbm.sess.run(output, feed_dict={input_data: test_data, tf_bias_matrix: bias_matrix})

# output = rbm.h
# result = rbm.sess.run(output, feed_dict={rbm.x: test_data})


result = result[:, 20:]
result = np.argmax(result, 1)

accuracy = (result == test_lbls).sum() / float(len(test_lbls))
print 'Accuracy', accuracy
