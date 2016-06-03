__author__ = 'sharare'
import ibsr
import numpy as np
from settings import USE_PCA, PCA_COMPONENTS_COUNT, NUMBER_OF_CLASSES, window_height, window_width
from oneRBM import RestrictedBoltzmannMachine
import tensorflow as tf
first_layer = PCA_COMPONENTS_COUNT if USE_PCA else window_height * window_width
layers = [first_layer, window_height * window_width * NUMBER_OF_CLASSES]
learning_rate = 0.65

ibsr.train_set.batch_size = ibsr.train_set.count()
imgs, labels = ibsr.train_set.next_batch()
concatenated_images_with_labels = []


label_for_window = np.zeros([4])
for i in xrange(len(labels) - 1):
    label = labels[i]
    a = np.sum(label[np.where(label == 0)])
    b = np.sum(label[np.where(label == 1)])
    c = np.sum(label[np.where(label == 2)])
    d = np.sum(label[np.where(label == 3)])
    label_for_window[np.argmax([a, b, c, d])] = 1
    concatenated_images_with_labels.append(np.concatenate(label_for_window, imgs[i]))

rbm = RestrictedBoltzmannMachine('rbm', first_layer, window_height * window_width * NUMBER_OF_CLASSES)
update_w, update_vb, update_hb, error, h_end = rbm.cd1(concatenated_images_with_labels, learning_rate)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
weights = tf.Variable(sess.run(update_w))
bias = tf.Variable(sess.run(update_hb))
input_images = sess.run(rbm.propup(concatenated_images_with_labels))