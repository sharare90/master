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
test_imgs, test_lbls = ibsr.test_set.next_batch()

concatenated_images_with_labels = []
concatenated_test_images = []

label_for_window = np.zeros([4])
for i in xrange(len(labels) - 1):
    label = labels[i]
    a = (label == 0).sum()
    b = (label == 1).sum()
    c = (label == 2).sum()
    d = (label == 3).sum()
    max_index = np.argmax([a, b, c, d])
    label_for_window[max_index] = 1
    concatenated_images_with_labels.append(np.concatenate((label_for_window, imgs[i]), axis=0))
    label_for_window[max_index] = 0

rbm = RestrictedBoltzmannMachine('rbm', first_layer, window_height * window_width * NUMBER_OF_CLASSES)
update_w, update_vb, update_hb, error, h_end = rbm.cd1(concatenated_images_with_labels)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
weights = tf.Variable(sess.run(update_w))
bias = tf.Variable(sess.run(update_hb))
label_for_testset = np.zeros([4])
number_of_accurate_predict = 0
for i in xrange(len(test_lbls)):
    concatenated_test_images.append(np.concatenate((label_for_window, test_imgs[i]), axis=0))
    output_test_images = sess.run(rbm.propup(concatenated_test_images))
    test_label = test_lbls[i]
    a = (test_label == 0).sum()
    b = (test_label == 1).sum()
    c = (test_label == 2).sum()
    d = (test_label == 3).sum()
    max_index = np.argmax([a, b, c, d])
    max_predicted_index = np.argmax(output_test_images[0:4])
    if max_index == max_predicted_index:
        number_of_accurate_predict += 1

accuracy = number_of_accurate_predict/len(test_lbls)
print accuracy
