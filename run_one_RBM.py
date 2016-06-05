__author__ = 'sharare'
import ibsr
import numpy as np
from settings import USE_PCA, PCA_COMPONENTS_COUNT, NUMBER_OF_CLASSES, window_height, window_width
from models_in_gpu import RestrictedBoltzmanMachine
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
rbm = RestrictedBoltzmanMachine('rbm', first_layer + 4, (window_height * window_width * NUMBER_OF_CLASSES) + 4)

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

concatenated_images_with_labels = np.multiply(concatenated_images_with_labels, 1)
concatenated_images_with_labels = tf.cast(concatenated_images_with_labels, tf.float32)
tf_concatenated_images_with_labels = tf.Variable(concatenated_images_with_labels)
update_w, update_vb, update_hb, h_end = rbm.cd1(tf_concatenated_images_with_labels)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
weights = tf.Variable(sess.run(update_w))
bias = tf.Variable(sess.run(update_hb))
label_for_testset = np.zeros([4])
for i in xrange(len(test_lbls) - 1):
    concatenated_test_images.append(np.concatenate((label_for_window, test_imgs[i]), axis=0))

concatenated_test_images = np.multiply(concatenated_test_images, 1)
tf_concatenated_test_images = tf.cast(concatenated_test_images, tf.float32)
output_test_images = sess.run(rbm.propup(tf_concatenated_test_images))

# print output_test_images
correct_0 = 0
correct_128 = 0
correct_192 = 0
correct_254 = 0
for i in xrange(len(test_lbls) - 1):
    test_label = test_lbls[i]
    a = (test_label == 0).sum()
    b = (test_label == 1).sum()
    c = (test_label == 2).sum()
    d = (test_label == 3).sum()
    label_0 = np.equal(np.argmax([a, b, c, d]), 0)
    label_128 = np.equal(np.argmax([a, b, c, d]), 1)
    label_192 = np.equal(np.argmax([a, b, c, d]), 2)
    label_254 = np.equal(np.argmax([a, b, c, d]), 3)
    guess_0 = np.equal(np.argmax(output_test_images[i, 0:4]), 0)
    guess_128 = np.equal(np.argmax(output_test_images[i, 0:4]), 1)
    guess_192 = np.equal(np.argmax(output_test_images[i, 0:4]), 2)
    guess_254 = np.equal(np.argmax(output_test_images[i, 0:4]), 3)
    correct_0 += np.multiply(label_0, guess_0)
    correct_128 += np.multiply(label_128, guess_128)
    correct_192 += np.multiply(label_192, guess_192)
    correct_254 += np.multiply(label_254, guess_254)

print float(correct_0 + correct_128 + correct_192 + correct_254)/float(len(test_lbls))


# guess = tf.placeholder(tf.float32, [None, 4 + window_height * window_width])
# tf_labels = tf.placeholder(tf.float32, [None, window_height * window_width])
#
# a = (test_lbls == 0).sum()
# b = (test_lbls == 1).sum()
# c = (test_lbls == 2).sum()
# d = (test_lbls == 3).sum()
#
# label_0 = tf.cast(tf.equal(np.argmax([a, b, c, d]), 0), tf.float32)
# label_128 = tf.cast(tf.equal(np.argmax([a, b, c, d]), 1), tf.float32)
# label_192 = tf.cast(tf.equal(np.argmax([a, b, c, d]), 2), tf.float32)
# label_254 = tf.cast(tf.equal(np.argmax([a, b, c, d]), 3), tf.float32)
#
# guess_0 = tf.cast(tf.equal(np.argmax(output_test_images[0:4]), 0), tf.float32)
# guess_128 = tf.cast(tf.equal(np.argmax(output_test_images[0:4]), 1), tf.float32)
# guess_192 = tf.cast(tf.equal(np.argmax(output_test_images[0:4]), 2), tf.float32)
# guess_254 = tf.cast(tf.equal(np.argmax(output_test_images[0:4]), 3), tf.float32)
#
# correct_0 = tf.reduce_sum(tf.mul(guess_0, label_0))
# correct_128 = tf.reduce_sum(tf.mul(guess_128, label_128))
# correct_192 = tf.reduce_sum(tf.mul(guess_192, label_192))
# correct_254 = tf.reduce_sum(tf.mul(guess_254, label_254))
#
# accuracy_0 = correct_0 / tf.reduce_sum(label_0)
# accuracy_128 = correct_128 / tf.reduce_sum(label_128)
# accuracy_192 = correct_192 / tf.reduce_sum(label_192)
# accuracy_254 = correct_254 / tf.reduce_sum(label_254)
#
# accuracy = (correct_128 + correct_192 + correct_254) / (
#     tf.reduce_sum(label_128) + tf.reduce_sum(label_192) + tf.reduce_sum(label_254))
#
# print "0 accuracy: %0.4f" % sess.run(accuracy_0, feed_dict={tf_labels: test_lbls, guess: output_test_images})
# print "128 accuracy: %0.4f" % sess.run(accuracy_128, feed_dict={tf_labels: test_lbls, guess: output_test_images})
# print "192 accuracy: %0.4f" % sess.run(accuracy_192, feed_dict={tf_labels: test_lbls, guess: output_test_images})
# print "254 accuracy: %0.4f" % sess.run(accuracy_254, feed_dict={tf_labels: test_lbls, guess: output_test_images})
# # accuracy = tf.reduce_mean(tf.cast(tf.equal(guess, tf_labels), tf.float32))
# print sess.run(accuracy, feed_dict={tf_labels: test_lbls, guess: output_test_images})
