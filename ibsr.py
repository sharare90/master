from sklearn.decomposition import KernelPCA, PCA
import tensorflow as tf
import numpy as np
import read_data
from settings import THRESHOLD_0, THRESHOLD_128, THRESHOLD_192, THRESHOLD_254, height_start, height_end, width_start, \
    width_end, height, width, PCA_COMPONENTS_COUNT, USE_PCA, window_height, window_width
import settings
import random
from operator import itemgetter


class DataSet(object):
    def __init__(self, images, labels, batch_size, dtype=tf.float32):
        sample_numbers = len(images)
        dtype = tf.as_dtype(dtype).base_dtype

        if dtype == tf.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = np.multiply(images, 1.0 / 256.0)
            labels = np.multiply(labels, 1)

            # labels[np.where(labels == 1)] = 0
            # labels[np.where(labels == 2)] = 0
            # labels[np.where(labels == 3)] = 1

            # labels[np.where(labels == 1)] = 1
            # labels[np.where(labels > 0)] = 1

            self.sample_numbers = sample_numbers
            self.batch_size = batch_size
            self._images = images
            self._labels = labels
            self._epochs_completed = 0
            self._index_in_epoch = 0

    def count(self):
        return self.sample_numbers

    def next_batch(self):
        start = self._index_in_epoch
        self._index_in_epoch += self.batch_size
        if self._index_in_epoch > self.sample_numbers:
            # epoch is finished
            self._epochs_completed += 1
            # Shuffle data
            perm = np.arange(self.sample_numbers)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = self.batch_size
            assert self.batch_size <= self.sample_numbers
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


thresholds = {
    0: THRESHOLD_0,
    128: THRESHOLD_128,
    192: THRESHOLD_192,
    254: THRESHOLD_254,
}


def get_rectangle(image, start_point):
    return image[start_point[0]:start_point[0] + window_height, start_point[1]:start_point[1] + window_width]


def convert_label_to_thresholds(element):
    return thresholds[element]


convert_label_to_thresholds = np.vectorize(convert_label_to_thresholds)


def crop(image, label):
    image = image[height_start:height_end, width_start:width_end]
    label = label[height_start:height_end, width_start:width_end]
    return image, label


imgs = []
labels = []

for i in xrange(1126):
    print i
    image, label = read_data.get_file(i + 1, column_format=True)

    image = image.reshape(256, 256)
    label = label.reshape(256, 256)

    if settings.CROP:
        image, label = crop(image, label)

    number_of_height_partitions = (height_end - height_start) / window_height if settings.PARTITION else 1
    number_of_width_partitions = (width_end - width_start) / window_width if settings.PARTITION else 1

    for j in xrange(number_of_height_partitions):
        for k in xrange(number_of_width_partitions):
            if settings.PARTITION:
                start_point = [window_height * j, window_width * k]
                img = get_rectangle(image, start_point)
                lbl = get_rectangle(label, start_point)
            img = img.reshape(img.shape[0] * img.shape[1], )
            lbl = lbl.reshape(lbl.shape[0] * lbl.shape[1], )
            lbl = lbl.astype('float')
            lbl = convert_label_to_thresholds(lbl)
            if settings.SUPER_PIXEL:
                count_0 = (lbl == thresholds[0]).sum()
                count_128 = (lbl == thresholds[128]).sum()
                count_192 = (lbl == thresholds[192]).sum()
                count_254 = (lbl == thresholds[254]).sum()

                lbl = np.argmax([count_0, count_128, count_192, count_254])

            # max_img = np.max(img)
            # min_img = np.min(img)
            # if max_img - min_img == 0:
            #     max_img = 100000
            # img = np.multiply(img - min_img, 1.0 / float(max_img - min_img))

            imgs.append(img)
            labels.append(lbl)

train_test_separator = 900000

train_imgs = imgs[:train_test_separator]
train_labels = labels[:train_test_separator]

test_imgs = imgs[train_test_separator:]
test_imgs = np.multiply(test_imgs, 1)

pca = None
if USE_PCA:
    pca = PCA(n_components=PCA_COMPONENTS_COUNT)
    pca.fit(train_imgs)
    train_imgs = pca.transform(train_imgs)
    test_imgs = pca.transform(test_imgs)

if settings.OVER_SAMPLING:
    over_sampled_set_images = []
    over_sampled_set_labels = []
    number_of_samples_for_each_class = {0: settings.samples_for_0,
                                        1: settings.samples_for_128,
                                        2: settings.samples_for_192,
                                        3: settings.samples_for_254}

    for label, number_of_samples in number_of_samples_for_each_class.iteritems():
        indices = list(np.where(train_labels == label)[0])
        label_imgs = itemgetter(*indices)(train_imgs)
        label_labels = itemgetter(*indices)(train_labels)
        counter = 0

        while counter + len(label_imgs) < number_of_samples:
            over_sampled_set_images.extend(label_imgs)
            over_sampled_set_labels.extend(label_labels)
            counter += len(label_imgs)

        number_of_required_samples = number_of_samples - counter
        samples = random.sample(range(len(label_imgs)), number_of_required_samples)
        over_sampled_set_images.extend(itemgetter(*samples)(label_imgs))
        over_sampled_set_labels.extend(itemgetter(*samples)(label_labels))

    train_imgs = over_sampled_set_images
    train_labels = over_sampled_set_labels

train_imgs = np.multiply(train_imgs, 1)

train_set = DataSet(train_imgs, train_labels, 10, dtype=tf.float32)
test_set = DataSet(test_imgs, labels[train_test_separator + 1:], 25, dtype=tf.float32)
