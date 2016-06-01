from sklearn.decomposition import KernelPCA, PCA
import tensorflow as tf
import numpy
import read_data
from settings import THRESHOLD_0, THRESHOLD_128, THRESHOLD_192, THRESHOLD_254, height_start, height_end, width_start, \
    width_end, height, width, PCA_COMPONENTS_COUNT, USE_PCA


class DataSet(object):
    def __init__(self, images, labels, batch_size, dtype=tf.float32):
        sample_numbers = len(images)
        dtype = tf.as_dtype(dtype).base_dtype

        if dtype == tf.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = numpy.multiply(images, 1.0 / 1.0)
            labels = numpy.multiply(labels, 1)

            # labels[numpy.where(labels == 1)] = 0
            # labels[numpy.where(labels == 2)] = 0
            # labels[numpy.where(labels == 3)] = 1

            # labels[numpy.where(labels == 1)] = 1
            # labels[numpy.where(labels > 0)] = 1

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
            perm = numpy.arange(self.sample_numbers)
            numpy.random.shuffle(perm)
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


def convert_label_to_thresholds(element):
    return thresholds[element]


convert_label_to_thresholds = numpy.vectorize(convert_label_to_thresholds)

imgs = []
labels = []


for i in xrange(1126):
    print i
    img, lbl = read_data.get_file(i + 1, column_format=True)
    # img = lbl.copy()
    img = img.reshape(256, 256)
    img = img[height_start:height_end, width_start:width_end]
    lbl = lbl.reshape(256, 256)
    lbl = lbl[height_start:height_end, width_start:width_end]

    # img = gaussian_filter(img, sigma=0.001)
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    max_img = numpy.max(img)
    min_img = numpy.min(img)
    if max_img - min_img == 0:
        max_img = 12
        min_img = 0
    img = numpy.multiply(img - min_img, 1.0 / float(max_img - min_img))
    img = img.reshape(height * width, )
    lbl = lbl.reshape(height * width, )
    lbl = lbl.astype('float')
    lbl = convert_label_to_thresholds(lbl)
    imgs.append(img)
    labels.append(lbl)

train_test_separator = 1000

train_imgs = imgs[:train_test_separator]
train_imgs = numpy.multiply(train_imgs, 1)

test_imgs = imgs[train_test_separator:]
test_imgs = numpy.multiply(test_imgs, 1)

pca = None
if USE_PCA:
    pca = PCA(n_components=PCA_COMPONENTS_COUNT)
    pca.fit(train_imgs)
    train_imgs = pca.transform(train_imgs)
    test_imgs = pca.transform(test_imgs)


train_set = DataSet(train_imgs, labels[:train_test_separator], 40, dtype=tf.float32)
test_set = DataSet(test_imgs, labels[train_test_separator + 1:], 25, dtype=tf.float32)