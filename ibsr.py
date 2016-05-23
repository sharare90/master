import tensorflow as tf
import numpy
import read_data
from settings import THRESHOLD_0, THRESHOLD_128, THRESHOLD_192, THRESHOLD_254


class DataSet(object):
    def __init__(self, images, labels, batch_size, dtype=tf.float32):
        sample_numbers = len(images)
        dtype = tf.as_dtype(dtype).base_dtype

        if dtype == tf.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = numpy.multiply(images, 1.0 / 1.0)
            labels = numpy.multiply(labels, 1)
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
    max_img = numpy.max(img)
    img = numpy.multiply(img, 1.0 / float(max_img))
    lbl = lbl.astype('float')
    lbl = convert_label_to_thresholds(lbl)
    imgs.append(img)
    labels.append(lbl)


train_set = DataSet(imgs[:1000], labels[:1000], 40, dtype=tf.float32)
test_set = DataSet(imgs[1000 + 1:], labels[1000 + 1:], 125, dtype=tf.float32)
