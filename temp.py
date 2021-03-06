def normalization(image):
    max_image = image.max()
    min_image = image.min()
    image = (image - min_image) / float(max_image - min_image)
    # image = np.round(image, DECIMAL_POINT_ROUND)
    return image


def pre_process(image):
    image = image.reshape(256 * 256, 1)
    image = discretization(image)
    return image


def post_process(lbl):
    img = np.zeros((256, 256))
    for i in xrange(256):
        for j in xrange(256):
            ind = 4 * (256 * i + j)
            if lbl[ind] == 1:
                img[i, j] = 0
            elif lbl[ind + 1] == 1:
                img[i, j] = 128
            elif lbl[ind + 2] == 1:
                img[i, j] = 192
            else:
                img[i, j] = 254
    return img


def discretization(label):
    discretized_label = -1 * np.ones(shape=(len(label) * 4, 1))
    for i in xrange(len(label)):
        if label[i] == 0:
            discretized_label[4 * i] = 1
        elif label[i] == 128:
            discretized_label[4 * i + 1] = 1
        elif label[i] == 192:
            discretized_label[4 * i + 2] = 1
        else:
            discretized_label[4 * i + 3] = 1
    return discretized_label


def polarization(label):
    label = label.astype(float)
    for i in xrange(len(label)):
        if label[i] == 254:
            label[i] = 1
        elif label[i] == 192:
            label[i] = -1
        elif label[i] == 128:
            label[i] = 0.5
    return label


def random_select(ratio):
    """Selects a part of data for train and another part for test randomly
    then returns 4 objects. train_set, test_set, train_labels, test_labels"""
    train_set = []
    test_set = []
    train_labels = []
    test_labels = []
    number_of_images = get_number_of_images()
    test_image_numbers = range(number_of_images)
    train_image_numbers = sample(test_image_numbers, int(ratio * number_of_images))
    counter = 0
    train_image_numbers = train_image_numbers[:10]
    for i in train_image_numbers:
        img, label = get_file(i)
        label = pre_process(label)
        train_set.append(img)
        train_labels.append(label)
        print 'Image number: %d' % i
        counter += 1
        print '%d file processed' % counter
        test_image_numbers.remove(i)

    test_image_numbers = test_image_numbers[:10]
    for i in test_image_numbers:
        test_img, test_lbl = get_file(i)
        test_set.append(test_img)
        print 'Image number: %d' % i
        counter += 1
        print '%d file processed' % counter
        test_labels.append(test_lbl)

    return train_set, test_set, train_labels, test_labels, train_image_numbers, test_image_numbers
