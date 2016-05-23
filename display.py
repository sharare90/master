import matplotlib
# matplotlib.rcParams['backend'] = "Qt4Agg"

import matplotlib.pyplot as plt

from read_data import get_file

import numpy as np

IMAGE_NUMBER_TO_DISPLAY = 234


def display(image, guess=None, label=None):
    """ This function displays the image with its label.
    inputs: image, write_to_file=False, label=None
    If lable is not None the values of label will be colorized.
    If label value is 128 the pixel will be red.
    If label valus is 192 the pixel will be green.
    If label value is 254 the pixel will be blue.
    """
    if label is not None:
        im = np.zeros([256, 256, 3])
        im[:, :, 0] = image[:, :]
        im[:, :, 1] = image[:, :]
        im[:, :, 2] = image[:, :]
        for i in range(256):
            for j in range(256):
                if label[i, j] == 128:
                    im[i, j, 0] = 1
                elif label[i, j] == 192:
                    im[i, j, 1] = 1
                elif label[i, j] == 254:
                    im[i, j, 2] = 1
                # if guess[i, j] != label[i, j]:
                #     im[i, j, 1:2] = 1
                #     im[i, j, 0] = 0
        image = im
    plt.imshow(image, cmap="gray")
    plt.show()


if __name__ == '__main__':
    img, lbl = get_file(IMAGE_NUMBER_TO_DISPLAY)
    display(lbl.reshape([256, 256]), guess=lbl.reshape(256, 256), label=lbl.reshape(256, 256))
