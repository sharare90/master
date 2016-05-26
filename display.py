import matplotlib
# matplotlib.rcParams['backend'] = "Qt4Agg"

import matplotlib.pyplot as plt

from read_data import get_file

import numpy as np
from settings import height, width, height_start, height_end, width_start, width_end
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
        im = np.zeros([height, width, 3])
        im[:, :, 0] = image[:, :]
        im[:, :, 1] = image[:, :]
        im[:, :, 2] = image[:, :]
        for i in range(height):
            for j in range(width):
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
    img = img.reshape(256, 256)
    img = img[height_start:height_end, width_start:width_end]
    img = img.reshape(height * width,)
    lbl = lbl.reshape(256, 256)
    lbl = lbl[height_start:height_end, width_start:width_end]
    lbl = lbl.reshape(height * width,)
    img[(np.where(img > 30)) and (np.where(lbl == 0))] = 0
    display(img.reshape([height, width]), guess=lbl.reshape(height, width), label=lbl.reshape(height, width))
