import matplotlib.pyplot as plt
from read_data import get_file
import numpy as np
from settings import DECIMAL_POINT_ROUND

IMAGE_NUMBER_TO_DISPLAY = 20


def display(image, write_to_file=False, label=None):
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
        image = im
    plt.imshow(image, cmap="gray")
    if write_to_file:
        output_filename = "JPCLN001.PNG"
        plt.savefig(output_filename)
    plt.show()


if __name__ == '__main__':
    img, lbl = get_file(IMAGE_NUMBER_TO_DISPLAY, column_format=True)
    display(img.reshape([256, 256]), label=lbl)
