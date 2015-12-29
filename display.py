import matplotlib.pyplot as plt
from read_data import get_file
import numpy as np
from settings import DECIMAL_POINT_ROUND

IMAGE_NUMBER_TO_DISPLAY = 20


def display(image, write_to_file=False, label=None):
    if label is not None:
        im = np.zeros([256, 256, 3])
        im[:, :, 0] = image[:, :]
        im[:, :, 1] = image[:, :]
        im[:, :, 2] = image[:, :]
        for i in range(256):
            for j in range(256):
                ind = 256 * i + j
                if label[4 * ind + 1, 0] == 1:
                    im[i, j, 0] = 1
                elif label[4 * ind + 2, 0] == 1:
                    im[i, j, 1] = 1
                elif label[4 * ind + 3, 0] == 1:
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
