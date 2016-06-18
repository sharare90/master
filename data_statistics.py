from read_data import get_file
from display import display
import matplotlib.pyplot as plt

for i in xrange(1126):
    img, lbl = get_file(i)
    img = img - min(img)
    img /= float(max(img))
    max_for_each_image = max(img)
    min_for_each_image = min(img)
    more_than_256_count = (img > 256).sum()
    print i, min_for_each_image, max_for_each_image, more_than_256_count

img, lbl = get_file(759)
figure = plt.figure()
sub_plot = figure.add_subplot(1, 2, 1)
display(img.reshape(256, 256))
sub_plot = figure.add_subplot(1, 2, 2)
display(lbl.reshape(256, 256))
plt.show()
