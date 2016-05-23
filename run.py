from evaluations import SimpleEvaluation
import ibsr

from models_in_gpu import NeuralNetwork, DeepBeliefNetwork
from display import display

from read_data import get_file

DISPLAY_NEURAL_NETWORK_SEGMENTATION = True

IMAGE_NUMBER_TO_DISPLAY_FOR_SEGMENTATION = 20
layers = [256 * 256, 10, 256 * 256]
learning_rate = 0.6
iteration_number = 1

nn = NeuralNetwork(layers, learning_rate)
# nn = DeepBeliefNetwork(layers, learning_rate)
nn.train(ibsr.train_set, iteration_number)
print 'checked'

simple_evaluation = SimpleEvaluation(nn)
accuracy = simple_evaluation.evaluate(ibsr.test_set)
print 'accuracy: %0.4f' % accuracy

from settings import THRESHOLD_128,THRESHOLD_192,THRESHOLD_254,THRESHOLD_0
import numpy
img, lbl = get_file(234)
img = img.reshape(1, img.size)
guess = nn.test(img)
a = abs(guess - THRESHOLD_0)
b = abs(guess - THRESHOLD_128)
c = abs(guess - THRESHOLD_192)
d = abs(guess - THRESHOLD_254)
distances = numpy.concatenate([a, b, c, d], 0)
min_distance = numpy.argmin(distances, 0)
min_distance = min_distance.reshape([1, 256 * 256])
guess[numpy.where(min_distance == 0)] = 0
guess[numpy.where(min_distance == 1)] = 128
guess[numpy.where(min_distance == 2)] = 192
guess[numpy.where(min_distance == 3)] = 254
display(img.reshape(256, 256), guess=guess.reshape(256, 256), label=lbl.reshape(256, 256))
