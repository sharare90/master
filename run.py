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

# nn = NeuralNetwork(layers, learning_rate)
nn = DeepBeliefNetwork(layers, learning_rate)
nn.train(ibsr.train_set, iteration_number)
print 'checked'

simple_evaluation = SimpleEvaluation(nn)
accuracy = simple_evaluation.evaluate(ibsr.test_set)
print 'accuracy: %0.4f' % accuracy

img, lbl = get_file(10)
display(img.reshape(256, 256), label=lbl.reshape(256, 256))
