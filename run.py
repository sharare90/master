from evaluations import SimpleEvaluation
import ibsr

from models_in_gpu import NeuralNetwork, DeepBeliefNetwork


DISPLAY_NEURAL_NETWORK_SEGMENTATION = True

IMAGE_NUMBER_TO_DISPLAY_FOR_SEGMENTATION = 20
layers = [256 * 256, 128, 128, 256 * 256]
learning_rate = 0.9
iteration_number = 10

# nn = NeuralNetwork(layers, learning_rate)
nn = DeepBeliefNetwork(layers, learning_rate)
nn.train(ibsr.train_set, iteration_number)
print 'checked'

simple_evaluation = SimpleEvaluation(nn)
accuracy = simple_evaluation.evaluate(ibsr.test_set)
print accuracy
