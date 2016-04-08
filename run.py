from evaluations import SimpleEvaluation
from read_data import random_select, post_process, pre_process
# from models import DeepBeliefNetwork, NeuralNetwork
from models_in_gpu import DeepBeliefNetwork, NeuralNetwork

from read_data import get_file
from display import display

__author__ = 'sharare'
DISPLAY_NEURAL_NETWORK_SEGMENTATION = True

IMAGE_NUMBER_TO_DISPLAY_FOR_SEGMENTATION = 20
layers = [256 * 256, 10, 256 * 256 * 4]
learning_rate = 0.5
iteration_number = 1
rbm_iteration_number = 1

# deep_belief_net = DeepBeliefNetwork(layers, learning_rate)
deep_belief_net = NeuralNetwork(layers, learning_rate)

train_set, test_set, train_labels, test_labels, \
    train_image_numbers, test_image_numbers = random_select(0.96)

# deep_belief_net.train(train_set, train_labels, train_image_numbers, iteration_number,
# rbm_iteration_number=rbm_iteration_number)
deep_belief_net.train(train_set, train_labels, train_image_numbers, iteration_number)

simple_evaluation = SimpleEvaluation(deep_belief_net)
accuracy = simple_evaluation.evaluate(test_set, test_labels, test_image_numbers)
print accuracy

# if DISPLAY_NEURAL_NETWORK_SEGMENTATION:
#     img, lbl = get_file(IMAGE_NUMBER_TO_DISPLAY_FOR_SEGMENTATION)
#     lbl = pre_process(lbl)
#     guess = deep_belief_net.test(img)
#     guess = post_process(guess)
#     display(img.reshape(256, 256), label=guess)
