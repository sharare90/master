from abc import ABCMeta, abstractmethod

from utils import similarity

__author__ = 'sharare'


class Evaluation(object):
    __metaclass__ = ABCMeta

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def evaluate(self, test_set, test_labels, test_image_numbers):
        pass


class SimpleEvaluation(Evaluation):
    def evaluate(self, test_set, test_labels, test_image_numbers):
        sum_evaluation = 0
        for i in range(len(test_set)):
            sum_evaluation += similarity(self.model.test(test_set[i]), test_labels[i])
        accuracy = sum_evaluation/float(len(test_set))
        return accuracy
