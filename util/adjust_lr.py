

import numpy as np


class DecayLearningRate(object):

    def __init__(self, lr=0.007, epochs=100, factor=0.9):
        self.lr = lr
        self.epochs = epochs
        self.factor = factor

    def get_learning_rate(self, epoch, step=None):
        rate = np.power(1.0 - epoch / float(self.epochs + 1), self.factor)
        return rate * self.lr