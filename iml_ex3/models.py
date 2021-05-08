import numpy as np
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        """
        Given a training set, this method learns the parameters of the model
        and stores the trained model in self.model
        :param X: a numpy array of size m x d.
        :param y: a binary vector of size m.
        :return:
        """
        self.weights = np.zeroes(shape=(X.shape[1]))  # init column vec of zeroes
        hom_X = np.hstack(np.ones((X.shape[0], 1), X))
        while True:
            exists = False
            for i in range(len(y)):
                if y[i] * np.dot(self.weights, hom_X[i]) <= 0:
                    exists = True  # update flag
                    self.weights = self.weights + y[i] * hom_X[i]
            if not exists:
                return self.weights

    def predict(self, X):
        """
        uses the fitted parameters of the model to predict over a given dataset.
        :param X: numpy array of dimension mxd
        :return: the prediction of the trained model
        """
        hom_X = np.hstack(np.ones((X.shape[0], 1), X))
        return np.sign(hom_X @ self.weights)

    def score(self, X, y):
        pass


class LDA:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass


class SVM:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass
