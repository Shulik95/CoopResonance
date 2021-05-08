import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from abc import ABC, abstractmethod


class AbsClass(ABC):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def score(self, X, y):
        """
        :param X: unlabeled test set given as numpy array of dim mXd
        :param y: numpy array of dim m - the true labeles
        :return: a dictionary with the following fields:
        num_samples: number of samples in test set.
        error: error (misclassification rate)
        accuracy: number of correct classifications out of all predictions
        FPR: false positive rate
        TPR: true positive rate
        precision:
        specificity:
        """
        ret_dict = {"num_samples": 0, "error": 0, "accuracy": 0, "FPR": 0, "TPR": 0, "precision": 0, "specificty": 0}
        self.fit(X, y)
        y_hat = self.predict(X)
        n = len(y)
        ret_dict["num_samples"] = n
        ret_dict["error"] = np.sum(y != y_hat) / n
        ret_dict["accuracy"] = np.sum(y == y_hat) / n
        ret_dict["FPR"] = np.sum((y - y_hat) == -2) / (np.sum((y - y_hat) == -2) + np.sum((y + y_hat) == -2))
        ret_dict["TPR"] = np.sum((y + y_hat) == 2) / np.sum(y_hat == 1)
        ret_dict["precision"] = np.sum((y + y_hat) == 2) / (np.sum((y + y_hat) == 2) + np.sum((y - y_hat) == -2))
        ret_dict["specificty"] = np.sum((y + y_hat) == -2) / np.sum(y_hat == -1)
        return ret_dict


class Perceptron(AbsClass):

    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        """
        Given a training set, this method learns the parameters of the model
        and stores the trained model in self.model
        :param X: a numpy array of size m x d.
        :param y: a binary vector of size m.
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
                break

    def predict(self, X):
        """
        uses the fitted parameters of the model to predict over a given dataset.
        :param X: numpy array of dimension mxd
        :return: the prediction of the trained model
        """
        hom_X = np.hstack(np.ones((X.shape[0], 1), X))
        return np.sign(hom_X @ self.weights)

    def score(self, X, y):
        """
        documentation available at the implementation.
        """
        super().score(X, y)


class LDA(AbsClass):
    def __init__(self):
        self.deltas = []

    def fit(self, X, y):
        """
        Given a training set, this method learns the parameters of the model
        and stores the trained model in self.model
        :param X: a numpy array of size m x d.
        :param y: a binary vector of size m.
        """
        for _y in [1, -1]:
            prob = np.sum(y == _y) / len(y)  # probability to get val _y
            ln_P = np.log(prob)
            mean_vec = np.array([np.mean(x_i) for x_i in X[y == _y]])
            inv_sig = np.pinv(np.cov(X))

            delta = [(x.T @ inv_sig @ mean_vec - 0.5 * mean_vec.T @ inv_sig @ mean_vec + ln_P) for x in X]
            self.deltas.append(delta)

    def predict(self, X):
        """
        uses the fitted parameters of the model to predict over a given dataset.
        :param X: numpy array of dimension mxd
        :return: the prediction of the trained model
        """
        del_1, del_min1 = 0, 1
        y_hat = np.array(
            [{0: 1., 1: -1.}[np.argmax([self.deltas[del_1][i],
                                        self.deltas[del_min1][i]])]
             for i in range(X.shape[0])])
        return y_hat

    def score(self, X, y):
        """
        documentation available at the implementation.
        """
        super().score(X, y)


class SVM(AbsClass):
    def __init__(self):
        self.model = SVC(C=1e10, kernel='linear')

    def fit(self, X, y):
        """
        Given a training set, this method learns the parameters of the model
        and stores the trained model in self.model
        :param X: a numpy array of size m x d.
        :param y: a binary vector of size m.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        uses the fitted parameters of the model to predict over a given dataset.
        :param X: numpy array of dimension mxd
        :return: the prediction of the trained model
        """
        return self.model.predict(X)

    def score(self, X, y):
        super().score(X, y)


class Logistic(AbsClass):
    def __init__(self):
        self.model = LogisticRegression(solver='liblinear')

    def fit(self, X, y):
        """
        Given a training set, this method learns the parameters of the model
        and stores the trained model in self.model
        :param X: a numpy array of size m x d.
        :param y: a binary vector of size m.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        uses the fitted parameters of the model to predict over a given dataset.
        :param X: numpy array of dimension mxd
        :return: the prediction of the trained model
        """
        return self.model.predict(X)

    def score(self, X, y):
        super().score(X, y)


class DecisionTree(AbsClass):
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def fit(self, X, y):
        """
        Given a training set, this method learns the parameters of the model
        and stores the trained model in self.model
        :param X: a numpy array of size m x d.
        :param y: a binary vector of size m.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        uses the fitted parameters of the model to predict over a given dataset.
        :param X: numpy array of dimension mxd
        :return: the prediction of the trained model
        """
        return self.model.predict(X)

    def score(self, X, y):
        super().score(X, y)
