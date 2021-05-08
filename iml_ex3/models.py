import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class Perceptron:

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
        pass


# noinspection PyTypeChecker
class LDA:
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
        del_1, del_min1 = 1., -1.
        y_hat = np.array(
            [{0: 1., 1: -1.}[np.argmax([self.deltas[del_1][i],
                                        self.deltas[del_min1][i]])]
             for i in range(X.shape[0])])
        return y_hat

    def score(self, X, y):
        pass


class SVM:
    def __init__(self):
        self.svm = SVC(C=1e10, kernel='linear')

    def fit(self, X, y):
        """
        Given a training set, this method learns the parameters of the model
        and stores the trained model in self.model
        :param X: a numpy array of size m x d.
        :param y: a binary vector of size m.
        """
        self.svm.fit(X, y)

    def predict(self, X):
        """
        uses the fitted parameters of the model to predict over a given dataset.
        :param X: numpy array of dimension mxd
        :return: the prediction of the trained model
        """
        return self.svm.predict(X)

    def score(self, X, y):
        pass


class Logistic:
    def __init__(self):
        self.logistic = LogisticRegression(solver='liblinear')

    def fit(self, X, y):
        """
        Given a training set, this method learns the parameters of the model
        and stores the trained model in self.model
        :param X: a numpy array of size m x d.
        :param y: a binary vector of size m.
        """
        self.logistic.fit(X, y)

    def predict(self, X):
        """
        uses the fitted parameters of the model to predict over a given dataset.
        :param X: numpy array of dimension mxd
        :return: the prediction of the trained model
        """
        return self.logistic.predict(X)

    def score(self, X, y):
        pass


class DecisionTree:
    def __init__(self):
        self.tree = DecisionTreeClassifier()

    def fit(self, X, y):
        """
        Given a training set, this method learns the parameters of the model
        and stores the trained model in self.model
        :param X: a numpy array of size m x d.
        :param y: a binary vector of size m.
        """
        self.tree.fit(X, y)

    def predict(self, X):
        """
        uses the fitted parameters of the model to predict over a given dataset.
        :param X: numpy array of dimension mxd
        :return: the prediction of the trained model
        """
        return self.tree.predict(X)

    def score(self, X, y):
        pass
