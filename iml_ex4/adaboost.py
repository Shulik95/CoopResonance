"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

"""
import numpy as np

from ex4_tools import *

NO_NOISE = 0
SMALL_NOISE = 0.01


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """

        D = np.ones(y.shape) / y.shape[0]  # init parameters
        for i in range(0, self.T):
            self.h[i] = self.WL(D, X, y)  # construct a weak learner
            y_hat = (self.h[i]).predict(X)
            err_t = np.sum(D[y != y_hat])
            self.w[i] = 0.5 * np.log((1 / err_t) - 1)
            D = np.multiply(D, np.exp(-y * self.w[i] * y_hat))  # update sample weights
            D = D / np.sum(D)
        return D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """

        predictions = [self.w[i] * self.h[i].predict(X) for i in range(max_t)]
        return np.sign(sum(predictions))

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        """
        pred_vec = self.predict(X, max_t)
        wrong = sum(np.ones(y[y != pred_vec].shape))  # number of wrong classifications
        return wrong / len(y)


def q_13(m=5000, T=500, m_test=200):
    # create data and train model
    X_test, y_test, X_train, y_train, ab, D = init_data_and_mod(T, m, m_test)
    test_err, train_err = [], []

    # get error for each
    for i in range(0, T):
        train_err.append(ab.error(X_train, y_train, i))
        test_err.append(ab.error(X_test, y_test, i))
    T_range = np.arange(T)

    # plots
    plt.plot(T_range, train_err, label='Train Error')
    plt.plot(T_range, test_err, label='Test Error')
    plt.legend()
    plt.grid(True)
    plt.title("Error vs. T")


def q_14(m=5000, m_test=200):
    T_arr = [5, 10, 50, 100, 200, 500]

    for i in range(len(T_arr)):
        T = T_arr[i]
        # create data and train model
        X_test, y_test, X_train, y_train, ab, D = init_data_and_mod(T, m, m_test)

        test_err = ab.error(X_test, y_test, T)

        plt.subplot(2, 3, i + 1)
        decision_boundaries(ab, X_test, y_test, T)
    plt.tight_layout(pad=1.)
    # plt.show()


def init_data_and_mod(T, m, m_test):
    """
    generates data, inits a classifier and trains it.
    :param T: number of decision stumps
    :param m: number of data samples
    :param m_test: number of test samples
    :return: X_test, y_test, X_train, y_train, classifier
    """
    X_train, y_train = generate_data(m, SMALL_NOISE)
    X_test, y_test = generate_data(m_test, SMALL_NOISE)
    ab = AdaBoost(DecisionStump, T)
    D = ab.train(X_train, y_train)
    return X_test, y_test, X_train, y_train, ab, D


def q_15(m=1000, m_test=200, T=500):
    min_err, min_T, model = 1, -1, 0
    X_train, y_train = generate_data(m, SMALL_NOISE)
    X_test, y_test = generate_data(m_test, SMALL_NOISE)
    ab = AdaBoost(DecisionStump, T)
    ab.train(X_train, y_train)
    for max_t in range(T):
        test_err = ab.error(X_test, y_test, max_t)
        if test_err < min_err:
            min_err = test_err
            min_T = max_t
            model = ab
    decision_boundaries(model, X_train, y_train, min_T)
    plt.savefig("q15_plot")
    plt.show()
    return min_err, min_T


def q_16(T, m, m_test):
    X_test, y_test, X_train, y_train, ab, D = init_data_and_mod(T, m, m_test)
    D = D / np.max(D) * 10
    decision_boundaries(ab, X_train, y_train, T, D)
    plt.savefig("q16_plot")
    plt.show()


if __name__ == '__main__':
    q_16(500, 5000, 200)
