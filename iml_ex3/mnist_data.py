import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models import Logistic, SVM, DecisionTree
from sklearn.neighbors import KNeighborsClassifier
import time
from comparison import plot_acc_vs_m


def load_data():
    """
    loads mnist dataset
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_images = np.logical_or((y_train == 0), (y_train == 1))
    test_images = np.logical_or((y_test == 0), (y_test == 1))
    x_train, y_train = x_train[train_images], y_train[train_images]
    x_test, y_test = x_test[test_images], y_test[test_images]
    return x_train, y_train, x_test, y_test


def rearrange_data(X):
    """
    flattens a given array
    :param X:data of size n x 28 x 28
    :return: data of size n x 784
    """
    return X.reshape(X.shape[0], 784)


def calc_accuracy(x_train, y_train, x_test, y_test, times=50):
    """

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param times:
    :return:
    """
    accuracy = []
    mean_time = []
    for m in [50, 100, 300, 500]:
        acc = [0, 0, 0, 0]
        t = [0, 0, 0, 0]
        # generate data
        for i in range(times):
            X, y = None, np.zeros(m)
            while (0 not in y) or (1 not in y):
                indices = np.random.choice(len(x_train), m)
                X, y = x_train[indices], y_train[indices]

            # train models
            new_X = rearrange_data(X)
            new_test = rearrange_data(x_test)
            svm = SVM()
            lgstic = Logistic()
            tree = DecisionTree()
            neigh = KNeighborsClassifier(n_neighbors=5)
            models = [lgstic, svm, tree, neigh]
            for j in range(len(models)):
                s_time = time.time()
                models[j].fit(new_X, y)
                t[j] += time.time() - s_time
                _y = models[j].predict(new_test)
                acc[j] += np.sum(y_test == _y) / len(y_test)
        mean_time.append(np.array([t[0] / times, t[1] / times, t[2] / times, t[3] / times]))
        accuracy.append(np.array([acc[0] / times, acc[1] / times, acc[2] / times, acc[3] / times]))
    return accuracy, np.array(mean_time)


