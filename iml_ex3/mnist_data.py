import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models import Logistic, SVM, DecisionTree
from sklearn.neighbors import KNeighborsClassifier


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


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    




