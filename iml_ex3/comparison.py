import numpy as np
import matplotlib.pyplot as plt
from models import Perceptron, SVM, LDA


def draw_points(m):
    """

    :param m: integer
    :return: a tuple - X,y where x is a m X 2 matrix where each columns
    represents and i.i.d sample from the distribution and y is a binary label
    vector.
    """
    const_vec = np.array([0.3, -0.5])
    mean = np.zeros(2)
    _cov = np.identity(2)
    X = np.random.multivariate_normal(mean, _cov, m)
    y = np.apply_along_axis(lambda x: np.sign(np.dot(const_vec, x) + 0.1), 1, X)
    return X, y


def draw_for_real():
    """
    plots hyperplanes for different classifiers for dataset of different size.
    """
    m_arr = [5, 10, 15, 25, 70]
    for m in m_arr:
        X, y = draw_points(m)

        # plot points
        blue_pts, red_pts = X[y == 1], X[y == -1]
        plt.scatter(blue_pts[:, 0], blue_pts[:, 1], c="b")
        plt.scatter(red_pts[:, 0], red_pts[:, 1], c="r")

        # create classifiers
        perc = Perceptron()
        perc.fit(X, y)
        svm = SVM()
        svm.model.fit(X, y)
        w = svm.model.coef_[0]
        a = -w[0] / w[1]

        # create functions
        perc_func = lambda x: -(perc.model[1] * x + perc.model[0]) / perc.model[2]
        true_func = lambda x: -(0.3 * x + 0.1) / (-0.5)
        svm_func = lambda x: a * x - (svm.model.intercept_[0]) / w[1]

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        for _f in [true_func, perc_func, svm_func]:
            plt.plot([x_min, x_max], [_f(x_min), _f(x_max)])
            plt.title("m = " + str(m))
        plt.gca().legend(("true hypothesis", "perceptron", "SVM"))
        plt.savefig("Hyperplane m =" + str(m) + ".jpeg")
        plt.show()


def test_mode_acc(times=50, k=10000, m_arr=(5, 10, 15, 25, 70)):
    """
    tests the accuracy of all models
    :param times: number of times to repeat process
    :param k: number of test points.
    """
    accuracy = []
    for m in m_arr:
        m_acc = [0, 0, 0]
        for i in range(times):
            X, y = None, np.ones(shape=(m,))
            while np.sum(y == 1) == len(y) or np.sum(y == -1) == len(y):
                X, y = draw_points(m)
            X_k, y_k = draw_points(k)

            perc = Perceptron()
            svm = SVM()
            lda = LDA()

            # train classifiers
            models = [perc, svm, lda]
            for j in range(len(models)):
                models[j].fit(X, y)
                _y = models[j].predict(X_k)
                m_acc[j] += np.sum(y_k == _y) / len(y_k)
        accuracy.append(np.array([m_acc[0] / times, m_acc[1] / times, m_acc[2] / times]))
    return accuracy


def plot_acc_vs_m(acc_arr, m_arr, legend_tup):
    """
    plots accuracy as function of training data size.
    :param m_arr:
    :param legend:
    :param acc_arr: python list of tuples of form (perc_acc, svm_acc, lda_acc)
    where each the mean accuracy for a given m.
    """

    as_np = np.array(acc_arr)
    for i in range(as_np.shape[1]):
        y = as_np[:, i]
        plt.plot(m_arr, y)
    plt.legend(legend_tup)
    plt.show()


