import numpy as np
import matplotlib.pyplot as plt


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

