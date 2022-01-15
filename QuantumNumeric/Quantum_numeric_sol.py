import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from scipy.special import genlaguerre as L
from sympy import *


def calc_R_nl(r, n, l):
    """
    calculates radial part of the basis
    :param r: radius of the atom
    :param n: int - quantum number of energy level
    :param l: int - quantum number of angular momentum, must be < n
    """
    rho = 2 * r
    poly_deg = n - l - 1
    in_sqrt = 8 * ((np.math.factorial(poly_deg)) / (np.math.factorial(n + l + 1)))
    outer = rho * L(poly_deg, 2 * l + 2)(rho) * np.exp(-rho / 2)
    return r * np.sqrt(in_sqrt) * outer


def lag_weights_roots(n):
    """
    calculates the the weights and roots for gaussian quadrature
    :param n: int - number of roots and weights
    :return: tuple containing two array of size 1xn (x_i, w_i)
    """
    x = Symbol("x")
    roots = Poly(laguerre(n, x)).all_roots()
    x_i = [rt.evalf(20) for rt in roots]
    w_i = [(rt / ((n + 1) * laguerre(n + 1, rt)) ** 2).evalf(20) for rt in roots]
    return x_i, w_i


def calc_g_x(func, x_i):
    return func(x_i) * np.exp(x_i)


def integrate(func):
    """
    integrate over given function using gaussian quadrature
    :param func: function to integrate over using gaussian quadrature
    :return:
    """
    x_i_arr, w_i_arr = lag_weights_roots(5)
    return np.sum([w_i_arr[i] * calc_g_x(func, x_i_arr[i]) for i in range(len(w_i_arr))])


if __name__ == '__main__':


