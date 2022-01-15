import numpy as np
from matplotlib import pyplot as plt
from scipy.special import genlaguerre as L
from sympy import *


def calc_R_nl(r, n, l):
    """
    calculates radial part of the basis
    :param r: radius of the atom
    :param n: int - quantum number of energy level
    :param l: int - quantum number of angular momentum, must be < n
    """
    rho = float(2 * r)
    poly_deg = n - l - 1
    in_sqrt = 8 * ((np.math.factorial(poly_deg)) / (np.math.factorial(n + l + 1)))
    outer = (rho ** l) * L(poly_deg, 2 * l + 2)(np.around(rho, 20)) * np.exp(-rho / 2)
    return np.sqrt(in_sqrt) * outer


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
    """
    helper function for integration
    :param func: original function to integrate over
    :param x_i: i'th root of laguerre poly
    :return:
    """
    return func(x_i) * np.exp(x_i)


def q1_func(r, n, m, l):
    """
    helper function for question 1
    :return: r^2 * R_n_l(r) * R_m_l(r)
    """
    return (r ** 2) * calc_R_nl(r, n, l) * calc_R_nl(r, m, l)


def integrate_q1(num_roots, n, m, l, func=q1_func):
    """
    integrate over given function using gaussian quadrature
    :param l:
    :param m:
    :param n:
    :param num_roots:
    :param func: function to integrate over using gaussian quadrature
    :return:
    """
    sum = 0
    x_i_arr, w_i_arr = lag_weights_roots(num_roots)
    for i in range(num_roots):
        x_i = x_i_arr[i]
        f_x = func(x_i, n, m, l) * np.exp(float(x_i))
        sum += w_i_arr[i] * f_x
    return sum


def deriv_R_nl(r, n, l):
    """

    :param r:
    :param n:
    :param l:
    :return:
    """
    rho, poly_deg = float(2 * r), n - l - 1
    in_sqrt = 8 * ((np.math.factorial(poly_deg)) / (np.math.factorial(n + l + 1)))
    outer1 = l * (rho ** (l - 1) * L(poly_deg, 2 * l + 2)(np.around(rho, 20))) * np.exp(-rho / 2)
    outer2 = (rho ** l) * L(poly_deg - 1, 2 * l + 3)(np.around(rho, 20)) * np.exp(-rho / 2)
    outer3 = (rho ** l) * L(poly_deg, 2 * l + 2)(np.around(rho, 20)) * np.exp(-rho / 2)
    return in_sqrt * (outer1 - outer2 - outer3)


def integrate_q2(r, n1, n2, l1, l2, num_roots):
    """
    calculates the kinetic energy using numerical evaluation of the integral
    """
    delta = 1 if l1 == l2 else 0
    integral1, integral2 = 0, 0
    x_i_arr, w_i_arr = lag_weights_roots(num_roots)
    for i in range(num_roots):
        x_i = x_i_arr[i]
        integral1 += w_i_arr[i] * (r**2) * deriv_R_nl(r, n1, l1) * deriv_R_nl(r, n2, l2) * np.exp(float(x_i))
        integral2 += w_i_arr[i] * calc_R_nl(r, n1, l1) * calc_R_nl(r, n2, l2) * np.exp(float(x_i))
    return delta*(0.5 * integral1 + 0.5 * l1 * (l1 + 1) * integral2)


if __name__ == '__main__':
    x_arr = [i for i in range(1, 21)]
    n_eq_m_arr, n_neq_m_arr = [], []
    for x in x_arr:
        n_eq_m_arr.append(integrate_q1(x, 3, 3, 1))
        n_neq_m_arr.append(integrate_q1(x, 3, 2, 1))
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    plt.xlabel("# of roots")
    plt.subplots_adjust(hspace=.0)
    ax1.set_title(r'$\int r^{2}*R_{nl}*R_{ml}$ vs # of roots')
    ax1.scatter(x_arr, n_eq_m_arr, c='firebrick', label=r'n=m'), ax1.legend(), ax1.grid()
    ax2.scatter(x_arr, n_neq_m_arr, c='navy', label=r'$n \neq m$'), ax2.legend(), ax2.grid()
    plt.savefig('Q1 figure')
    plt.show()
