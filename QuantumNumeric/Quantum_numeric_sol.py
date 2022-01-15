import numpy as np
from sympy.physics.hydrogen import E_nl, R_nl
from scipy.misc import derivative
from numpy import linalg as LA
from matplotlib import pyplot as plt
from scipy.special.orthogonal import genlaguerre as L

from sympy import *


def calc_R_nl(r, n, l):
    """
    calculates radial part of the basis
    :param r: radius of the atom
    :param n: int - quantum number of energy level
    :param l: int - quantum number of angular momentum, must be < n
    """
    # rho = float(2 * r)
    # poly_deg = n - l - 1
    # in_sqrt = 8 * ((np.math.factorial(poly_deg)) / (np.math.factorial(n + l + 1)))
    # outer = (rho ** l) * L(poly_deg, 2 * l + 2)(np.around(rho, 20)) * np.exp(-rho / 2)
    # return np.sqrt(in_sqrt) * outer
    return R_nl(n, l, r)


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


def q1_func(r, n, m, l1, l2, q3=False):
    """
    helper function for question 1
    :return: r^2 * R_n_l(r) * R_m_l(r)
    """
    ret = (r ** 2) * calc_R_nl(r, n, l1) * calc_R_nl(r, m, l1) if not q3 else (r ** 2) * calc_R_nl(r, n, l1) * (
            1 / r) * calc_R_nl(r, m, l2)
    return ret


def integrate_q1(num_roots, n, m, l1, l2, q3=False, func=q1_func):
    """
    integrate over given function using gaussian quadrature
    :param l1:
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
        f_x = func(x_i, n, m, l1, l2, q3) * np.exp(float(x_i))
        sum += w_i_arr[i] * f_x
    return sum


def deriv_R_nl(r, n, l):
    """

    :param r:
    :param n:
    :param l:
    :return:
    """

    # rho, poly_deg = np.around(float(2 * r), 20), n - l - 1
    # in_sqrt = 8 * ((np.math.factorial(poly_deg)) / (np.math.factorial(n + l + 1)))
    # outer1 = 2 * l * (rho ** (l - 1) * L(poly_deg, 2 * l + 2)(rho)) * np.exp(-rho / 2)
    # outer2 = (rho ** l) * np.exp(-rho / 2) if n - l - 2 < 0 else (rho ** l) * L(poly_deg - 1, 2 * l + 3)(rho) * np.exp(
    #     -rho / 2)
    # outer3 = (rho ** l) * L(poly_deg, 2 * l + 2)(rho) * np.exp(-rho / 2)
    # return np.sqrt(in_sqrt) * (outer1 - 2 * outer2 - outer3)
    def f(x):
        return R_nl(n, l, x)

    return derivative(f, r, dx=1e-7)


def integrate_q2(n1, n2, l1, l2, num_roots):
    """
    calculates the kinetic energy using numerical evaluation of the integral
    """
    delta = 1 if l1 == l2 else 0
    integral1, integral2 = 0, 0
    x_i_arr, w_i_arr = lag_weights_roots(num_roots)
    for i in range(num_roots):
        x_i = x_i_arr[i]
        integral1 += w_i_arr[i] * (x_i ** 2) * deriv_R_nl(x_i, n1, l1) * deriv_R_nl(x_i, n2, l2) * np.exp(float(x_i))
        integral2 += w_i_arr[i] * calc_R_nl(x_i, n1, l1) * calc_R_nl(x_i, n2, l2) * np.exp(float(x_i))
    return delta * (0.5 * integral1 + 0.5 * l1 * (l1 + 1) * integral2)


def integrate_q3(n, m, l1, l2, num_roots):
    """
    numerically evaluates the matrix elements of the electrical potential
    :return:
    """
    delta = 1 if l1 == l2 else 0
    return - delta * integrate_q1(num_roots, n, m, l1, l2, True)


def calc_hamiltonian(N=21):
    """
    calculates the hamiltonian matrix
    :param N: int - size of hamiltonian will be NxN
    """
    H = np.zeros((N, N))
    for i in range(1, N):
        for j in range(1, N):
            H[i, j] = calc_H_cell(i, j)

    # diagonlize
    w, v = LA.eig(H[1:, 1:])  # ignore first row and col
    return w, v


def calc_H_cell(n1, n2, num_roots=20, l=0, ignore_B=True):
    """

    :param n1:
    :param n2:
    :param num_roots:
    :param l:
    :param ignore_B:
    :return:
    """
    T_n1n2 = integrate_q2(n1, n2, l, l, num_roots)
    C_n1n2 = integrate_q3(n1, n2, l, l, num_roots)
    return T_n1n2 + C_n1n2 if ignore_B else 0  # TODO: handle case with magnetic field


def calc_W(n1, n2, l1, l2, num_roots=20):
    """
    calculates the integral expression for W
    """
    ret_sum = 0
    x_i_arr, w_i_arr = lag_weights_roots(num_roots)
    for i in range(num_roots):
        x_i = x_i_arr[i]
        ret_sum += w_i_arr[i] * R_nl(n1, l1, x_i) * (x_i ** 4) * R_nl(n2, l2, x_i) * np.exp(float(x_i))
    return ret_sum


def calc_I(l1, l2):
    """

    :param l1:
    :param l2:
    :return:
    """
    temp = l1 - l2
    match temp:
        case temp if temp == -2:
            return (-1 / (np.sqrt((2 * l1 + 1) * (2 * l1 + 5)))) \
                ((l1 + 2) * (l1 + 1) / (2 * l1 + 3))
        case temp if temp == 0:
            return 1 - ((l1 + 1) ** 2) / ((2 * l1 + 3) * (2 * l1 + 1)) - \
                   l1 ** 2 / ((2 * l1 + 1) * (2 * l1 - 1))
        case temp if temp == 2:
            return -l1 * (l1 - 1) / ((2 * l1 - 1) * np.sqrt((2 * l1 + 1) * (2 * l1 - 3)))


def calc_B_nl(n1, n2, l1, l2):
    """
    calculates the matrix element of the energy added by the magnetic field

    """
    return calc_I(l1, l2) * calc_W(n1, n2, l1, l2)


def Q8(nmax=12):
    


if __name__ == '__main__':
    x_arr = [i for i in range(1, 21)]
    # n_eq_m_arr, n_neq_m_arr = [], []
    # for x in x_arr:
    #     n_eq_m_arr.append(integrate_q1(x, 3, 3, 1, 1))
    #     n_neq_m_arr.append(integrate_q1(x, 3, 2, 1, 1))
    # fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    # plt.xlabel("# of roots")
    # plt.subplots_adjust(hspace=.0)
    # ax1.set_title(r'$\int r^{2}*R_{nl}*R_{ml}$ vs # of roots')
    # ax1.scatter(x_arr, n_eq_m_arr, c='firebrick', label=r'n=m'), ax1.legend(), ax1.grid()
    # ax2.scatter(x_arr, n_neq_m_arr, c='navy', label=r'$n \neq m$'), ax2.legend(), ax2.grid()
    # # plt.savefig('Q1 figure')
    # plt.show()
    w, v = calc_hamiltonian()
    # print(np.diag(w))

    plt.scatter(x_arr, [E_nl(i) for i in range(1, 21)], alpha=0.5, c='hotpink', label='Ideal'), plt.xlabel(
        r'n'), plt.ylabel(r'$E_n$')
    plt.scatter(x_arr, w, c='navy', alpha=0.5, label='Numeric'), plt.legend(), plt.title(r'$E_n$ vs. n'), plt.show()
