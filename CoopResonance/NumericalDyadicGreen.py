import numpy as np
from numpy.linalg import norm, eig
from matplotlib import pyplot as plt
from numba import njit
import pandas as pd

gamma = 24e6


def set_sim(N=2500, _lambda=0.5e-6, fourier=False):
    """

    :param fourier: boolean, if False - diagonalize G normally, else - move to fourier space.
    :param _lambda: wave length.
    :param N: int, number of atoms. assumes array is square so sqrt(N) must be an inb aswell.
    :return:
    """
    # a_v = np.arange(0.2 * _lambda, _lambda, 0.02)  # different lattice constants
    a_v = np.linspace(0.2 * _lambda, _lambda, num=100)
    ret, specific_a = [], None
    k = 2 * np.pi / _lambda
    for i, curr_a in enumerate(a_v):  # iterate over different lattice constants
        a, b = curr_a, curr_a
        R = np.array(create_2d_arr(N, a, b))
        G = np.zeros((N, N), dtype=complex)  # init empty array
        G = calc_tot_G(G, N, np.array(R), curr_a, k)
        if fourier:
            fourier_G = np.fft.fft2(G)
            # print(
            #     f'The matrix is diagonal i k-space: {np.count_nonzero(fourier_G - np.diag(np.diagonal(fourier_G))) == 0}')  # sanity check
            g_k = np.sum(fourier_G[0:1, :])

        else:  # diagonalize numerically and use eigen values.
            w, v = eig(G)  # find e-values and e-vectors
            idx = np.argsort(w)
            w = w[idx]  # sort eigen values
            v = v[:, idx]  # sort eigen-vectors
            plt.plot(np.real(v.T[0]) * gamma * _lambda, label="Real"), plt.plot(gamma * _lambda * np.imag(v.T[0]),
                                                                                label="Imaginary"), plt.title(
                fr"e-value = {np.around(w[0] / gamma, 3)}, a = {np.around(curr_a / _lambda, 3)}*$\lambda$")
            plt.show()
            g_k = np.sum(G[0:1, :])
        big_Delta, big_Gamma = (-3 / 2) * gamma * _lambda * np.real(g_k), 3 * gamma * _lambda * np.imag(g_k)
        if np.around(curr_a / _lambda, 2) == 0.62:
            specific_a = w, np.around(curr_a / _lambda, 2)
        ret.append([big_Delta, curr_a, big_Gamma, w[0]])

        print(f'{np.around((i / len(a_v) * 100), 2)}% Done')
    return ret, specific_a


@njit
def calc_tot_G(G, N, loc_arr, a, k):
    """

    :param G:
    :param N:
    :param loc_arr:
    :param a:
    :param k:
    :return:
    """
    for i in range(N):
        for j in range(i + 1, N):
            x_ij_pol = (a ** 2) * ((loc_arr[i][0] - loc_arr[j][0]) ** 2)  # TODO: ask Rivka why is that?
            x_ji_pol = (a ** 2) * ((loc_arr[j][0] - loc_arr[i][0]) ** 2)
            r_rel_ij = norm(loc_arr[i] - loc_arr[j])
            r_rel_ji = norm(loc_arr[j] - loc_arr[i])
            G[i, j] = calc_single_G_cell(k, r_rel_ij, x_ij_pol)
            G[j, i] = calc_single_G_cell(k, r_rel_ji, x_ji_pol)

    return G


@njit
def calc_single_G_cell(k, r_n, x_pol):
    """
    calculates single cell in G
    :param k: float - wave number
    :param r_n: float - norm of the relative vector v_i - v_j
    :param x_pol: the x polarization of that cell
    """
    outer = np.exp(1j * k * r_n) / (4 * np.pi * r_n)
    inner1 = 1 + (1j * k * r_n - 1) / ((k ** 2) * (r_n ** 2))
    inner2 = (3 - 3 * 1j * k * r_n - (k ** 2) * (r_n ** 2)) / ((k ** 2) * (r_n ** 2)) * (x_pol / (2 * (r_n ** 2)))
    return outer * (inner1 + inner2)


@njit
def create_2d_arr(N, a, b):
    """
    inits an array of size [N,3]. each row is the location of an atom in the
    array.
    :param N: int - total number of atoms, assumes sqrt(N) is an int
    :param a: int - lattice constant in x axis
    :param b: int - lattice constant in y axis
    """
    ret = []
    for i in range(int(np.sqrt(N))):
        for j in range(int(np.sqrt(N))):
            ret.append(np.array([a * j, b * i, 0]))
    return ret


if __name__ == '__main__':
    _lambda = 0.5e-6
    output, spec = set_sim()
    output = np.array(output)
    big_del, a_arr = output[:, 0], output[:, 1]  # take first column

    plt.plot(a_arr / _lambda, big_del / gamma), plt.ylabel(r"$\Delta/\gamma$"), plt.xlabel(
        r"a/$\lambda$")
    plt.axhline(y=0, c='firebrick', linestyle='--')
    plt.show()

    # plot min{Re(E_i)} vs a
    e_val_arr = output[:, 3]
    plt.plot(a_arr / _lambda, np.real(e_val_arr) / gamma), plt.ylabel(r"$E_{a}/\lambda$"), plt.xlabel(
        r'a/$\lambda$')
    plt.title(r'min{Re($E_{a}$)} vs. a/$\lambda$')
    plt.show()

    # plot eigen values for specific a
    plt.plot(np.real(spec[0]) / gamma), plt.ylabel(r'Eigen value'), plt.title(
        f"Eigen values for a = {np.around(np.real(spec[1]), 3)}")
    plt.show()
