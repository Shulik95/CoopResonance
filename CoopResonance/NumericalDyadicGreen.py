import numpy as np
from numpy.linalg import norm, eig
from matplotlib import pyplot as plt
from numba import njit

gamma = 24e6


def set_sim(N=81, _lambda=24e-6):
    """

    :param _lambda:
    :param N:
    :return:
    """
    # a_v = np.arange(0.2 * _lambda, _lambda, 0.02)  # different lattice constants
    a_v = np.linspace(0.2 * _lambda, _lambda, num=100)
    ret = []
    k = 2 * np.pi / _lambda
    for i, curr_a in enumerate(a_v):  # iterate over different lattice constants
        a, b = curr_a, curr_a
        R = create_2d_arr(N, a, b)
        G = calc_tot_G(N, R, curr_a, k)
        w, v = eig(G)  # find e-values and e-vectors

        plt.scatter(range(N), np.real(w) / gamma, label="real"), plt.scatter(range(N), np.imag(w) / gamma,
                                                                             label='imag'), plt.title(
            f"a = {curr_a}")
        if i ==0:
            plt.plot(np.real(v[:, 20]))
        plt.legend()
        plt.show()
        # according to Shamoon
        # rhs = (-3 / 2) * gamma * _lambda * (G[0].sum())
        # big_delta = np.real(rhs)
        # big_gamma = 2 * np.imag(rhs)
        # ret.append([big_delta, curr_a, big_gamma])
    return ret


def calc_tot_G(N, loc_arr, a, k):
    """

    :param N:
    :param loc_arr:
    :param a:
    :param k:
    :return:
    """
    G = np.zeros((N, N), dtype=complex)  # init empty array
    loc_arr = np.array(loc_arr)
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
    return np.array(ret)


if __name__ == '__main__':
    _lambda = 0.5e-6
    output = np.array(set_sim())
    big_del, a_arr = output[:, 0], output[:, 1]  # take first column
    plt.plot(a_arr / _lambda, big_del / gamma), plt.ylabel(r"$\Delta/\gamma$"), plt.xlabel(r"a/$\lambda$")
    plt.show()
