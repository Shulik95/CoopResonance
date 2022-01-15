import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



if __name__ == '__main__':
    # lab code #
    # get data #
    df = pd.read_excel("200Hz_workFile.xlsx", sheet_name="sine100Hz", index_col=0)
    time = np.array([item for item in df._iter_column_arrays()])[0]
    data = np.array([item for item in df._iter_column_arrays()])[1]
    # print(time)
    # print("done")
    # print(data)

    #     # get time interval and num of points #
    N = len(time)
    dt = time[1] - time[0]
    xf = np.linspace(0.0, 1.0 / (2.0 * dt), N // 2)
    freq = 200
    aq_sin = np.sin(2 * np.pi * freq * time)
    transformed = np.fft.fft(aq_sin)
    # plot #
    plt.subplot(1, 2, 1)
    plt.title("Sine - 200Hz")
    plt.plot(time, data)
    plt.xlabel("Time(s)")
    plt.subplot(1, 2, 2)
    plt.title("Fourier Sine")
    plt.ylabel('Amplitude')
    plt.xlabel(' f(Hz) ')
    plt.plot(xf, (2 / N) * np.abs(transformed[:N // 2]), color='firebrick', label='sin({}Hz)'.format(freq))

    plt.xlim([100, 300])
    plt.tight_layout(pad=1.0)
    plt.savefig("200Hz_Sine")
    plt.show()
