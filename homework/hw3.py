import numpy as np
from matplotlib import pyplot as plt


def dtft(x, fs=1, t0=0):
    # Copied directly from pydsm.ft.dtft on GitHub
    # https://github.com/sergiocallegari/PyDSM/blob/master/pydsm/ft.py
    """
    Computes the discrete time Fourier transform (DTFT).

    Returns a function that is the DTFT of the given vector.

    Parameters
    ----------
    x :  array_like
        the 1-D vector to compute the DTFT upon

    Returns
    -------
    X : callable
        a function of frequency as in X(f), corresponding to the DTFT of x

    Other Parameters
    ----------------
    fs : real, optional
        sample frequency for the input vector (defaults to 1)
    t0 : real, optional
        the time when x[0] is sampled (defaults to 0). This is expressed
        in sample intervals.
    """
    return lambda f: np.sum(x*np.exp(-2j*np.pi*f/fs*(np.arange(len(x))-t0)))


def a():
    fs = 2E3  # Hz
    f0 = 50   # Hz
    f1 = 150  # Hz
    phi0 = 0
    phi1 = 0

    n = np.arange(2000)
    t = n / fs

    xn = 2 * np.cos(2 * np.pi * f0 * t + phi0) +\
        2 * np.cos(2 * np.pi * f1 * t + phi1)

    plt.stem(n[:50], xn[:50])
    plt.show()

    ws = np.linspace(-np.pi, np.pi, len(n))
    fft_func = dtft(xn, fs=fs)
    xw = np.array([fft_func(w) for w in ws])

    # Plotting code copied directly from scipy documentation example
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html
    fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')
    ax1.plot(ws, np.abs(xw), 'b')  # 20 * np.log10(abs(xw)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [radians]')

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(xw))
    ax2.plot(ws, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid(True)
    ax2.axis('tight')
    plt.show()

    # Part C
    x1sn = np.array([xn[4 * k] for k in np.arange(500)])
    plt.stem(np.arange(50), x1sn[:50])
    plt.show()

    fft_func = dtft(x1sn, fs=fs/4)  # ??????????
    xw1 = np.array([fft_func(w) for w in ws])

    fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')
    ax1.plot(ws, np.abs(xw1), 'b')  # 20 * np.log10(abs(xw)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [radians]')

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(xw1))
    ax2.plot(ws, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid(True)
    ax2.axis('tight')
    plt.show()


if __name__ == "__main__":
    a()
