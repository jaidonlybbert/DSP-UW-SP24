import scipy
from scipy import signal
from matplotlib import pyplot as plt
import math
import numpy as np
from ee518_custom_fct import zplane


def m1():
    a = [1, -4, 1.25, 6 - 5 / 4, 3 / 2]
    b = [1,  0,    0,         0,     0]

    (r, p, k) = scipy.signal.residuez(b, a)

    print("a: ", a)
    print("b: ", b)
    print("r: ",  r)
    print("p: ", p)
    print("k: ", k)


def m2():
    a = [1, -1.85 * math.cos(math.pi / 18), 0.83]
    b = [1, 1 / 3, 0]

    (w, h) = signal.freqz(b, a)

    # Plotting code copied directly from scipy documentation example
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html
    fig, ax1 = plt.subplots()
    ax1.set_title('M2.B Digital filter frequency response')
    ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [rad/sample]')

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid(True)
    ax2.axis('tight')
    zplane(b, a)
    plt.show()


if __name__ == "__main__":
    m1()
    m2()
