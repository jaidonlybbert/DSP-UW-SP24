from matplotlib import pyplot as plt
import scipy
import numpy as np


def part_one():
    h = [1, 1, 1, -1, 1]
    h = np.array([h, np.arange(0, len(h), 1)], dtype=np.int64)
    x = [1, -1, 1, 1, 1]
    x = np.array([x, np.arange(0, len(x), 1)], dtype=np.int64)
    y = scipy.signal.convolve(h[0], x[0])
    y = np.array([y, np.arange(0, len(y), 1)], dtype=np.int64)
    fig = plt.figure(figsize=(7, 7), layout='constrained')
    axs = fig.subplot_mosaic([["h", "x"],
                              ["y", "y"]])
    fig.suptitle("Part 1: Convolution")

    axs["h"].stem(h[1], h[0])
    axs["h"].set_title("h[n]")
    axs["x"].stem(x[1], x[0])
    axs["x"].set_title("x[n]")
    axs["y"].stem(y[1], y[0])
    axs["y"].set_title("y[n]")
    fig.align_labels(axs=None)
    plt.show()


def part_two():
    x = np.zeros((1, 51), dtype=np.float64)
    x[0, 0] = 1
    print(x)
    a = np.array([1., -1.7163, 1.1724, -0.2089], dtype=np.float64)
    b = np.array([0.5264, -1.5224,  1.5224, -0.5264], dtype=np.float64)
    y = scipy.signal.lfilter(b, a, x)
    print(y.shape)
    print(y)

    fig = plt.figure(figsize=(7, 7), layout='constrained')
    axs = fig.subplot_mosaic([["mag"],
                              ["phase"]])
    fig.suptitle("Part 2: Difference Equation")
    w, h = scipy.signal.freqz(a, b)
    axs["mag"].plot(w, abs(h))
    axs["mag"].set_xlabel("Frequency [rad/sample]")
    axs["mag"].set_ylabel("Amplitude")
    axs["phase"].plot(w, np.angle(h))
    axs["phase"].set_ylabel("Angle (radians)")
    axs["phase"].set_xlabel("Frequency [rad/sample]")
    plt.show()


if __name__ == "__main__":
    part_one()
    part_two()
