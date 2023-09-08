from typing import Callable

from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt


def a(x):
    """
    normal form: A*sin(2pi * frequency * x)
    So frequency is 5 --> pump at 5 frequency in fourier transform is expected
    :param x:
    :return:
    """
    return np.sin(10 * np.pi * x)


def b(x):
    """
    Signal is composed of two sine-waves: so two bumps are expected:
    --> one hat 0.2 the other at 1.5
    :param x:
    :return:
    """
    return np.sin(0.4 * np.pi * x) + 3 * np.sin(3 * np.pi * x)


def c(x):
    """
    gaussian basic form: a * exp(- (x - b)**2 / 2c**2)
    a - height of the curves peak
    b - position of center of the peak
    c - width of the function(bell)


    --> F[exp(-a * x**2)] --> sqrt(pi / a) * exp(-pi**2 * k**2 / a)

    --> a von 1 zu: sqrt(pi * 2)  ~ 2.5 => higher peak
    --> 2c**2 von -1/2 zu - pi**2 * 2  => more narrow
    :param x:
    :return:
    """
    return np.exp(-x ** 2 / 2)


def d(x):
    """
    see c:

    --> a von 1 zu sqrt(pi * 16) => even higher peak
    --> 2c**2 von - 1/16 zu - pi**2 * 16 => even more narrow
    :param x:
    :return:
    """

    return np.exp(-x ** 2 / 16)


def e(x):
    """
    higher peak
    :param x:
    :return:
    """
    return 5 * c(x)


def fourier(f: Callable,
            T: int = 30,
            dt: float = 0.001,
            show_n: int = 200):
    """
    T - Interval length
    dt - sample rate
    :param T:
    :param dt:
    :return:
    """

    # number of points
    N = int(T / dt)
    # points for evaluation
    x = np.linspace(-T / 2, T / 2, N, endpoint=False)
    # evaluate function
    y = f(x)
    # calculate DFT
    yf = fft(y)
    # get frequencies ( depends on sampling rate )
    xf = fftfreq(N, dt)
    print(x, x.shape)
    print(y, y.shape)
    print(xf, xf.shape)
    print(yf, yf.shape)
    # plot DFT for positive frequencies
    #fig = plt.figure()
    #fig = plt.figure(figsize=(3, 1))
    plt.plot(xf[:show_n], 2.0 / N * np.abs(yf[:show_n]),
                #s=1
                )
    plt.xlabel(r" Frequency $\ omega$ ( cycles per unit length )")
    plt.ylabel(r"|F($\ omega$ )| ( Amplitude )")
    plt.title(f.__name__)
    plt.show()
    plt.plot(x[:show_n], y[:show_n])
    plt.show()


if __name__ == "__main__":
    #fourier(a, T=30)
    #fourier(b, T=30)
    xx = 40
    yy = 0.00001
    zz = int(int(xx / yy) // 2)
    fourier(a, T=xx, show_n=zz, dt=yy)
    #fourier(d, T=4, show_n=400, dt=0.01)
    #fourier(e, T=4, show_n=400, dt=0.01)

