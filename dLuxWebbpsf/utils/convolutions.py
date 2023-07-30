from jax import vmap
import jax.numpy as np
from jax.scipy.signal import convolve, correlate

"""
This is a copy of the scipy gaussian filter, but with jax functions.
This should eventually be moved into the jax main repo.
"""


def gaussian_kernel_1d(sigma):
    """
    Computes a 1-d Gaussian convolution kernel.
    """
    radius = int(4.0 * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma**2 * x**2)
    return phi_x / phi_x.sum()


# Convolution
def conv_axis(arr, kernel, axis):
    """
    Convolves the kernel along the axis
    """
    return vmap(convolve, in_axes=(axis, None, None))(arr, kernel, "same")


def gaussian_filter(arr, sigma):
    """
    Applies a 1d gaussian filter along each axis of the input array

    Note this currently does not work correctly near the edges
    """
    kernel = gaussian_kernel_1d(sigma)[::-1]
    k = len(kernel) // 2
    arr = np.pad(arr, k, mode="symmetric")

    for i in range(arr.ndim):
        arr = conv_axis(arr, kernel, i)
    return arr[tuple([slice(k, -k, 1) for i in range(arr.ndim)])]


# Correlation
def corr_axis(arr, kernel, axis):
    """
    Correlates the kernel along the axis
    """
    return vmap(correlate, in_axes=(axis, None, None))(arr, kernel, "same")


def gaussian_filter_correlate(arr, sigma):
    """
    Applies a 1d gaussian filter along each axis of the input array
    """
    kernel = gaussian_kernel_1d(sigma)[::-1]
    k = len(kernel) // 2
    arr = np.pad(arr, k, mode="symmetric")

    for i in range(arr.ndim):
        arr = corr_axis(arr, kernel, i)
    return arr[tuple([slice(k, -k, 1) for i in range(arr.ndim)])]
