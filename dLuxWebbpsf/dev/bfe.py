from functools import partial
from jax import vmap, jit
from dLux.utils import get_pixel_positions
from dLux.detectors import DetectorLayer
import jax.numpy as np


class ApplyBFE(DetectorLayer):
    """
    Applies a non-linear convolution with a gaussian to model the BFE
    """

    m: float
    ksize: int

    def __init__(self, m, ksize=5):
        """
        Constructor
        """
        super().__init__("ApplyBFE")
        self.m = np.asarray(m, dtype=float)
        self.ksize = int(ksize)
        if self.ksize % 2 != 1:
            raise ValueError("ksize must be odd")

    def __call__(self, image):
        """
        This will need to be updated to work for new Image classes
        """
        im_out = self.apply_BFE(image, ksize=self.ksize)
        return im_out

    # Building the kernels
    def pixel_variance(self, pixel_flux):
        """Relationship between the flux and the kernel parameters. Not
        necessarily linear. Will need to be learnt"""
        return self.m * pixel_flux

    def gauss(self, coords, sigma, eps=1e-8):
        """Paceholder gaussian kernel"""
        gaussian = np.exp(
            -np.square(coords / (2 * np.sqrt(sigma) + eps)).sum(0)
        )
        return gaussian / gaussian.sum()

    def build_kernels(self, image, npix=5):
        sigmas = self.pixel_variance(image)
        coords = get_pixel_positions(npix)
        fn = vmap(vmap(self.gauss, in_axes=(None, 0)), in_axes=(None, 1))
        return fn(coords, sigmas)

    @partial(vmap, in_axes=(None, None, None, 0, None, None))
    @partial(vmap, in_axes=(None, None, None, None, 0, None))
    def conv_kernels(self, padded_image, kernels, i, j, k):
        return padded_image[i, j] * kernels[j - k, i - k]

    # Get indexes
    def build_indexs(self, ksize, i, j):
        vals = np.arange(ksize)
        xs, ys = np.tile(vals, ksize), np.repeat(vals, ksize)
        out = np.array([xs, ys]).T
        inv_out = np.flipud(out)
        out_shift = out + np.array([i, j])
        indxs = np.concatenate([out_shift, inv_out], 1)
        return indxs

    @partial(vmap, in_axes=(None, None, None, None, 0))
    @partial(vmap, in_axes=(None, None, None, 0, None))
    def build_and_sum(self, array, ksize, i, j):
        indexes = self.build_indexs(ksize, j, i)
        return vmap(lambda x, i: x[tuple(i)], in_axes=(None, 0))(
            array, indexes
        ).sum()

    def apply_BFE(self, array, ksize=5):
        assert ksize % 2 == 1
        k = ksize // 2
        npix = array.shape[0]
        kernels = self.build_kernels(array, npix=ksize)
        Is, Js = np.arange(k, array.shape[0] + k), np.arange(
            k, array.shape[1] + k
        )
        convd = self.conv_kernels(np.pad(array, k), kernels, Is, Js, k)
        convd_pad = np.pad(convd, ((k, k), (k, k), (0, 0), (0, 0)))
        Is, Js = np.arange(npix), np.arange(npix)
        return self.build_and_sum(convd_pad, ksize, Is, Js)
