import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
from jax import Array
from dLuxWebbpsf import utils
from jax.scipy.ndimage import map_coordinates


"""
Script for all the various forms of detector transformations required to model
webbpsf.
"""

__all__ = [
    "ApplyJitter",
    "Rotate",
    "DistortionFromSiaf",
    "ApplySiafDistortion",
]


class ApplyJitter(dl.detector_layers.DetectorLayer):
    """
    Applies a gaussian jitter to the image. This is designed to match the
    scipy.ndimage.gaussian_filter function in the same vein of webbpsf.
    """

    sigma: Array

    def __init__(self, sigma):
        super().__init__()
        self.sigma = float(sigma)

    def __call__(self, image):
        # Convert sigma to pixels, note this assumes sigma has the same units
        # as the pixel scale
        jitter_pix = self.sigma / image.pixel_scale
        jittered = utils.gaussian_filter_correlate(
            image.image, jitter_pix, ksize=3
        )
        return image.set("image", jittered)


class Rotate(dl.RotateDetector):
    """
    A simple rotation layer that overwrites the default dLux rotation layer
    in order to apply a cubic-spline interpolation.
    """

    def __call__(self, image):
        rotated = utils.rotate(image.image, self.angle, order=3)
        return image.set("image", rotated)


class DistortionFromSiaf:
    def __new__(cls, aper, oversample):
        degree = aper.Sci2IdlDeg + 1
        coeffs_dict = aper.get_polynomial_coefficients()
        coeffs = np.array([coeffs_dict["Sci2IdlX"], coeffs_dict["Sci2IdlY"]])
        sci_refs = np.array([aper.XSciRef, aper.YSciRef])
        sci_cens = (
            np.array([aper.XSciRef, aper.YSciRef]) + 0.5
        )  # Note this may not be foolproof
        pixelscale = np.array(
            [aper.XSciScale, aper.YSciScale]
        ).mean()  # Note this may not be foolproof
        return ApplySiafDistortion(
            degree,
            coeffs,
            sci_refs,
            sci_cens,
            pixelscale / oversample,
            oversample,
        )


class ApplySiafDistortion(dl.detector_layers.DetectorLayer):
    """
    Applies Science to Ideal distortion following webbpsf/pysaif
    """

    degree: int
    Sci2Idl: float
    SciRef: float
    sci_cen: float
    pixel_scale: float
    oversample: int
    xpows: Array
    ypows: Array

    def __init__(
        self, degree, Sci2Idl, SciRef, sci_cen, pixel_scale, oversample
    ):
        super().__init__("ApplySiafDistortion")
        self.degree = int(degree)
        self.Sci2Idl = float(Sci2Idl)
        self.SciRef = float(SciRef)
        self.sci_cen = float(sci_cen)
        self.pixel_scale = float(pixel_scale)
        self.oversample = int(oversample)
        self.xpows, self.ypows = self.get_pows()

    def get_pows(self):
        n = dlu.triangular_number(self.degree)
        vals = np.arange(n)

        # Ypows
        tris = dlu.triangular_number(np.arange(self.degree))
        ydiffs = np.repeat(tris, np.arange(1, self.degree + 1))
        ypows = vals - ydiffs

        # Xpows
        tris = dlu.triangular_number(np.arange(1, self.degree + 1))
        xdiffs = np.repeat(n - np.flip(tris), np.arange(self.degree, 0, -1))
        xpows = np.flip(vals - xdiffs)

        return xpows, ypows

    def distort_coords(
        self,
        A,
        B,
        X,
        Y,
    ):
        """
        Applts the distortion to the coordinates
        """

        # Promote shapes for float inputs
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        # Exponentiate
        Xpow = X[None, :, :] ** self.xpows[:, None, None]
        Ypow = Y[None, :, :] ** self.ypows[:, None, None]

        # Calcaulate new coordinates
        Xnew = np.sum(A[:, None, None] * Xpow * Ypow, axis=0)
        Ynew = np.sum(B[:, None, None] * Xpow * Ypow, axis=0)
        return Xnew, Ynew

    def apply_Sci2Idl_distortion(self, image):
        """
        Applies the distortion from the science (ie images) frame to the
        idealised telescope frame
        """
        # Convert sci cen to idl frame
        xidl_cen, yidl_cen = self.distort_coords(
            self.Sci2Idl[0],
            self.Sci2Idl[1],
            self.sci_cen[0] - self.SciRef[0],
            self.sci_cen[1] - self.SciRef[1],
        )

        # Get paraxial pixel coordinates and detector properties
        xarr, yarr = dl.utils.get_pixel_positions(tuple(image.shape))

        # Scale and shift coordinate arrays to 'idl' frame
        xidl = xarr * self.pixel_scale + xidl_cen
        yidl = yarr * self.pixel_scale + yidl_cen

        # Scale and shift coordinate arrays to 'sci' frame
        xnew = xarr / self.oversample + self.sci_cen[0]
        ynew = yarr / self.oversample + self.sci_cen[1]

        # Convert requested coordinates to 'idl' coordinates
        xnew_idl, ynew_idl = self.distort_coords(
            self.Sci2Idl[0],
            self.Sci2Idl[1],
            xnew - self.SciRef[0],
            ynew - self.SciRef[1],
        )

        # Create interpolation coordinates
        centre = (xnew_idl.shape[0] - 1) / 2

        coords_distort = (
            np.array([ynew_idl - yidl_cen, xnew_idl - xidl_cen])
            / self.pixel_scale
        ) + centre

        # Apply distortion
        return map_coordinates(image, coords_distort, order=1)

    def __call__(self, image):
        image_out = self.apply_Sci2Idl_distortion(image.image)
        return image.set("image", image_out)
