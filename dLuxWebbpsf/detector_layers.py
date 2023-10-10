import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
from jax import Array
from jax.scipy.ndimage import map_coordinates
import webbpsf

from . import utils


"""
Script for all the various forms of detector transformations required to model
webbpsf.
"""

__all__ = [
    "ApplyJitter",
    "ApplyChargeDiffusion",
    "Rotate",
    "Convolve",
    "SiafDistortion",
    "DistortionFromSiaf",
]


class ApplyJitter(dl.detector_layers.DetectorLayer):
    """
    Applies a gaussian jitter to the PSF. This is designed to match the
    scipy.ndPSF.gaussian_filter function in the same vein of webbpsf.
    """

    sigma: Array

    def __init__(self, sigma):
        super().__init__()
        self.sigma = float(sigma)

    def apply(self, PSF):
        jitter_pix = self.sigma / dlu.rad2arcsec(PSF.pixel_scale)
        jittered = utils.gaussian_filter_correlate(PSF.data, jitter_pix, ksize=3)
        return PSF.set("data", jittered)


class ApplyChargeDiffusion(dl.detector_layers.DetectorLayer):
    """
    Applies a gaussian filter to the PSF. This is designed to match the
    scipy.ndPSF.gaussian_filter function in the same vein of webbpsf.
    This is to emulate charge diffusion on the detector.
    """

    sigma: float
    kernel_size: int

    def __init__(self, instrument_key: str, kernel_size: int = 11):
        super().__init__()
        charge_def_params = (
            webbpsf.constants.INSTRUMENT_DETECTOR_CHARGE_DIFFUSION_DEFAULT_PARAMETERS
        )
        if instrument_key not in charge_def_params.keys():
            raise ValueError(f"instrument_key must be in {charge_def_params.keys()}")

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer")

        self.kernel_size = int(kernel_size)
        self.sigma = float(charge_def_params[instrument_key])  # arcsec

    def apply(self, PSF):
        """Convert sigma to units of pixels. NOTE: this assumes sigma has
        the same units as the pixel scale."""
        sigma_pix = self.sigma / dlu.rad2arcsec(PSF.pixel_scale)

        diffused = utils.gaussian_filter_correlate(
            PSF.data, sigma_pix, ksize=self.kernel_size
        )
        return PSF.set("data", diffused)


class Rotate(dl.Rotate):
    """
    A simple rotation layer that overwrites the default dLux rotation layer
    in order to apply a cubic-spline interpolation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.order = 3

    def apply(self, PSF):
        rotated = utils.rotate(PSF.data, self.angle, order=3)
        return PSF.set("data", rotated)


# TODO this class may be implemented into dLux proper in which case this
# should be removed.
class Convolve(dl.layers.detector_layers.DetectorLayer):
    """Performs a convolution with a given kernel. Pixel scale of the
    kernel is assumed to be the same as the pixel scale of the PSF."""

    kernel: Array

    def __init__(self, kernel):
        super().__init__()
        self.kernel = np.asarray(kernel, float)

    def apply(self, PSF):
        return PSF.convolve(self.kernel)


def gen_powers(degree):
    """
    Generates the powers required for a 2d polynomial
    """
    n = dlu.triangular_number(degree)
    vals = np.arange(n)

    # Ypows
    tris = dlu.triangular_number(np.arange(degree))
    ydiffs = np.repeat(tris, np.arange(1, degree + 1))
    ypows = vals - ydiffs

    # Xpows
    tris = dlu.triangular_number(np.arange(1, degree + 1))
    xdiffs = np.repeat(n - np.flip(tris), np.arange(degree, 0, -1))
    xpows = np.flip(vals - xdiffs)

    return xpows, ypows


def eval_poly_2d(x, y, A, B, xpows, ypows):
    """
    Evaluate a 2D polynomial of the form: $$ z(x, y) = \Sigma_{i=0}^{deg}
    \Sigma_{j=0}^{deg} \ \ A_{i, j} x^i y^j + B_{i, j} x^j y^i $$
    """
    # Promote shapes for float inputs
    X = np.atleast_2d(x)
    Y = np.atleast_2d(y)

    # Exponentiate
    Xpow = X[None, :, :] ** xpows[:, None, None]
    Ypow = Y[None, :, :] ** ypows[:, None, None]

    # Calcaulate new coordinates
    Xnew = np.sum(A[:, None, None] * Xpow * Ypow, axis=0)
    Ynew = np.sum(B[:, None, None] * Xpow * Ypow, axis=0)
    return Xnew, Ynew


class SiafDistortion(dl.detector_layers.DetectorLayer):
    """
    Applies Science to Ideal distortion following webbpsf/pysaif.

    Note some of these parameters are redundant because this class is designed
    to be potentially extended to the Sci2Idl distortion transform.
    """

    # Coordinates
    Sci2Idl: Array
    Idl2Sci: Array

    # Reference points
    SciRefs: Array
    SciCens: Array

    # Scale
    SciScales: Array
    pixelscale: float
    oversample: int

    # Powers
    pows: Array

    def __init__(
        self,
        degree,
        Sci2Idl,
        Idl2Sci,
        SciRef,
        SciCen,
        SciScale,
        pixelscale,
        oversample,
    ):
        super().__init__()
        # Coefficients
        self.Sci2Idl = np.array(Sci2Idl, float)
        self.Idl2Sci = np.array(Idl2Sci, float)

        # Reference points
        self.SciRefs = np.array(SciRef, float)
        self.SciCens = np.array(SciCen, float)

        # Scale
        self.SciScales = np.array(SciScale, float)
        self.pixelscale = float(pixelscale)
        self.oversample = int(oversample)

        # Powers
        xpows, ypows = gen_powers(int(degree) + 1)
        self.pows = np.array([xpows, ypows])

    def sci_to_idl(self, coords):
        """
        Converts from science frame to ideal frame.
        """
        coords_out = np.array(
            eval_poly_2d(
                coords[0] - self.SciRefs[0],
                coords[1] - self.SciRefs[1],
                self.Sci2Idl[0],
                self.Sci2Idl[1],
                self.pows[0],
                self.pows[1],
            )
        )

        return coords_out + self.SciRefs[:, None, None]

    def idl_to_sci(self, coords):
        """
        Converts from ideal frame to science frame
        """
        return np.array(
            eval_poly_2d(
                coords[0],
                coords[1],
                self.Idl2Sci[0],
                self.Idl2Sci[1],
                self.pows[0],
                self.pows[1],
            )
        )

    def Idl2Sci_transform(self, PSF):
        """
        Applies the Idl2Sci distortion to an input PSF.

        Q: Are xsci_cen and ysci_cen the psf centers or the PSF centers?
        This should be tested as it could require source position input
        """
        # Create a coordiante array for the input PSF
        coords = dlu.nd_coords(PSF.shape)

        # Scale and shift to get to sci coordinates
        coords /= (self.SciScales / self.pixelscale)[:, None, None]
        coords += self.SciCens[:, None, None]

        # Convert coordinates and centers to 'ideal' coordinates
        coords_idl = self.sci_to_idl(coords)
        idl_cens = self.sci_to_idl(self.SciCens)

        # Shift, scale and shift back to get to pixel coordinates
        centres = (np.array(PSF.shape) - 1) / 2
        coords_distort = (coords_idl - idl_cens) / self.pixelscale
        coords_distort += centres[:, None, None]

        # Flip to go from (x, y) indexing to (i, j)
        new_coords = np.flip(coords_distort, 0)

        # Apply distortion
        return map_coordinates(PSF, new_coords, order=1)

    def apply(self, PSF):
        PSF_out = self.Idl2Sci_transform(PSF.data)
        return PSF.set("data", PSF_out)


def DistortionFromSiaf(instrument, optics):
    """
    NOTE: This class assumes that the optsys parameter is already populated
    """
    # Get siaf 'aperture' and reference points
    aper = instrument._detector_geom_info.aperture
    SciRefs = np.array([aper.XSciRef, aper.YSciRef])
    SciScales = np.array([aper.XSciScale, aper.YSciScale])

    # Get distortion polynomial coefficients
    coeffs = aper.get_polynomial_coefficients()
    Sci2Idl = np.array([coeffs["Sci2IdlX"], coeffs["Sci2IdlY"]])
    Idl2Sci = np.array([coeffs["Idl2SciX"], coeffs["Idl2SciY"]])

    # Get detector parameters
    det_plane = optics.planes[-1]
    pixelscale = (det_plane.pixelscale / det_plane.oversample).value

    # Get detector position (This one is weird for some reason)
    det_pos = np.array(instrument.detector_position)
    if det_plane.fov_pixels.value % 2 == 0:
        det_pos += 0.5  # even arrays must be at a half pixel
    SciCens = np.array(det_pos)

    # Generate Class
    return SiafDistortion(
        aper.Sci2IdlDeg,
        Sci2Idl,
        Idl2Sci,
        SciRefs,
        SciCens,
        SciScales,
        pixelscale,
        det_plane.oversample,
    )
