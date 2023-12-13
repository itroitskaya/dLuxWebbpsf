import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
from jax import Array
from jax.scipy.ndimage import map_coordinates
import webbpsf
from astropy.io import fits
import os
from astropy.convolution.kernels import CustomKernel

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
    "get_detector_ipc_model",
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

    def __init__(self, instrument, pixelscale):
        if instrument.name == "NIRCam":
            instrument_key = self._check_nircam_filter(instrument)
        else:
            instrument_key = instrument.name

        charge_def_params = (
            webbpsf.constants.INSTRUMENT_DETECTOR_CHARGE_DIFFUSION_DEFAULT_PARAMETERS
        )

        if instrument_key not in charge_def_params.keys():
            raise ValueError(
                f"instrument_key {instrument_key} not in {charge_def_params.keys()}"
            )

        """Convert sigma to units of pixels. NOTE: this assumes sigma has
        the same units as the pixel scale."""
        self.sigma = float(charge_def_params[instrument_key]) / pixelscale
        self.kernel_size = int(2*round(4*self.sigma) + 1)

        super().__init__()

    def apply(self, PSF):
        diffused = utils.gaussian_filter_correlate(PSF.data, self.sigma, ksize=self.kernel_size)
        return PSF.set("data", diffused)

    @staticmethod
    def _check_nircam_filter(instrument):
        """
        Checking to see if the filter is in the short or long wavelength. We do this
        because the accepted NIRCam Instrument keys are "NIRCAM_SW" and "NIRCAM_LW".

        Parameters
        ----------
        instrument : webbpsf instrument object

        Returns
        -------
        str
            Either "NIRCAM_SW" or "NIRCAM_LW"
        """

        if instrument.name == "NIRCam":
            sw_channel = [
                "F070W",
                "F090W",
                "F115W",
                "F140M",
                "F150W",
                "F162M",
                "F164N",
                "F150W2",
                "F182M",
                "F187N",
                "F200W",
                "F210M",
                "F212N",
            ]
            lw_channel = [
                "F250M",
                "F277W",
                "F300M",
                "F322W2",
                "F323N",
                "F335M",
                "F356W",
                "F360M",
                "F405N",
                "F410M",
                "F430M",
                "F444W",
                "F460M",
                "F466N",
                "F470N",
                "F480M",
            ]
            if instrument.filter in sw_channel:
                return "NIRCAM_SW"
            elif instrument.filter in lw_channel:
                return "NIRCAM_LW"
            else:
                raise NotImplementedError(
                    f"Filter {instrument.filter} not currently supported."
                )


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


def get_detector_ipc_model(instrument):
    """Retrieve detector interpixel capacitance model.
    The details of the available calibration data vary per instrument.
    Altered version of webbpsf function at
    https://github.com/spacetelescope/webbpsf/blob/ca1fdf6a151434ee93a59729436c903858f527b9/webbpsf/detectors.py#L15

    Parameters:
    -----------
    instrument : webbpsf instrument object

    Returns:
    --------
    kernel : numpy.ndarray
        Convolution kernel
    """

    inst_name = instrument.name  # TODO probably will break for NIRCam
    det = instrument._detector  # detector name

    if inst_name == "NIRCam":
        det2sca = {
            "NRCA1": "481",
            "NRCA2": "482",
            "NRCA3": "483",
            "NRCA4": "484",
            "NRCA5": "485",
            "NRCB1": "486",
            "NRCB2": "487",
            "NRCB3": "488",
            "NRCB4": "489",
            "NRCB5": "490",
        }

        # IPC effect
        # read the SCA extension for the detector
        sca_path = os.path.join(
            webbpsf.utils.get_webbpsf_data_path(),
            "NIRCam",
            "IPC",
            "KERNEL_IPC_CUBE.fits",
        )
        kernel_ipc = CustomKernel(
            fits.open(sca_path)[det2sca[det]].data[0]
        )  # we read the first slice in the cube

        # Note: WebbPSF generates a kernel for PPC effect as well
        # This is ignored here
        sca_path_ppc = os.path.join(
            webbpsf.utils.get_webbpsf_data_path(),
            "NIRCam",
            "IPC",
            "KERNEL_PPC_CUBE.fits",
        )
        kernel_ppc = CustomKernel(
            fits.open(sca_path_ppc)[det2sca[det]].data[0]
        )  # we read the first slice in the cube

        kernel = (
            kernel_ipc.array,
            kernel_ppc.array,
        )  # Return two distinct convolution kernels in this case

        # kernel = kernel_ipc.array

    elif inst_name == "NIRISS":
        # NIRISS IPC files distinguish between the 4 detector readout channels, and
        # whether or not the pixel is within the region of a large detector epoxy void
        # that is present in the NIRISS detector.

        # this set-up the input variables as required by Kevin Volk IPC code
        # image = psf_hdulist[ext].data
        xposition = instrument._detector_position[0]
        yposition = instrument._detector_position[1]

        # find the voidmask fits file
        voidmask10 = os.path.join(
            webbpsf.utils.get_webbpsf_data_path(), "NIRISS", "IPC", "voidmask10.fits"
        )

        if os.path.exists(voidmask10):
            maskimage = fits.getdata(voidmask10)
        else:
            maskimage = None

        nchannel = int(yposition) // 512
        try:
            flag = maskimage[nchannel, int(xposition)]
        except:
            # This marks the pixel as non-void by default if the maskimage is not
            # read in properly
            flag = 0
        frag1 = ["A", "B", "C", "D"]
        frag2 = ["notvoid", "void"]

        ipcname = "ipc5by5median_amp" + frag1[nchannel] + "_" + frag2[flag] + ".fits"
        ipc_file = os.path.join(
            webbpsf.utils.get_webbpsf_data_path(), "NIRISS", "IPC", ipcname
        )
        if os.path.exists(ipc_file):
            kernel = fits.getdata(ipc_file)
        else:
            kernel = None

    elif inst_name in ["FGS", "NIRSPEC", "WFI", "MIRI"]:
        kernel = None  # No IPC models yet implemented for these

    return kernel


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
