import jax.numpy as np

import poppy
import scipy
from dLux.optical_layers import OpticalLayer
import dLux as dl
from jax import Array

import scipy.special
from dLuxWebbpsf.utils import j1, get_pixel_positions, generate_jwst_hexike_basis

__all__ = [
    "JWSTPrimary",
    "JWSTAberratedPrimary",
    "NircamCirc",
    "CoronOcculter",
    "NIRCamFieldAndWavelengthDependentAberration",
]


class JWSTPrimary(dl.Optic):
    """
    A class used to represent the JWST primary mirror. This is essentially a
    wrapper around the dLux.Optic class that simply enforces the normalisation
    at this plane, inverts the y axis automatically, and is slightly more
    efficient than the native implementation.
    """

    def __init__(
        self: OpticalLayer,
        transmission: Array = None,
        opd: Array = None,
    ):
        """
        Parameters
        ----------
        transmission: Array = None
            The Array of transmission values to be applied to the input
            wavefront.
        opd : Array, metres = None
            The Array of OPD values to be applied to the input wavefront.
        """
        super().__init__(transmission=transmission, opd=opd, normalise=True)

    def __call__(self, wavefront):
        # Apply transmission and normalise
        amplitude = wavefront.amplitude * self.transmission
        amplitude /= np.linalg.norm(amplitude)

        # Apply phase
        phase = wavefront.phase + wavefront.wavenumber * self.opd

        # Update and return
        return wavefront.set(["amplitude", "phase"], [amplitude, phase])


class JWSTAberratedPrimary(JWSTPrimary, dl.optical_layers.BasisLayer):
    """
    Child class of JWSTPrimary which adds the functionality to store a Hexike basis and coefficients.
    """

    def __init__(
        self,
        transmission: Array,
        opd: Array,
        radial_orders: Array | list = None,
        npix: int = 1024,
        noll_indices: Array | list = None,
        coefficients: Array | list = None,
        AMI: bool = False,
        mask: bool = False,
    ):
        """
        Parameters
        ----------
        transmission: Array
            The Array of transmission values to be applied to the input
            wavefront.
        opd : Array
            The Array of OPD values to be applied to the input wavefront.
        radial_orders : Array
            The radial orders of the zernike polynomials to be used for the
            aberrations. Input of [0, 1] would give [Piston, Tilt X, Tilt Y],
            [1, 2] would be [Tilt X, Tilt Y, Defocus, Astig X, Astig Y], etc.
            The order must be increasing but does not have to be consecutive.
            If you want to specify specific zernikes across radial orders the
            noll_indices argument should be used instead.
        noll_indices : Array
            The zernike noll indices to be used for the aberrations. [1, 2, 3]
            would give [Piston, Tilt X, Tilt Y], [2, 3, 4] would be [Tilt X,
            Tilt Y, Defocus].
        coefficients : Array
            The coefficients to be applied to the Hexike basis vectors.
        AMI : bool
            Whether to use the AMI segments or not.
        mask : bool
            Whether to apodise the basis with the AMI mask or not. Recommended is False.
        """

        super().__init__(transmission=transmission, opd=opd)
        self.basis = generate_jwst_hexike_basis(
            radial_orders=radial_orders,
            noll_indices=noll_indices,
            npix=npix,
            AMI=AMI,
            mask=mask,
        )
        self.coefficients = coefficients

    @property
    def basis_opd(self):
        """
        Returns the OPD calculated from the basis and coefficients.
        """
        return self.calculate(self.basis, self.coefficients)

    def __call__(self, wavefront):
        # Apply transmission and normalise
        amplitude = wavefront.amplitude * self.transmission
        amplitude /= np.linalg.norm(amplitude)

        total_opd = self.opd + self.basis_opd

        # Apply phase
        phase = wavefront.phase + wavefront.wavenumber * total_opd

        # Update and return
        return wavefront.set(["amplitude", "phase"], [amplitude, phase])


class CoronOcculter(OpticalLayer):
    """
    Layer for efficient propagation through a coronagraphic occulter.

    The order of operations is as follows:
        1. Pad the input wavefront by the factor 'pad'
        2. FFT's the wavefront
        3. Applies the occulting layer
        4. IFFT's the wavefront
        5. Crops the wavefront back to the original size
    """

    pad: int
    occulter: OpticalLayer

    def __init__(self, occulter, pad):
        """ "
        Constructor for the CoronOcculter class.
        """
        super().__init__()
        self.pad = int(pad)

        if not isinstance(occulter, OpticalLayer):
            raise TypeError("The occulter must be an instance of a dLux OpticalLayer.")

        self.occulter = occulter

    def __call__(self, wavefront):
        """
        Applies the coronagraphic occulter to the input wavefront.
        """
        npixels_in = wavefront.npixels
        wavefront = wavefront.FFT(pad=self.pad)
        wavefront = self.occulter(wavefront)
        wavefront = wavefront.IFFT(pad=1)
        return wavefront.crop_to(npixels_in)

    def __getattr__(self, name):
        if hasattr(self.occulter, name):
            return getattr(self.occulter, name)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no " f"attribute '{name}'"
            )


class NircamCirc(OpticalLayer):
    sigma: float
    diam: None
    npix: None
    oversample: None

    def __init__(self, sigma, diam, npix, oversample):
        self.sigma = float(sigma)
        self.diam = float(diam)
        self.npix = int(npix)
        self.oversample = int(oversample)

        super().__init__()

    def __call__(self, wavefront):
        # jax.debug.print("wavelength: {}", vars(wavefront))

        amplitude = self.get_transmission(wavefront.wavelength, wavefront.pixel_scale)
        return wavefront * amplitude

    def get_transmission(self, wavelength, pixelscale):
        # s = 2035
        # jax.debug.print("wavelength start: {}", wavelength)
        # jax.debug.print("pixelscale rad: {}", pixelscale)

        pixelscale = pixelscale * 648000.0 / np.pi  # from rad to arcsec

        # pixelscale = wavelength * 648000.0 / np.pi
        # pixelscale = pixelscale / self.diam / self.oversample

        # jax.debug.print("pixelscale arcsec: {}", pixelscale)

        npix = self.npix * self.oversample

        x, y = get_pixel_positions((npix, npix), (pixelscale, pixelscale))

        # jax.debug.print("x: {}", x[s:-s,s:-s][0].tolist())

        r = np.sqrt(x**2 + y**2)
        # jax.debug.print("r: {}", r[s:-s,s:-s][0].tolist())

        sigmar = self.sigma * r

        # clip sigma: The minimum is to avoid divide by zero
        #             the maximum truncates after the first sidelobe to match the hardware
        bessel_j1_zero2 = scipy.special.jn_zeros(1, 2)[1]

        sigmar = sigmar.clip(
            np.finfo(sigmar.dtype).tiny, bessel_j1_zero2
        )  # avoid divide by zero -> NaNs

        transmission = 1 - (2 * j1(sigmar) / sigmar) ** 2

        # jax.debug.print("transmission: {}", transmission[s:-s,s:-s][0].tolist())
        # jax.debug.print("transmission: {}", transmission[s:-s,s:-s].tolist())

        transmission = np.where(r == 0, 0, transmission)

        # the others have two, one in each corner, both halfway out of the 10x10 box.
        transmission = np.where(
            ((y < -5) & (y > -10)) & (np.abs(x) > 7.5) & (np.abs(x) < 12.5),
            np.sqrt(1e-3),
            transmission,
        )
        transmission = np.where(
            ((y < 10) & (y > 8)) & (np.abs(x) > 9) & (np.abs(x) < 11),
            np.sqrt(1e-3),
            transmission,
        )

        # mask holder edge
        transmission = np.where(y > 10, 0.0, transmission)

        # edge of mask itself
        # TODO the mask edge is complex and partially opaque based on CV3 images?
        # edge of glass plate rather than opaque mask I believe. To do later.
        # The following is just a temporary placeholder with no quantitative accuracy.
        # but this is outside the coronagraph FOV so that's fine - this only would matter in
        # modeling atypical/nonstandard calibration exposures.
        transmission = np.where((y < -11.5) & (y > -13), 0.7, transmission)

        return transmission


class NIRCamFieldAndWavelengthDependentAberration(OpticalLayer):
    opd: None
    amplitude: None

    zernike_coeffs: None
    defocus_zern: None
    tilt_zern: None

    focusmodel: None
    deltafocus: None
    opd_ref_focus: None

    ctilt_model: None
    tilt_offset: None
    tilt_ref_offset: None

    def __init__(self, instrument, amplitude, opd, zernike_coeffs):
        super().__init__()

        self.amplitude = np.asarray(amplitude, dtype=float)
        self.opd = np.asarray(opd, dtype=float)
        self.zernike_coeffs = np.asarray(zernike_coeffs, dtype=float)

        # Check for coronagraphy
        pupil_mask = instrument._pupil_mask
        is_nrc_coron = (pupil_mask is not None) and (
            ("LYOT" in pupil_mask.upper()) or ("MASK" in pupil_mask.upper())
        )

        # Polynomial equations fit to defocus model. Wavelength-dependent focus
        # results should correspond to Zernike coefficients in meters.
        # Fits were performed to the SW and LW optical design focus model
        # as provided by Randal Telfer.
        # See plot at https://github.com/spacetelescope/webbpsf/issues/179
        # The relative wavelength dependence of these focus models are very
        # similar for coronagraphic mode in the Zemax optical prescription,
        # so we opt to use the same focus model in both imaging and coronagraphy.
        defocus_to_rmswfe = -1.09746e7  # convert from mm defocus to meters (WFE)
        sw_focus_cf = (
            np.array(
                [
                    -5.169185169,
                    50.62919436,
                    -201.5444129,
                    415.9031962,
                    -465.9818413,
                    265.843112,
                    -59.64330811,
                ]
            )
            / defocus_to_rmswfe
        )
        lw_focus_cf = (
            np.array([0.175718713, -1.100964635, 0.986462016, 1.641692934])
            / defocus_to_rmswfe
        )

        # Coronagraphic tilt (`ctilt`) offset model
        # Primarily effects the LW channel (approximately a 0.031mm diff from 3.5um to 5.0um).
        # SW module is small compared to LW, but we include it for completeness.
        # Values have been determined using the Zernike offsets as reported in the
        # NIRCam Zemax models. The center reference positions will correspond to the
        # NIRCam target acquisition filters (3.35um for LW and 2.1um for SW)
        sw_ctilt_cf = np.array([125.849834, -289.018704]) / 1e9
        lw_ctilt_cf = (
            np.array([146.827501, -2000.965222, 8385.546158, -11101.658322]) / 1e9
        )

        # Get the representation of focus in the same Zernike basis as used for
        # making the OPD. While it looks like this does more work here than needed
        # by making a whole basis set, in fact because of caching behind the scenes
        # this is actually quick
        basis = poppy.zernike.zernike_basis_faster(
            nterms=len(self.zernike_coeffs), npix=self.opd.shape[0], outside=0
        )
        self.defocus_zern = np.asarray(basis[3])
        self.tilt_zern = np.asarray(basis[2])

        # Which wavelength was used to generate the OPD map we have already
        # created from zernikes?

        # Apply wavelength-dependent tilt offset for coronagraphy
        # We want the reference wavelength to be that of the target acq filter
        # Final offset will position TA ref wave at the OPD ref wave location
        #   (wave_um - opd_ref_wave) - (ta_ref_wave - opd_ref_wave) = wave_um - ta_ref_wave
        if instrument.channel.upper() == "SHORT":
            self.focusmodel = sw_focus_cf
            opd_ref_wave = 2.12
        else:
            self.focusmodel = lw_focus_cf
            opd_ref_wave = 3.23

            if is_nrc_coron:
                self.opd_ref_focus = np.polyval(self.focusmodel, opd_ref_wave)
            else:
                self.opd_ref_focus = 1.206e-7  # Not coronagraphy (e.g., imaging)

        # If F323N or F212N, then no focus offset necessary
        if ("F323N" in instrument.filter) or ("F212N" in instrument.filter):
            self.deltafocus = lambda wl: 0
        else:
            self.deltafocus = (
                lambda wl: np.polyval(self.focusmodel, wl) - self.opd_ref_focus
            )

        if is_nrc_coron:
            if instrument.channel.upper() == "SHORT":
                self.ctilt_model = sw_ctilt_cf
                ta_ref_wave = 2.10
            else:
                self.ctilt_model = lw_ctilt_cf
                ta_ref_wave = 3.35

            self.tilt_ref_offset = np.polyval(self.ctilt_model, ta_ref_wave)

            # print("opd_ref_focus: {}", self.opd_ref_focus)
            # print("tilt_ref_offset: {}", self.tilt_ref_offset)

            self.tilt_offset = (
                lambda wl: np.polyval(self.ctilt_model, wl) - self.tilt_ref_offset
            )
        else:
            self.tilt_ref_offset = None
            self.ctilt_model = None
            self.tilt_offset = lambda wl: 0

    def __call__(self, wavefront):
        wavelength = wavefront.wavelength * 1e6

        # jax.debug.print("wavelength: {}", wavelength)

        mod_opd = self.opd - self.deltafocus(wavelength) * self.defocus_zern
        mod_opd = mod_opd + self.tilt_offset(wavelength) * self.tilt_zern

        wavefront = wavefront * self.amplitude

        return wavefront.add_opd(mod_opd)
