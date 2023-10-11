from __future__ import annotations
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
import webbpsf

from .detector_layers import *
from .optical_layers import *
from .basis import generate_jwst_basis

__all__ = ["NIRISS", "NIRCam"]


class JWST(dl.Telescope):
    """
    Class for the James Webb Space Telescope.
    """

    filter_wavelengths: None
    filter_weights: None
    # pixelscale: None

    def __init__(
        self: JWST,
        instrument: str,  # string: name of the instrument
        filter=None,  # string: name of the filter
        pupil_mask=None,  # string: name of the pupil mask
        image_mask=None,  # string: name of the image mask
        detector=None,  # string: name of the detector
        aperture=None,  # string: name of the aperture
        fft_oversample=4,  # int: oversampling factor for the FFT
        detector_oversample=None,  # int: oversampling factor for the detector
        fov_arcsec=5,  # float: field of view in arcseconds
        fov_pixels=None,  # int: number of pixels in the field of view
        options=None,  # dict: additional options for the webbpsf.OpticalSystem
        nlambda=None,  # int: number of wavelengths
        clean=False,  # bool: When true, only the measured primary mirror aberrations are applied.
        wavefront_downsample=1,  # int: downsampling factor for the wavefront
        monochromatic=None,  # float: wavelength in microns
        offset=None,  # tuple: offset in radians
        phase_retrieval_terms=0,  # Number of terms in phase retrieval layer
        flux=1,  # float: flux of point source
        source=None,  # Source: dLux source object
    ):
        # Get configured instrument and optical system
        instrument, osys = self._configure_instrument(
            instrument,
            filter=filter,
            pupil_mask=pupil_mask,
            image_mask=image_mask,
            detector=detector,
            aperture=aperture,
            options=options,
            fov_arcsec=fov_arcsec,
            fov_pixels=fov_pixels,
            fft_oversample=fft_oversample,
            detector_oversample=detector_oversample,
        )

        # Construct Optics
        optics = self._construct_optics(
            osys.planes,
            image_mask,
            pupil_mask,
            clean,
            instrument,
            wavefront_downsample,
            phase_retrieval_terms,
        )

        # Construct source object
        source = self._construct_source(
            instrument=instrument,
            source=source,
            nlambda=nlambda,
            monochromatic=monochromatic,
            offset=offset,
            flux=flux,
        )

        # Construct Detector Object
        detector = self._construct_detector(instrument, osys)

        # Construct the instrument
        super().__init__(optics=optics, source=source, detector=detector)

    @staticmethod
    def _configure_instrument(
        instrument,
        *,
        filter,
        pupil_mask,
        image_mask,
        detector,
        aperture,
        options,
        fov_arcsec,
        fov_pixels,
        fft_oversample,
        detector_oversample,
    ):
        """
        Configure a WebbPSF instrument for use with dLuxWebbpsf.
        """
        # Check Valid Instrument
        if instrument not in ("NIRCam", "NIRISS"):
            raise NotImplementedError(
                f"Instrument {instrument} not currently supported."
            )

        # Get WebbPSF instrument
        webb_osys = getattr(webbpsf, instrument)()

        # TODO: Instrument specific checks for valid inputs

        # Configure the instrument
        if filter is not None:
            webb_osys.filter = filter
        if pupil_mask is not None:
            webb_osys.pupil_mask = pupil_mask
        if image_mask is not None:
            webb_osys.image_mask = image_mask
        if detector is not None:
            webb_osys.detector = detector
        if aperture is not None:
            webb_osys.aperture_name = aperture
        if options is not None:
            webb_osys.options.update(options)

        webb_osys.set_position_from_aperture_name(aperture)

        # Configure optics input arguments
        kwargs = {}
        if fov_arcsec is not None:
            kwargs["fov_arcsec"] = fov_arcsec
        if fov_pixels is not None:
            kwargs["fov_pixels"] = fov_pixels
        if fft_oversample is not None:
            kwargs["fft_oversample"] = fft_oversample
        if detector_oversample is not None:
            kwargs["detector_oversample"] = detector_oversample
        if options is not None:
            kwargs["options"] = options

        # Get the optical system - Note we calculate the PSF here because
        # otherwise some optical planes do not have its attributes populated,
        # namely the CLEARP mask transmission.
        # SIC! Check that these planes are not wavelength-dependent. There should be no need to call calc_psf!
        optics = webb_osys.get_optical_system(**kwargs)
        # optics.calc_psf()
        return webb_osys, optics

    def _construct_detector(self, instrument, optics):
        """Constructs a detector object for the instrument."""

        options = instrument.options
        layers = []

        # Jitter - webbpsf default units are arcseconds, so we assume that psf
        # units are also arcseconds
        if "jitter" in options.keys() and options["jitter"] == "gaussian":
            layers.append(
                ("jitter", ApplyJitter(sigma=instrument.options["jitter_sigma"]))
            )

        # Rotation - probably want to eventually change units to degrees
        # angle = instrument._detector_geom_info.aperture.V3IdlYAngle
        # layers.append(Rotate(angle=dlu.deg2rad(-angle)))

        # Distortion
        if "add_distortion" not in options.keys() or options["add_distortion"]:
            layers.append(DistortionFromSiaf(instrument, optics))

        # # Charge migration (via \jitter) units of arcseconds
        # c = webbpsf.constants
        # sigma = c.INSTRUMENT_DETECTOR_CHARGE_DIFFUSION_DEFAULT_PARAMETERS[
        #     instrument.name
        # ]
        # layers.append(
        #     (ApplyJitter(sigma=sigma), "charge_migration")
        # )

        # Downsample - assumes last plane is detector
        layers.append(dl.Downsample(optics.planes[-1].oversample))

        # Construct detector
        return dl.LayeredDetector(layers)

    def _construct_source(
        self, *, instrument, source, nlambda, monochromatic, offset, flux
    ):
        """Constructs a source object for the instrument."""

        if nlambda is None or nlambda == 0:
            nlambda = instrument._get_default_nlambda(instrument.filter)

        wavelengths, weights = instrument._get_weights(
            source=None, nlambda=nlambda, monochromatic=monochromatic
        )

        self.filter_wavelengths = wavelengths
        self.filter_weights = weights

        if source is None:
            # Generate point source if no source is provided
            position = [0, 0]
            if offset is not None:
                position = offset
            source = dl.PointSource(wavelengths=wavelengths, weights=weights, position=position, flux=flux)

        return source


class NIRISS(JWST):
    """
    Class for the NIRISS instrument on the James Webb Space Telescope.
    """

    def __init__(
        self: NIRISS,
        filter=None,  # string: name of the filter
        pupil_mask=None,  # string: name of the pupil mask
        image_mask=None,  # string: name of the image mask
        detector=None,  # string: name of the detector
        aperture=None,  # string: name of the aperture
        fft_oversample=4,  # int: oversampling factor for the FFT
        detector_oversample=None,  # int: oversampling factor for the detector
        fov_arcsec=5,  # float: field of view in arcseconds
        fov_pixels=None,  # int: number of pixels in the field of view
        options=None,  # dict: additional options for the webbpsf.OpticalSystem
        nlambda=None,  # int: number of wavelengths
        clean=False,  # bool: When true, only the measured primary mirror aberrations are applied.
        wavefront_downsample=1,  # int: downsampling factor for the wavefront
        monochromatic=None,  # float: wavelength in microns
        offset=None,  # tuple: offset in radians
        phase_retrieval_terms=0,  # Number of terms in phase retrieval layer
        flux=1,  # float: flux of point source
        source=None,  # Source: dLux source object
    ):
        super().__init__(
            "NIRISS",
            filter=filter,
            pupil_mask=pupil_mask,
            image_mask=image_mask,
            detector=detector,
            aperture=aperture,
            fft_oversample=fft_oversample,
            detector_oversample=detector_oversample,
            fov_arcsec=fov_arcsec,
            fov_pixels=fov_pixels,
            options=options,
            nlambda=nlambda,
            clean=clean,
            wavefront_downsample=wavefront_downsample,
        )

    def _construct_optics(
        self,
        planes,
        image_mask,
        pupil_mask,
        clean,
        instrument,
        wavefront_downsample,
        phase_retrieval_terms,
    ):
        """Constructs an optics object for the instrument."""

        # Construct downsampler
        def dsamp(x): return dlu.downsample(x, wavefront_downsample)

        npix = 1024 // wavefront_downsample

        # Primary mirror
        layers = [
            ("pupil", JWSTPrimary(dsamp(planes[0].amplitude), dsamp(planes[0].opd))),
            ("InvertY", dl.Flip(0)),
        ]

        # If 'clean', then we don't want to apply pre-calc'd aberrations
        if not clean:
            # NOTE: We keep amplitude here because the circumscribed circle clips
            # the edge of the primary mirrors, see:
            # https://github.com/spacetelescope/webbpsf/issues/667, Long term we
            # are unlikely to want this.
            FDA = dl.Optic(dsamp(planes[2].amplitude), dsamp(planes[2].opd))
            layers.append(("aberrations", FDA))

        # No image mask layers yet for NIRISS
        if image_mask is not None:
            raise NotImplementedError("Coronagraphic masks not yet implemented for NIRISS.")

        # Index this from the end of the array since it will always be the second
        # last plane in the osys. Note this assumes there is no OPD in that optic.
        # We need this logic here since MASK_NRM plane has an amplitude, where
        # as the CLEARP plane has a transmission... for some reason.
        pupil_plane = planes[-2]
        if instrument.pupil_mask == "CLEARP":
            layer = dl.Optic(dsamp(pupil_plane.transmission))
        elif instrument.pupil_mask == "MASK_NRM":
            layer = dl.Optic(dsamp(pupil_plane.amplitude))
        else:
            raise NotImplementedError("Only CLEARP and MASK_NRM are supported.")
        layers.append(("pupil_mask", layer))

        # Finally, construct the actual Optics object
        return dl.AngularOpticalSystem(
            wf_npixels=npix,
            diameter=planes[0].pixelscale.to("m/pix").value * planes[0].npix,
            layers=layers,
            psf_npixels=planes[-1].fov_pixels.value,
            psf_pixel_scale=(planes[-1].pixelscale).to("arcsec/pix").value,
            oversample=planes[-1].oversample,
        )


class NIRCam(JWST):
    """
    Class for the NIRCam instrument on the James Webb Space Telescope.
    """

    def __init__(
        self: NIRCam,
        filter=None,  # string: name of the filter
        pupil_mask=None,  # string: name of the pupil mask
        image_mask=None,  # string: name of the image mask
        detector=None,  # string: name of the detector
        aperture=None,  # string: name of the aperture
        fft_oversample=4,  # int: oversampling factor for the FFT
        detector_oversample=None,  # int: oversampling factor for the detector
        fov_arcsec=5,  # float: field of view in arcseconds
        fov_pixels=None,  # int: number of pixels in the field of view
        options=None,  # dict: additional options for the webbpsf.OpticalSystem
        nlambda=None,  # int: number of wavelengths
        clean=False,  # bool: When true, only the measured primary mirror aberrations are applied.
        wavefront_downsample=1,  # int: downsampling factor for the wavefront
        monochromatic=None,  # float: wavelength in microns
        offset=None,  # tuple: offset in radians
        phase_retrieval_terms=0,  # Number of terms in phase retrieval layer
        flux=1,  # float: flux of point source
        source=None,  # Source: dLux source object
    ):
        super().__init__(
            "NIRCam",
            filter=filter,
            pupil_mask=pupil_mask,
            image_mask=image_mask,
            detector=detector,
            aperture=aperture,
            fft_oversample=fft_oversample,
            detector_oversample=detector_oversample,
            fov_arcsec=fov_arcsec,
            fov_pixels=fov_pixels,
            options=options,
            nlambda=nlambda,
            clean=clean,
            wavefront_downsample=wavefront_downsample,
            monochromatic=monochromatic,
            offset=offset,
            phase_retrieval_terms=phase_retrieval_terms,
            flux=flux,
            source=source,
        )

    def _construct_optics(
        self,
        planes,
        image_mask,
        pupil_mask,
        clean,
        instrument,
        wavefront_downsample,
        phase_retrieval_terms,
    ):
        """Constructs an optics object for the instrument."""

        # Construct downsampler
        def dsamp(x): return dlu.downsample(x, wavefront_downsample)

        diameter = planes[0].pixelscale.to("m/pix").value * planes[0].npix
        npix = planes[0].npix // wavefront_downsample

        # Primary mirror
        layers = []

        pupil_transmission = dsamp(planes[0].amplitude)
        pupil_opd = dsamp(planes[0].opd)

        pscale = planes[0].pixelscale.value * wavefront_downsample

        if phase_retrieval_terms > 0:
            # Add phase retrieval layer

            basis = generate_jwst_basis(phase_retrieval_terms, npix, pscale)
            basis_flat = basis.reshape((phase_retrieval_terms * 18, npix, npix))
            coeffs = np.zeros((basis_flat.shape[0]))

            pupil_transmission = pupil_transmission

            pupil = JWSTSimplePrimary(pupil_transmission, pupil_opd, pscale, basis_flat, coeffs)

            layers.append(("pupil", pupil))
        else:
            pupil = JWSTSimplePrimary(pupil_transmission, pupil_opd, pscale)
            layers.append(("pupil", pupil))

        layers.extend([
                # Plane 1: Coordinate Inversion in y axis
                ("InvertY", dl.Flip(0))
        ])

        # Coronagraphic masks
        # TODO: This should explicity check for the correct mask (ie CIRCLYOT)
        if pupil_mask is not None and image_mask is not None:
            occulter = NircamCirc(planes[2].sigma, diameter, npix, planes[2].oversample)
            layers.append(("image_mask", CoronOcculter(occulter, planes[2].oversample)))
            # Pupil mask, note this assumes there is no OPD in that optic.
            layers.append(("pupil_mask", dl.Optic(dsamp(planes[3].amplitude))))

        # If 'clean', then we don't want to apply pre-calc'd aberrations
        if not clean:
            # NOTE: We keep amplitude here because the circumscribed circle clips
            # the edge of the primary mirrors, see:
            # https://github.com/spacetelescope/webbpsf/issues/667, Long term we
            # are unlikely to want this.

            FDA = NIRCamFieldAndWavelengthDependentAberration(
                instrument,
                dsamp(planes[-2].amplitude),
                dsamp(planes[-2].opd),
                planes[-2].zernike_coeffs,
            )
            layers.append(("aberrations", FDA))

        # Finally, construct the actual Optics object
        return dl.AngularOpticalSystem(
            wf_npixels=npix,
            diameter=diameter,
            layers=layers,
            psf_npixels=planes[-1].fov_pixels.value,
            psf_pixel_scale=(planes[-1].pixelscale).to("arcsec/pix").value,
            oversample=planes[-1].oversample,
        )
