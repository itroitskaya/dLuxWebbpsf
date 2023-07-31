from __future__ import annotations
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
import dLuxWebbpsf
import webbpsf


__all__ = ["NIRISS", "NIRCam"]


class JWST(dl.Instrument):
    """
    Class for the James Webb Space Telescope.
    """

    @staticmethod
    def _configure_instrument(
        instrument,
        filter=None,
        pupil_mask=None,
        image_mask=None,
        detector=None,
        aperture=None,
        options=None,
        fov_arcsec=None,
        fov_pixels=None,
        fft_oversample=None,
        detector_oversample=None,
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
            webb_osys.set_position_from_aperture_name(aperture)
        if options is not None:
            webb_osys.options.update(options)

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

        # Get the optical system
        optics = webb_osys.get_optical_system(**kwargs)
        return webb_osys, optics

    def _construct_detector(self, instrument, optics):
        """Constructs a detector object for the instrument."""

        # Jitter - webbpsf default units are arcseconds, so we assume that psf
        # units are also arcseconds
        options = instrument.options
        if "jitter" in options.keys() and options["jitter"] == "gaussian":
            layers = [
                dLuxWebbpsf.ApplyJitter(
                    sigma=instrument.options["jitter_sigma"]
                )
            ]

        # Rotation - probably want to eventually change units to degrees
        angle = instrument._detector_geom_info.aperture.V3IdlYAngle
        layers.append(dLuxWebbpsf.Rotate(angle=dlu.deg_to_rad(angle)))

        # Distortion
        if "add_distortion" in options.keys() and options["add_distortion"]:
            siaf = instrument._detector_geom_info.aperture["aperture"]
            layers.append(dLuxWebbpsf.DistortionFromSiaf(siaf))

        # Downsample - assumes last plane is detector
        layers.append(dl.IntegerDownsample(optics.planes[-1].oversample))

        # Construct detector
        return dl.LayeredDetector(layers)

    def _construct_source(self, instrument, nlambda):
        """Constructs a source object for the instrument."""
        wavelengths, weights = instrument._get_weights(
            source=None,
            nlambda=nlambda,
        )
        return dl.PointSource(wavelengths=wavelengths, weights=weights)


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
    ):
        # Get configured instrument and optical system
        instrument, osys = self._configure_instrument(
            "NIRISS",
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

        """Construct Optics"""
        optics = self._construct_optics(
            osys.planes, image_mask, pupil_mask, clean
        )

        """Construct source object"""
        source = (self._construct_source(instrument, nlambda), "source")

        """Construct Detector Object"""
        detector = self._construct_detector(instrument, osys)

        """Construct the instrument"""
        super().__init__(optics=optics, sources=source, detector=detector)

    def _construct_optics(self, planes, image_mask, pupil_mask, clean):
        """Constructs an optics object for the instrument."""
        # Primary mirror - note this class automatically flips about the y-axis
        layers = [
            (
                dLuxWebbpsf.optical_layers.JWSTPrimary(
                    planes[0].amplitude, planes[0].opd
                ),
                "pupil",
            )
        ]

        # If 'clean', then we don't want to apply pre-calc'd aberrations
        if not clean:
            # NOTE: We keep amplitude here because the circumscribed circle clips
            # the edge of the primary mirrors, see:
            # https://github.com/spacetelescope/webbpsf/issues/667, Long term we
            # are unlikely to want this.
            FDA = dl.Optic(planes[2].amplitude, planes[2].opd)
            layers.append((FDA, "FieldDependentAberration"))

        # No image mask layers yet for NIRISS
        if image_mask is not None:
            raise NotImplementedError(
                "Coronagraphic masks not yet implemented for NIRISS."
            )

        # Index this from the end of the array since it will always be the second
        # last plane in the osys. Note this assumes there is no OPD in that optic.
        if pupil_mask is not None:
            layers.append((dl.Optic(planes[-2].amplitude), "PupilMask"))

        # Now the Fourier transform to the detector
        # For now, we keep this in radians, but TODO either through dLux or
        # this package we will want to use a layer with arcsec units.
        osamp = planes[-1].oversample
        det_npix = (planes[-1].fov_pixels * osamp).value
        pscale = (planes[-1].pixelscale).to("arcsec/pix").value
        layers.append(dLuxWebbpsf.MFT(det_npix, pscale, oversample=osamp))

        # Finally, construct the actual Optics object
        return dl.LayeredOptics(
            wf_npixels=planes[0].npix,
            diameter=planes[0].pixelscale.to("m/pix").value * planes[0].npix,
            layers=layers,
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
        wavefront_downsample=1,  # int: downsampling factor for the base wavefront
    ):
        # Get configured instrument and optical system
        instrument, osys = self._configure_instrument(
            "NIRCam",
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

        """Construct Optics"""
        optics = self._construct_optics(
            osys.planes,
            image_mask,
            pupil_mask,
            clean,
            instrument,
            wavefront_downsample,
        )

        """Construct source object"""
        source = self._construct_source(instrument, nlambda)

        """Construct Detector Object"""
        detector = self._construct_detector(instrument, optics)

        """Construct the instrument"""
        super().__init__(optics=optics, sources=source, detector=detector)

    def _construct_optics(
        self,
        planes,
        image_mask,
        pupil_mask,
        clean,
        instrument,
        wavefront_downsample=1,
    ):
        """Constructs an optics object for the instrument."""
        npixels = planes[0].npix
        diameter = planes[0].pixelscale.to("m/pix").value * planes[0].npix

        # Primary mirror - note this class automatically flips about the y-axis
        layers = [
            (
                dLuxWebbpsf.optical_layers.JWSTPrimary(
                    planes[0].amplitude, planes[0].opd
                ),
                "pupil",
            )
        ]

        # Coronagraphic masks
        if image_mask is not None:
            occulter = dLuxWebbpsf.NircamCirc(
                planes[2].sigma, diameter, npixels, planes[2].oversample
            )
            layers.append(
                (
                    dLuxWebbpsf.CoronOcculter(occulter, planes[2].oversample),
                    "ImageMask",
                )
            )

        # Pupil mask, note this assumes there is no OPD in that optic. Index
        # from the end of the array so we dont have to check for image masks.
        if pupil_mask is not None:
            layers.append((dl.Optic(planes[-3].amplitude), "PupilMask"))

        # If 'clean', then we don't want to apply pre-calc'd aberrations
        if not clean:
            # NOTE: We keep amplitude here because the circumscribed circle clips
            # the edge of the primary mirrors, see:
            # https://github.com/spacetelescope/webbpsf/issues/667, Long term we
            # are unlikely to want this.

            # Not sure what this downsample is doing...
            aberration_mask = planes[-2].amplitude
            aberration_opd = planes[-2].opd

            if wavefront_downsample > 1:
                aberration_mask = dlu.downsample(
                    aberration_mask, wavefront_downsample
                )
                aberration_opd = dlu.downsample(
                    aberration_opd, wavefront_downsample
                )

            aberration_zernikes = planes[-2].zernike_coeffs

            FDA = dLuxWebbpsf.NIRCamFieldAndWavelengthDependentAberration(
                instrument,
                # aberration_mask,
                aberration_opd,
                aberration_zernikes,
            )
            layers.append((FDA, "NIRCamFieldAndWavelengthDependentAberration"))

        # Now the Fourier transform to the detector
        # For now, we keep this in radians, but TODO either through dLux or
        # this package we will want to use a layer with arcsec units.
        osamp = planes[-1].oversample
        det_npix = (planes[-1].fov_pixels * osamp).value
        pscale = (planes[-1].pixelscale).to("arcsec/pix").value
        layers.append(dLuxWebbpsf.MFT(det_npix, pscale, oversample=osamp))

        # Finally, construct the actual Optics object
        return dl.LayeredOptics(
            wf_npixels=planes[0].npix,
            diameter=planes[0].pixelscale.to("m/pix").value * planes[0].npix,
            layers=layers,
        )
