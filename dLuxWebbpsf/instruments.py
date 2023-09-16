from __future__ import annotations
import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
import webbpsf

from dLuxWebbpsf.optical_layers import *
from dLuxWebbpsf.basis import jwst_hexike_bases

__all__ = ["NIRCam"]

class NIRCam(dl.instruments.Instrument):

    filter_wavelengths: None
    filter_weights: None
    pixelscale: None

    def __init__(self,
                 *,
                 filter=None, # string: name of the filter
                 pupil_mask=None, # string: name of the pupil mask
                 coron_mask=None, # string: name of the image mask
                 detector=None, # string: name of the detector
                 aperture=None, # string: name of the aperture
                 fft_oversample=4, # int: oversampling factor for the FFT
                 detector_oversample=4, # int: oversampling factor for the detector
                 detector_keep_size=True, # bool: scale back the detector size after oversampling
                 fov_arcsec=5, # float: field of view in arcseconds
                 fov_pixels=None, # int: number of pixels in the field of view
                 wavefront_downsample=1, # int: downsampling factor for the base wavefront
                 options=None, # dict: additional options for the webbpsf.OpticalSystem
                 nlambda=None, # int: number of wavelengths
                 monochromatic=None, # float: wavelength in microns
                 offset=None, # tuple: offset in arcseconds
                 phase_retrieval_terms=0, # Number of terms in phase retrieval layer
                 flux=1,
                 source=None, # Source: dLux source object
                 ):
        
        # Initialize the instrument
        webb_osys = webbpsf.NIRCam()
        webb_osys.filter = filter
        webb_osys.pupil_mask = pupil_mask
        webb_osys.image_mask = coron_mask
        webb_osys.detector = detector
        webb_osys.set_position_from_aperture_name(aperture)

        if options is not None:
            webb_osys.options.update(options)

        nircam = webb_osys.get_optical_system(fft_oversample=fft_oversample,
                            detector_oversample=detector_oversample,
                            fov_arcsec=fov_arcsec, fov_pixels=fov_pixels,
                            options=options)
        
        # Get the pupil and detector planes
        pupil_plane = nircam.planes[0]

        # Define wavefront 
        npix = 1024 // wavefront_downsample
        diameter = pupil_plane.pixelscale.to('m/pix').value * pupil_plane.npix

        pupil_mask = pupil_plane.amplitude
        pupil_opd = pupil_plane.opd

        if wavefront_downsample > 1:
            pupil_mask = dlu.downsample(pupil_mask, wavefront_downsample)
            pupil_opd = dlu.downsample(pupil_opd, wavefront_downsample)

        optical_layers = [
            #Plane 0: Pupil plane: JWST Entrance Pupil
            (dl.Optic(pupil_mask, pupil_opd), "Pupil")]
        
        if phase_retrieval_terms > 0:
            pscale = pupil_plane.pixelscale.value * wavefront_downsample
            hmask, basis = jwst_hexike_bases(phase_retrieval_terms, npix, pscale)
            basis_flat = basis.reshape((phase_retrieval_terms*18, npix, npix))
            coeffs = np.zeros((basis_flat.shape[0]))

            optical_layers.extend([
                (JWSTBasis(hmask, basis_flat, coeffs), "JWSTBasis")
            ])


        optical_layers.extend([
            #Plane 1: Coordinate Inversion in y axis
            (InvertY(), "InvertY")
        ])

        # Define coronagraphic planes

        # TODO: Assert that both are present
        if pupil_mask is not None and coron_mask is not None:
            optical_layers.extend([
                #Plane 2: Image plane: NIRCam mask
                #TODO: Pad and crop only when oversample > 1
                Pad(npix * nircam.planes[2].oversample),
                dl.FFT(focal_length=None, pad=1),
                (NircamCirc(nircam.planes[2].sigma, diameter, npix, nircam.planes[2].oversample), "NIRCamCirc"),
                dl.FFT(focal_length=None, pad=1, inverse=True),
                Crop(npix),

                #Plane 3: Pupil plane
                (dl.Optic(nircam.planes[3].amplitude), "PupilMask"),
            ])


        #Plane 4: Pupil plane: NIRCamLWA internal WFE at V2V3=(1.46,-6.75)', near Z430R_A

        aberration_layer = nircam.planes[-2]
        aberration_mask = aberration_layer.amplitude
        aberration_opd = aberration_layer.opd

        if wavefront_downsample > 1:
            aberration_mask = dlu.downsample(aberration_mask, wavefront_downsample)
            aberration_opd = dlu.downsample(aberration_opd, wavefront_downsample)

        aberration_zernikes = aberration_layer.zernike_coeffs

        optical_layers.extend([
            (NIRCamFieldAndWavelengthDependentAberration(webb_osys, aberration_mask, aberration_opd, aberration_zernikes), "NIRCamFieldAndWavelengthDependentAberration")
        ])


        #Plane 5: Detector plane: NIRCam detector

        detector_plane = nircam.planes[-1]
        self.pixelscale = detector_plane.pixelscale

        det_npix = (detector_plane.fov_pixels * detector_plane.oversample).value
        pscale = (detector_plane.pixelscale / detector_plane.oversample).to('radian/pix').value
        print(f'pixel scale: {pscale}')
        optical_layers.extend([
            dl.MFT(det_npix, pscale)
        ])

        optics = dl.LayeredOptics(wf_npixels=npix, diameter=diameter, layers=optical_layers)

        # Define detector
        detector_layers = []

        if detector_keep_size == True:
            detector_layers.append(dl.IntegerDownsample(detector_oversample))

        detector = dl.LayeredDetector(detector_layers)


        # Define source
        if nlambda is None or nlambda == 0:
            nlambda = webb_osys._get_default_nlambda(webb_osys.filter)

        wavelengths, weights = webb_osys._get_weights(source=None, nlambda=nlambda, monochromatic=monochromatic)

        self.filter_wavelengths = wavelengths
        self.filter_weights = weights

        if source is None:
            # Generate point source if no source is provided
            spectrum = dl.Spectrum(wavelengths, weights)
            position = [0, 0]
            if offset is not None:
                position = offset
            # wavelengths are still required even if they are discarded
            source = dl.PointSource(wavelengths=wavelengths, spectrum=spectrum, position=position, flux=flux)

        # Initialize the instrument
        super().__init__(optics=optics, sources=(source, 'source'), detector=detector)