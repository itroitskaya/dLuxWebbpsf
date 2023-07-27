from __future__ import annotations
import dLux as dl
import webbpsf

from dLuxWebbpsf.optical_layers import *
from dLuxWebbpsf.utils import rebin

__all__ = ["NIRCam"]

class NIRCam(dl.instruments.Instrument):

    filter_wavelengths: None
    filter_weights: None

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
        detector_plane = nircam.planes[-1]

        # Define wavefront 
        npix = 1024 // wavefront_downsample
        diameter = pupil_plane.pixelscale.to('m/pix').value * pupil_plane.npix

        # Calculate detector parameters
        det_npix = (detector_plane.fov_pixels * detector_plane.oversample).value
        pscale = (detector_plane.pixelscale / detector_plane.oversample).to('radian/pix').value
        #print(f'pixel scale: {pscale}')

        # Define optical layers
        pupil_mask = pupil_plane.amplitude
        pupil_opd = pupil_plane.opd

        if wavefront_downsample > 1:
            pupil_mask = rebin(pupil_mask, (npix, npix))
            pupil_opd = rebin(pupil_opd, (npix, npix))

        optical_layers = [
            #Plane 0: Pupil plane: JWST Entrance Pupil
            (dl.Optic(pupil_mask, pupil_opd), "Pupil"),

            #Plane 1: Coordinate Inversion in y axis
            (InvertY(), "InvertY")
        ]

        layer_index = 2

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
            layer_index = 4

        detector_mask = nircam.planes[layer_index].amplitude
        detector_opd = nircam.planes[layer_index].opd

        if wavefront_downsample > 1:
            detector_mask = rebin(detector_mask, (npix, npix))
            detector_opd = rebin(detector_opd, (npix, npix))

        optical_layers.extend([
            #Plane 4: Pupil plane: NIRCamLWA internal WFE at V2V3=(1.46,-6.75)', near Z430R_A
            (NIRCamFieldAndWavelengthDependentAberration(webb_osys, detector_mask, detector_opd, nircam.planes[layer_index].zernike_coeffs), "NIRCamFieldAndWavelengthDependentAberration"),

            #Plane 5: Detector plane: NIRCam detector
            dl.MFT(det_npix, pscale)
        ])

        optics = dl.LayeredOptics(wf_npixels=npix, diameter=diameter, layers=optical_layers)

        # Define detector
        detector_layers = []

        if detector_keep_size == True:
            detector_layers.append(dl.IntegerDownsample(detector_oversample))

        detector = dl.LayeredDetector(detector_layers)

        if nlambda is None or nlambda == 0:
            nlambda = webb_osys._get_default_nlambda(webb_osys.filter)

        wavelengths, weights = webb_osys._get_weights(source=None, nlambda=nlambda, monochromatic=monochromatic)

        self.filter_wavelengths = wavelengths
        self.filter_weights = weights

        if source is None:
            spectrum = dl.Spectrum(wavelengths, weights)

            position = [0, 0]
            if offset is not None:
                position = offset

            # wavelengths are still required even if they are discarded
            source = dl.PointSource(wavelengths=wavelengths, spectrum=spectrum, position=position)

        super().__init__(optics=optics, sources=(source, 'source'), detector=detector)