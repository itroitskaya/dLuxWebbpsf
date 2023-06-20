import dLux as dl
import webbpsf

from dLuxWebbpsf.optical_layers import *
 
__all__ = ["NIRCam"]

class NIRCam(dl.optics.LayeredOptics):

    webb_osys : None

    """
    A class that represents the optical system of the NIRCam coronagraph instrument on the James Webb Space Telescope.
    Inherits from dl.optics.LayeredOptics.
    """

    def __init__(self,
                 *,
                 filter=None, # string: name of the filter
                 pupil_mask=None, # string: name of the pupil mask
                 coron_mask=None, # string: name of the image mask
                 detector=None, # string: name of the detector
                 aperture=None, # string: name of the aperture
                 fft_oversample=4, # int: oversampling factor for the FFT
                 detector_oversample=None, # int: oversampling factor for the detector
                 fov_arcsec=5, # float: field of view in arcseconds
                 fov_pixels=None, # int: number of pixels in the field of view
                 options=None, # dict: additional options for the webbpsf.OpticalSystem
                 nlambda=None, # int: number of wavelengths
                 monochromatic=None, # float: wavelength in microns
                 webb_osys=None, # webbpsf.OpticalSystem: a pre-defined webbpsf.OpticalSystem object
                 ):
        """
        Constructor for the NIRCamCoron class.
        Initializes the optical system of the NIRCam coronagraph instrument on the James Webb Space Telescope.
        """

        # TODO: Assert that the arguments are provided

        if webb_osys is None:
            webb_osys = webbpsf.NIRCam()
            webb_osys.filter = filter
            webb_osys.pupil_mask = pupil_mask
            webb_osys.image_mask = coron_mask
            webb_osys.detector = detector
            webb_osys.set_position_from_aperture_name(aperture)

            if options is not None:
                webb_osys.options.update(options)

        self.webb_osys = webb_osys

        # Get the dl.optics.OpticalSystem object
        nircam = webb_osys.get_optical_system(fft_oversample=fft_oversample,
                            detector_oversample=detector_oversample,
                            fov_arcsec=fov_arcsec, fov_pixels=fov_pixels,
                            options=options)
        
        # Get the pupil and detector planes
        pupil_plane = nircam.planes[0]
        det_plane = nircam.planes[-1]

        # Define wavefront 
        npix = 1024
        diameter = pupil_plane.pixelscale.to('m/pix').value * pupil_plane.npix

        # Calculate detector parameters
        det_npix = (det_plane.fov_pixels * det_plane.oversample).value
        pscale = det_plane.pixelscale/det_plane.oversample
        pscale = pscale.to('radian/pix').value

        optical_layers = [
            #Plane 0: Pupil plane: JWST Entrance Pupil
            (dl.Optic(nircam.planes[0].amplitude, nircam.planes[0].opd), "Pupil"),

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

        optical_layers.extend([
            #Plane 4: Pupil plane: NIRCamLWA internal WFE at V2V3=(1.46,-6.75)', near Z430R_A
            dl.Optic(nircam.planes[layer_index].amplitude),
            (NIRCamFieldAndWavelengthDependentAberration(webb_osys, nircam.planes[layer_index].opd, nircam.planes[layer_index].zernike_coeffs), "NIRCamFieldAndWavelengthDependentAberration"),

            #Plane 5: Detector plane: NIRCam detector (79x79 pixels, 0.063 arcsec / pix)
            dl.MFT(det_npix, pscale)
        ])

        # Call the constructor of the parent class
        super().__init__(wf_npixels=npix, diameter=diameter, layers=optical_layers)

    def get_wavelengths(self, *, monochromatic=None, nlambda=None):
        """
        Returns filtered wavelengths of the optical system.
        """
        
        if nlambda is None or nlambda == 0:
            nlambda = self.webb_osys._get_default_nlambda(self.webb_osys.filter)

        wavelens, weights = self.webb_osys._get_weights(source=None, nlambda=nlambda, monochromatic=monochromatic)

        return (wavelens, weights)
