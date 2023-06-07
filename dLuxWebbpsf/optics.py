import dLux as dl
import webbpsf

from dLuxWebbpsf.optical_layers import *
 
__all__ = ["NIRCamCoron"]


def get_osys(instrument, filtr, pup_mask, crn_mask, detector, aperture, options=None):
    webb_osys = getattr(webbpsf, instrument)()
    webb_osys.filter = filtr
    webb_osys.pupil_mask = pup_mask
    webb_osys.image_mask = crn_mask
    webb_osys.detector = detector
    webb_osys.set_position_from_aperture_name(aperture)

    if options is not None:
        webb_osys.options.update(options)
    
    return webb_osys

class NIRCamCoron(dl.optics.LayeredOptics):
    """
    A class that represents the optical system of the NIRCam coronagraph instrument on the James Webb Space Telescope.
    Inherits from dl.optics.LayeredOptics.
    """

    def __init__(self,
                 filter, # string: name of the filter
                 pupil_mask, # string: name of the pupil mask
                 coron_mask, # string: name of the image mask
                 detector, # string: name of the detector
                 aperture, # string: name of the aperture
                 fft_oversample=4, # int: oversampling factor for the FFT
                 detector_oversample=None, # int: oversampling factor for the detector
                 fov_arcsec=5, # float: field of view in arcseconds
                 fov_pixels=None, # int: number of pixels in the field of view
                 options=None # dict: additional options for the webbpsf.OpticalSystem
                 ):
        """
        Constructor for the NIRCamCoron class.
        Initializes the optical system of the NIRCam coronagraph instrument on the James Webb Space Telescope.
        """

        # Get the webbpsf.OpticalSystem object
        webb_osys = get_osys("NIRCam", filter, pupil_mask, coron_mask, detector, aperture, options)

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

        # Aperture and OPD
        webb_aper = nircam.planes[0].amplitude
        webb_opd = nircam.planes[0].opd
        pupil_mask = nircam.planes[3].amplitude

        # Calculate detector parameters
        det_npix = (det_plane.fov_pixels * det_plane.oversample).value
        pscale = det_plane.pixelscale/det_plane.oversample
        pscale = pscale.to('radian/pix').value

        # Make layers
        optical_layers = [
            #Plane 0: Pupil plane: JWST Entrance Pupil
            (dl.Optic(webb_aper, webb_opd), "Pupil"),

            #Plane 1: Coordinate Inversion in y axis
            (InvertY(), "InvertY"),

            #Plane 2: Image plane: NIRCam mask
            dl.FFT(focal_length=None, pad=nircam.planes[2].oversample),
            (NircamCirc(nircam.planes[2].sigma, diameter, npix, nircam.planes[2].oversample), "NIRCamCirc"),
            dl.FFT(focal_length=None, pad=nircam.planes[2].oversample, inverse=True),

            #Plane 3: Pupil plane
            (dl.Optic(pupil_mask), "PupilMask"),

            #Plane 4: Pupil plane: NIRCamLWA internal WFE at V2V3=(1.46,-6.75)', near Z430R_A
            dl.Optic(nircam.planes[4].amplitude),
            (NIRCamFieldAndWavelengthDependentAberration(webb_osys, nircam.planes[4].opd, nircam.planes[4].zernike_coeffs), "NIRCamFieldAndWavelengthDependentAberration"),

            #Plane 5: Detector plane: NIRCam detector (79x79 pixels, 0.063 arcsec / pix)
            dl.MFT(det_npix, pscale)
        ]

        # Call the constructor of the parent class
        super().__init__(wf_npixels=npix, diameter=diameter, layers=optical_layers)
