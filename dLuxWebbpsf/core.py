import dLux as dl
from dLuxWebbpsf.optics import *
 
__all__ = ["get_nircam_optics"]

def get_nircam_optics(webb_osys,
                        fft_oversample=4,
                        detector_oversample=None,
                        fov_arcsec=5,
                        fov_pixels=None,
                        options=None):
    nircam = webb_osys.get_optical_system(fft_oversample=fft_oversample,
                        detector_oversample=detector_oversample,
                        fov_arcsec=fov_arcsec, fov_pixels=fov_pixels,
                        options=options)
    pupil_plane = nircam.planes[0]
    det_plane = nircam.planes[-1]

    # Wavefront 
    npix = 1024
    diameter = pupil_plane.pixelscale.to('m/pix').value * pupil_plane.npix
    # diameter == 6.603464

    # Aperture and OPD
    webb_aper = nircam.planes[0].amplitude
    webb_opd = nircam.planes[0].opd
    pupil_mask = nircam.planes[3].amplitude

    # Detector
    det_npix = (det_plane.fov_pixels * det_plane.oversample).value
    print(det_npix)
    pscale = det_plane.pixelscale/det_plane.oversample
    print(pscale)
    pscale = pscale.to('radian/pix').value

    # pscale == 7.958548e-08

    # Make layers

    def get_optical_layers():
        optical_layers = [
            #Plane 0: Pupil plane: JWST Entrance Pupil
            dl.CreateWavefront(npix, diameter, 'Angular'),
            dl.TransmissiveOptic(webb_aper),
            dl.AddOPD(webb_opd),

            #Plane 1: Coordinate Inversion in y axis
            InvertY(),

            #Plane 2: Image plane: MASK430R
            #TODO: Check if padding is not needed
            Pad(npix * nircam.planes[2].oversample),
            dl.AngularFFT(),
            NircamCirc(nircam.planes[2].sigma, diameter, npix, nircam.planes[2].oversample),
            dl.AngularFFT(inverse=True),
            Crop(npix),

            #Plane 3: Pupil plane: CIRCLYOT
            dl.TransmissiveOptic(pupil_mask),

            #Plane 4: Pupil plane: NIRCamLWA internal WFE at V2V3=(1.46,-6.75)', near Z430R_A
            dl.TransmissiveOptic(nircam.planes[4].amplitude),
            NIRCamFieldAndWavelengthDependentAberration(webb_osys, nircam.planes[4].opd, nircam.planes[4].zernike_coeffs),

            #Plane 5: Detector plane: NIRCam detector (79x79 pixels, 0.063 arcsec / pix)
            dl.AngularMFT(det_npix, pscale)
        ]
        return optical_layers

    optical_layers = get_optical_layers()

    optics = dl.Optics(optical_layers)

    return optics