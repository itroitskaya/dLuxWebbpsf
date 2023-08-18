import webbpsf
from webbpsf import constants as const
import jax.numpy as np
import dLux
import numpy
import dLux.utils as dlu
from poppy.zernike import hexike_basis


def generate_jwst_hexike_basis(
    nterms: int = 10, npix: int = 1024, AMI: bool = False, mask: bool = False
):
    """
    Generates a basis for each segment of the JWST primary mirror.

    Parameters
    ----------
    nterms : int
        Number of terms in the hexike basis.
    npix : int
        Number of pixels in the output basis.
    AMI : bool
        Whether to use the AMI mask or not.
    mask : bool
        Whether to apodise the basis with the AMI mask or not. Recommended is False.

    Returns
    -------
    basis : Array
        Array of shape (nsegments, nterms, npix, npix) containing the basis for each segment.
    """

    # Get webbpsf model
    niriss = webbpsf.NIRISS()

    if AMI:
        if mask is None:
            amplitude_plane = 0
            seg_rad = const.JWST_SEGMENT_RADIUS
            shifts = np.zeros(2)  # no shift for full pupil
        elif mask is not None:
            amplitude_plane = 3
            # "The hexagonal sub-apertures are 0.82 m from one flat side of the hexagon to the other projected onto
            # the primary, as designed." - Rachel Cooper, 15/8/2023
            seg_rad = (0.82 / 2) / 0.8660254038
            # shifts from WebbPSF: CV3 on-orbit estimate (RPT028027) + OTIS delta from predicted (037134)
            shifts = const.JWST_CIRCUMSCRIBED_DIAMETER * np.array([0.0243, -0.0141])
        else:
            raise ValueError("AMI_type must be 'full' or 'masked'")
        keys = [
            "B2-9",
            "B3-11",
            "B4-13",
            "C5-16",
            "B6-17",
            "C1-8",
            "C6-18",
        ]  # the segments that are used in AMI
        niriss.pupil_mask = "MASK_NRM"

    elif not AMI:
        amplitude_plane = 0
        seg_rad = const.JWST_SEGMENT_RADIUS  # TODO check for overlapping pixels
        shifts = np.zeros(2)  # no shift for full pupil
        keys = const.SEGNAMES_WSS  # all mirror segments

    else:
        raise ValueError("AMI must be a boolean")

    niriss_osys = niriss.get_optical_system()
    seg_cens = dict(const.JWST_PRIMARY_SEGMENT_CENTERS)
    pscale = niriss_osys.planes[0].pixelscale.value * 1024 / npix

    # Scale mask
    transmission = niriss_osys.planes[amplitude_plane].amplitude
    transmission = dLux.utils.scale_array(transmission, npix, 1)

    # Generating a basis for each segment
    basis = []
    for key in keys:  # cycling through segments
        centre = np.array(seg_cens[key]) - shifts
        rhos, thetas = numpy.array(
            dlu.pixel_coordinates(
                (npix, npix), pscale, offsets=tuple(centre), polar=True
            )
        )
        basis.append(
            hexike_basis(nterms, npix, rhos / seg_rad, thetas, outside=0.0)
        )  # appending basis

    basis = np.array(basis)

    if mask:
        basis = np.flip(
            basis, axis=(2, 3)
        )  # need to apply flip for AMI as plane 1 is a flip
        basis *= transmission

    return basis
