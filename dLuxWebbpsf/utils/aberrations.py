import webbpsf
from webbpsf import constants as const
import jax.numpy as np
import dLux
import numpy
import dLux.utils as dlu
from poppy.zernike import hexike_basis, zernike_basis
from jax import Array


__all__ = ["generate_jwst_hexike_basis", "generate_jwst_secondary_basis"]

def get_noll_indices(
    radial_orders: Array | list = None,
    noll_indices: Array | list = None
):
    if radial_orders is not None:
        radial_orders = np.array(radial_orders)

        if (radial_orders < 0).any():
            raise ValueError("Radial orders must be >= 0")

        noll_indices = []
        for order in radial_orders:
            start = dlu.triangular_number(order)
            stop = dlu.triangular_number(order + 1)
            noll_indices.append(np.arange(start, stop) + 1)
        noll_indices = np.concatenate(noll_indices)

    elif noll_indices is None:
        raise ValueError("Must specify either radial_orders or noll_indices")

    if noll_indices is not None:
        noll_indices = np.array(noll_indices, dtype=int)

    return noll_indices

def get_hexike_basis(
    noll_indices: Array | list,
    npix: int,
    pscale: float,
    *,
    seg_rad: float = None,
    keys: Array | list = None,
    shifts: Array | list = None,
):
    if seg_rad is None:
        seg_rad = const.JWST_SEGMENT_RADIUS

    if keys is None:
        keys = const.SEGNAMES_WSS

    if shifts is None:
        shifts = np.zeros(2)
    
    nterms = int(noll_indices.max())
    seg_cens = dict(const.JWST_PRIMARY_SEGMENT_CENTERS)

    # Generating a basis for each segment (all terms up to highest noll index)
    basis = []
    for key in keys:  # cycling through segments
        centre = np.array([-1, 1]) * (np.array(seg_cens[key]) - shifts)
        rhos, thetas = numpy.array(
            dlu.pixel_coordinates(
                npixels=(npix, npix),
                pixel_scales=pscale,
                offsets=tuple(centre),
                polar=True,
            )
        )
        basis.append(
            hexike_basis(nterms, npix, rhos / seg_rad, thetas, outside=0.0)
        )  # appending basis

    # calculating overlaps and missing pixels
    pistons = sum(np.array([x[0] for x in basis]))
    mask = np.where(pistons==1, 1, 0)

    # reducing back down to request noll indices
    basis = np.array(basis)[:, noll_indices - 1, ...]

    return basis, mask

def generate_jwst_hexike_basis(
    npix: int = 1024,
    radial_orders: Array | list = None,
    noll_indices: Array | list = None,
    AMI: bool = False,
    mask: bool = False,
):
    """
    Generates a basis for each segment of the JWST primary mirror.

    Parameters
    ----------
    npix : int
        Number of pixels in the output basis.
    radial_orders : Array = None
        The radial orders of the zernike polynomials to be used for the
        aberrations. Input of [0, 1] would give [Piston, Tilt X, Tilt Y],
        [1, 2] would be [Tilt X, Tilt Y, Defocus, Astig X, Astig Y], etc.
        The order must be increasing but does not have to be consecutive.
        If you want to specify specific zernikes across radial orders the
        noll_indices argument should be used instead.
    noll_indices : Array = None
        The zernike noll indices to be used for the aberrations. [1, 2, 3]
        would give [Piston, Tilt X, Tilt Y], [2, 3, 4] would be [Tilt X,
        Tilt Y, Defocus].
    AMI : bool
        Whether to use the AMI mask or not.
    mask : bool
        Whether to apodise the basis with the AMI mask or not. Recommended is False.

    Returns
    -------
    basis : Array
        Array of shape (nsegments, nterms, npix, npix) containing the basis for each segment.
    """

    noll_indices = get_noll_indices(radial_orders, noll_indices)
    nterms = int(noll_indices.max())

    # Get webbpsf model
    niriss = webbpsf.NIRISS()

    if AMI:
        if mask is False:
            seg_rad = const.JWST_SEGMENT_RADIUS
            shifts = np.zeros(2)  # no shift for full pupil
        elif mask:
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
        seg_rad = const.JWST_SEGMENT_RADIUS  # TODO check for overlapping pixels
        shifts = np.zeros(2)  # no shift for full pupil
        keys = const.SEGNAMES_WSS  # all mirror segments

    else:
        raise ValueError("AMI must be a boolean")

    niriss_osys = niriss.get_optical_system()
    seg_cens = dict(const.JWST_PRIMARY_SEGMENT_CENTERS)
    pscale = niriss_osys.planes[0].pixelscale.value * 1024 / npix

    # Generating a basis for each segment (all terms up to highest noll index)
    basis = []
    for key in keys:  # cycling through segments
        centre = np.array([-1, 1]) * (np.array(seg_cens[key]) - shifts)
        rhos, thetas = numpy.array(
            dlu.pixel_coordinates(
                npixels=(npix, npix),
                pixel_scales=pscale,
                offsets=tuple(centre),
                polar=True,
            )
        )
        basis.append(
            hexike_basis(nterms, npix, rhos / seg_rad, thetas, outside=0.0)
        )  # appending basis

    # reducing back down to request noll indices
    basis = np.array(basis)[:, noll_indices - 1, ...]

    if mask and AMI is True:
        # Scale mask
        transmission = niriss_osys.planes[3].amplitude
        transmission = dLux.utils.scale_array(transmission, npix, 1)
        basis = np.flip(
            basis, axis=(2, 3)
        )  # need to apply flip for AMI as plane 1 is a flip
        basis *= transmission

    return basis

def generate_jwst_secondary_basis(
    npix: int = 1024,
    radial_orders: Array | list = None,
    noll_indices: Array | list = None,
):
    """
    Generates a basis for each segment of the JWST secondary mirror.

    Parameters
    ----------
    npix : int
        Number of pixels in the output basis.
    radial_orders : Array = None
        The radial orders of the zernike polynomials to be used for the
        aberrations. Input of [0, 1] would give [Piston, Tilt X, Tilt Y],
        [1, 2] would be [Tilt X, Tilt Y, Defocus, Astig X, Astig Y], etc.
        The order must be increasing but does not have to be consecutive.
        If you want to specify specific zernikes across radial orders the
        noll_indices argument should be used instead.
    noll_indices : Array = None
        The zernike noll indices to be used for the aberrations. [1, 2, 3]
        would give [Piston, Tilt X, Tilt Y], [2, 3, 4] would be [Tilt X,
        Tilt Y, Defocus].

    Returns
    -------
    basis : Array
        Array of shape (nsegments, nterms, npix, npix) containing the basis for each segment.
    """

    # Aberrations
    if radial_orders is not None:
        radial_orders = np.array(radial_orders)

        if (radial_orders < 0).any():
            raise ValueError("Radial orders must be >= 0")

        noll_indices = []
        for order in radial_orders:
            start = dlu.triangular_number(order)
            stop = dlu.triangular_number(order + 1)
            noll_indices.append(np.arange(start, stop) + 1)
        noll_indices = np.concatenate(noll_indices)

    elif noll_indices is None:
        raise ValueError("Must specify either radial_orders or noll_indices")

    if noll_indices is not None:
        noll_indices = np.array(noll_indices, dtype=int)

    nterms = int(noll_indices.max())

    rhos, thetas = numpy.array(
        dlu.pixel_coordinates((npix, npix), pixel_scales=2 / npix, polar=True)
    )
    secondary_basis = zernike_basis(nterms, npix, rhos, thetas, outside=0.0)

    # reducing back down to requested noll indices
    secondary_basis = np.array(secondary_basis)[noll_indices - 1, ...]

    return secondary_basis
