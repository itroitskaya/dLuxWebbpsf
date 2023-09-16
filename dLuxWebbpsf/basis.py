import webbpsf
import webbpsf.constants as const
import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
import numpy as onp

from matplotlib import pyplot as plt
from jax import config
from poppy.zernike import hexike_basis

def jwst_hexike_basis(nterms, npix, pscale):
    # Get webbpsf model
    seg_rad = const.JWST_SEGMENT_RADIUS  # TODO check for overlapping pixels
    keys = const.SEGNAMES_WSS  # all mirror segments

    seg_cens = dict(const.JWST_PRIMARY_SEGMENT_CENTERS)

    # Generating a basis for each segment
    basis = []
    for key in keys:  # cycling through segments
        centre = onp.array(seg_cens[key])
        rhos, thetas = onp.array(dlu.pixel_coordinates((npix, npix), pscale, offsets=tuple(centre), polar=True))
        basis.append(hexike_basis(nterms, npix, rhos / seg_rad, thetas, outside=0.0))  # appending basis

    pistons = sum(np.array([x[0] for x in basis]))
    mask = np.where(pistons==1, 1, 0)

    return mask, np.array(basis)
