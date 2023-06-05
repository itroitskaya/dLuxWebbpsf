import jax
from jax.tree_util import tree_map
import jax.numpy as np


__all__ = ["get_pixel_positions"]

Array = jax.numpy.ndarray

def get_pixel_positions(npixels, pixel_scales = None, indexing = 'xy'):
    # Turn inputs into tuples
    if isinstance(npixels, int):
        npixels = (npixels,)

        if pixel_scales is None:
            pixel_scales = (1.,)
        elif not isinstance(pixel_scales, (float, Array)):
            raise ValueError("pixel_scales must be a float or Array if npixels "
                             "is an int.")
        else:
            pixel_scales = (pixel_scales,)
        
    # Check input 
    else:
        if pixel_scales is None:
            pixel_scales = tuple([1.]*len(npixels))
        elif isinstance(pixel_scales, float):
            pixel_scales = tuple([pixel_scales]*len(npixels))
        elif not isinstance(pixel_scales, tuple):
            raise ValueError("pixel_scales must be a tuple if npixels is a tuple.")
        else:
            if len(pixel_scales) != len(npixels):
                raise ValueError("pixel_scales must have the same length as npixels.")
    
    def pixel_fn(n, scale):
        pix = np.arange(n) - n / 2.
        pix *= scale
        return pix
    
    pixels = tree_map(pixel_fn, npixels, pixel_scales)

    positions = np.array(np.meshgrid(*pixels, indexing=indexing))

    return np.squeeze(positions)

