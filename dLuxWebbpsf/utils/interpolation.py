from jax import Array
import jax.numpy as np
import dLux.utils as dlu

__all__ = ["rotate"]


def rotate(array: Array, angle: Array, order: int = 3) -> Array:
    """
    Rotates an array by the angle, using linear interpolation.

    Parameters
    ----------
    array : Array
        The array to rotate.
    angle : Array, radians
        The angle to rotate the array by.
    order : int = 3
        The interpolation order to use.

    Returns
    -------
    array : Array
        The rotated array.
    """

    def _rotate(coordinates: Array, rotation: Array) -> Array:
        x, y = coordinates[0], coordinates[1]
        new_x = np.cos(-rotation) * x + np.sin(-rotation) * y
        new_y = -np.sin(-rotation) * x + np.cos(-rotation) * y
        return np.array([new_x, new_y])

    # Get coordinates
    npixels = array.shape[0]
    centre = (npixels - 1) / 2
    coordinates = dlu.pixel_coordinates((npixels, npixels), indexing="ij")
    coordinates_rotated = _rotate(coordinates, angle) + centre

    # Interpolate
    return _map_coordinates(
        array, coordinates_rotated, order=order, mode="constant", cval=0.0
    )


"""
NOTE: All the code below is a copy-paste from my jax fork which implements
the map_coordinates function, extended for cubic B-Splines with natural boundary
conditions. This needs to be cleaned up or moved into a small external
repo, but for now this will work to get us machine-precision match to webbpsf.
"""


import functools
import itertools
import operator
import textwrap
from typing import Callable, Dict, List, Sequence, Tuple

import scipy.ndimage

from jax._src import api
from jax._src import util
from jax import lax, vmap

# from jax._src.numpy import lax_numpy as jnp
import jax.numpy as jnp
from jax._src.numpy.util import _wraps
from jax._src.typing import ArrayLike, Array
from jax._src.util import safe_zip as zip
from jax._src.numpy.linalg import inv


def _nonempty_prod(arrs: Sequence[Array]) -> Array:
    return functools.reduce(operator.mul, arrs)


def _nonempty_sum(arrs: Sequence[Array]) -> Array:
    return functools.reduce(operator.add, arrs)


def _mirror_index_fixer(index: Array, size: int) -> Array:
    s = size - 1  # Half-wavelength of triangular wave
    # Scaled, integer-valued version of the triangular wave |x - round(x)|
    return jnp.abs((index + s) % (2 * s) - s)


def _reflect_index_fixer(index: Array, size: int) -> Array:
    return jnp.floor_divide(
        _mirror_index_fixer(2 * index + 1, 2 * size + 1) - 1, 2
    )


_INDEX_FIXERS: Dict[str, Callable[[Array, int], Array]] = {
    "constant": lambda index, size: index,
    "nearest": lambda index, size: jnp.clip(index, 0, size - 1),
    "wrap": lambda index, size: index % size,
    "mirror": _mirror_index_fixer,
    "reflect": _reflect_index_fixer,
}


def _round_half_away_from_zero(a: Array) -> Array:
    return a if jnp.issubdtype(a.dtype, jnp.integer) else lax.round(a)


def _nearest_indices_and_weights(
    coordinate: Array,
) -> List[Tuple[Array, ArrayLike]]:
    index = _round_half_away_from_zero(coordinate).astype(jnp.int32)
    weight = coordinate.dtype.type(1)
    return [(index, weight)]


def _linear_indices_and_weights(
    coordinate: Array,
) -> List[Tuple[Array, ArrayLike]]:
    lower = jnp.floor(coordinate)
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = lower.astype(jnp.int32)
    return [(index, lower_weight), (index + 1, upper_weight)]


def _cubic_indices_and_weights(
    coordinate: Array,
) -> List[Tuple[Array, ArrayLike]]:
    return [(coordinate, jnp.zeros(coordinate.shape))]


def _build_matrix(n: int, diag: float = 4) -> Array:
    A = diag * jnp.eye(n)
    for i in range(n - 1):
        A = A.at[i, i + 1].set(1)
        A = A.at[i + 1, i].set(1)
    return A


def _construct_vector(data: Array, c2: Array, cnp2: Array) -> Array:
    yvec = data[1:-1]
    first = data[1] - c2
    last = data[-2] - cnp2
    yvec = yvec.at[0].set(first)
    yvec = yvec.at[-1].set(last)
    return yvec


def _solve_coefficients(data: Array, A_inv: Array, h=1) -> Array:
    # Calcualte second and second last coefficients
    c2 = 1 / 6 * data[0]
    cnp2 = 1 / 6 * data[-1]

    # Solve for internal cofficients
    yvec = _construct_vector(data, c2, cnp2)
    cs = jnp.dot(A_inv, yvec)

    # Calculate first and last coefficients
    c1 = 2 * c2 - cs[0]
    cnp3 = 2 * cnp2 - cs[-1]
    return jnp.concatenate([jnp.array([c1, c2]), cs, jnp.array([cnp2, cnp3])])


def _spline_coefficients(data: Array) -> Array:
    ndim = data.ndim
    for i in range(ndim):
        axis = ndim - i - 1
        A_inv = inv(_build_matrix(data.shape[axis] - 2))
        fn = lambda x: _solve_coefficients(x, A_inv)
        for j in range(ndim - 2, -1, -1):
            ax = int(j >= axis)
            fn = vmap(fn, ax, ax)
        data = fn(data)
    return data


def _spline_basis(t: Array) -> Array:
    at = jnp.abs(t)
    fn1 = lambda t: (2 - t) ** 3
    fn2 = lambda t: 4 - 6 * t**2 + 3 * t**3
    return jnp.where(
        at >= 1, jnp.where(at <= 2, fn1(at), 0), jnp.where(at <= 1, fn2(at), 0)
    )


def _spline_value(
    coefficients: Array, coordinate: Array, indexes: Array
) -> Array:
    coefficient = jnp.squeeze(
        lax.dynamic_slice(coefficients, indexes, [1] * coefficients.ndim)
    )
    fn = vmap(lambda x, i: _spline_basis(x - i + 1), (0, 0))
    return coefficient * fn(coordinate, indexes).prod()


def _spline_point(coefficients: Array, coordinate: Array) -> Array:
    index_fn = lambda x: (jnp.arange(0, 4) + jnp.floor(x)).astype(int)
    index_vals = vmap(index_fn)(coordinate)
    indexes = jnp.array(jnp.meshgrid(*index_vals, indexing="ij"))
    fn = lambda index: _spline_value(coefficients, coordinate, index)
    return vmap(fn)(indexes.reshape(coefficients.ndim, -1).T).sum()


def _cubic_spline(input: Array, coordinates: Array) -> Array:
    coefficients = _spline_coefficients(input)
    points = coordinates.reshape(input.ndim, -1).T
    fn = lambda coord: _spline_point(coefficients, coord)
    return vmap(fn)(points).reshape(coordinates.shape[1:])


@functools.partial(api.jit, static_argnums=(2, 3, 4))
def _map_coordinates(
    input: ArrayLike,
    coordinates: Sequence[ArrayLike],
    order: int,
    mode: str,
    cval: ArrayLike,
) -> Array:
    input_arr = jnp.asarray(input)
    coordinate_arrs = [jnp.asarray(c) for c in coordinates]
    cval = jnp.asarray(cval, input_arr.dtype)

    if len(coordinates) != input_arr.ndim:
        raise ValueError(
            "coordinates must be a sequence of length input.ndim, but "
            "{} != {}".format(len(coordinates), input_arr.ndim)
        )

    index_fixer = _INDEX_FIXERS.get(mode)
    if index_fixer is None:
        raise NotImplementedError(
            "jax.scipy.ndimage.map_coordinates does not yet support mode {}. "
            "Currently supported modes are {}.".format(
                mode, set(_INDEX_FIXERS)
            )
        )

    if mode == "constant":
        is_valid = lambda index, size: (0 <= index) & (index < size)
    else:
        is_valid = lambda index, size: True

    if order == 0:
        interp_fun = _nearest_indices_and_weights
    elif order == 1:
        interp_fun = _linear_indices_and_weights
    elif order == 3:
        interp_fun = _cubic_indices_and_weights
    else:
        raise NotImplementedError(
            "jax.scipy.ndimage.map_coordinates currently requires order<=1"
        )

    valid_1d_interpolations = []
    for coordinate, size in zip(coordinate_arrs, input_arr.shape):
        interp_nodes = interp_fun(coordinate)
        valid_interp = []
        for index, weight in interp_nodes:
            fixed_index = index_fixer(index, size)
            valid = is_valid(index, size)
            valid_interp.append((fixed_index, valid, weight))
        valid_1d_interpolations.append(valid_interp)

    outputs = []
    for items in itertools.product(*valid_1d_interpolations):
        indices, validities, weights = util.unzip3(items)
        if order == 3:
            if mode == "reflect":
                raise NotImplementedError(
                    "Cubic interpolation with mode='reflect' is not implemented."
                )
            interpolated = _cubic_spline(input_arr, jnp.array(indices))
            if all(valid is True for valid in validities):
                outputs.append(interpolated)
            else:
                all_valid = functools.reduce(operator.and_, validities)
                outputs.append(jnp.where(all_valid, interpolated, cval))
        else:
            if all(valid is True for valid in validities):
                # fast path
                contribution = input_arr[indices]
            else:
                all_valid = functools.reduce(operator.and_, validities)
                contribution = jnp.where(all_valid, input_arr[indices], cval)
            outputs.append(_nonempty_prod(weights) * contribution)
    result = _nonempty_sum(outputs)
    if jnp.issubdtype(input_arr.dtype, jnp.integer):
        result = _round_half_away_from_zero(result)
    return result.astype(input_arr.dtype)
