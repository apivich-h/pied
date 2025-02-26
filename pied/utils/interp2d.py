"""
Code adapted from https://github.com/adam-coogan/jaxinterp2d/blob/master/src/jaxinterp2d/__init__.py
"""

from typing import Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


Array = jnp.ndarray


class GridInterpolator:

    def __init__(self, lower_bound, upper_bound, grid_size, dims):
        """
        Initializer.

        Args:
            limits: collection of pairs specifying limits of input variables along
                each dimension of ``values``
            values: values to interpolate. These must be defined on a regular grid.
            mode: how to handle out of bounds arguments; see docs for ``map_coordinates``
            cval: constant fill value; see docs for ``map_coordinates``
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.grid_size = grid_size
        self.dims = dims
        
        self.dx = (self.lower_bound - self.upper_bound) / (self.grid_size - 1)
        
        space = jnp.linspace(lower_bound, upper_bound, num=grid_size, endpoint=True)
        self.nodal_points = jnp.array(jnp.meshgrid(*[space for _ in range(self.dims)])).reshape(dims, -1).T
        
        self.basis_fn = jax.jit(lambda x: jnp.prod(jnp.maximum(1. - jnp.abs((self.nodal_points - x[None, :]) / self.dx), 0.), axis=1))

    def __call__(self, values, x) -> Array:
        return jnp.sum(values * self.basis_fn(x))


# class GridInterpolator:

#     def __init__(self, limits: jnp.ndarray, values_dim: Tuple[int]):
#         """
#         Initializer.

#         Args:
#             limits: collection of pairs specifying limits of input variables along
#                 each dimension of ``values``
#             values: values to interpolate. These must be defined on a regular grid.
#             mode: how to handle out of bounds arguments; see docs for ``map_coordinates``
#             cval: constant fill value; see docs for ``map_coordinates``
#         """
#         self.values_dim = values_dim
#         self.limits = limits

#     def __call__(self, values, x) -> Array:
#         """
#         Perform interpolation.

#         Args:
#             coords: point at which to interpolate. These will be broadcasted if
#                 they are not the same shape.

#         Returns:
#             Interpolated values, with extrapolation handled according to ``mode``.
#         """
#         x = [
#             (x[...,i] - lo) * (n - 1) / (hi - lo)
#             for i, ((lo, hi), n) in enumerate(zip(self.limits, self.values_dim))
#         ]
#         # x = [
#         #     (c - lo) * (n - 1) / (hi - lo)
#         #     for (lo, hi), c, n in zip(self.limits, x, self.values_dim)
#         # ]
#         return map_coordinates(values, x, mode="nearest", order=1)

