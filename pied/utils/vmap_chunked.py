"""
Based on implementation from netket
url: https://netket.readthedocs.io/en/stable/_modules/netket/jax/_vmap_chunked.html
"""


from typing import Callable, Optional
from functools import partial

import jax
import jax.numpy as jnp

import jax
import jax.numpy as jnp

from jax.api_util import argnums_partial

if jax.__version_info__ >= (0, 4, 16):
    from jax.extend import linear_util as lu
else:
    from jax import linear_util as lu

# from netket.utils import HashablePartial

_tree_add = partial(jax.tree_map, jax.lax.add)
_tree_zeros_like = partial(jax.tree_map, lambda x: jnp.zeros(x.shape, dtype=x.dtype))


# TODO put it somewhere
def _multimap(f, *args):
    try:
        return tuple(map(lambda a: f(*a), zip(*args)))
    except TypeError:
        return f(*args)


def scan_append_reduce(f, x, append_cond, op=_tree_add, zero_fun=_tree_zeros_like):
    """Evaluate f element by element in x while appending and/or reducing the results

    Args:
        f: a function that takes elements of the leading dimension of x
        x: a pytree where each leaf array has the same leading dimension
        append_cond: a bool (if f returns just one result) or a tuple of bools (if f returns multiple values)
            which indicates whether the individual result should be appended or reduced
        op: a function to (pairwise) reduce the specified results. Defaults to a sum.
        zero_fun: a function which prepares the zero element of op for a given input shape/dtype tree. Defaults to zeros.

    Returns:
        returns the (tuple of) results corresponding to the output of f
        where each result is given by:
        if append_cond is True:
            a (pytree of) array(s) with leading dimension same as x,
            containing the evaluation of f at each element in x
        else (append_cond is False):
            a (pytree of) array(s) with the same shape as the corresponding output of f,
            containing the reduction over op of f evaluated at each x


    Example:

        import jax.numpy as jnp
        from netket.jax import scan_append_reduce

        def f(x):
             y = jnp.sin(x)
             return y, y, y**2

        N = 100
        x = jnp.linspace(0.,jnp.pi,N)

        y, s, s2 = scan_append_reduce(f, x, (True, False, False))
        mean = s/N
        var = s2/N - mean**2
    """
    # TODO: different op for each result

    x0 = jax.tree_map(lambda x: x[0], x)

    # special code path if there is only one element
    # to avoid having to rely on xla/llvm to optimize the overhead away
    if jax.tree_util.tree_leaves(x)[0].shape[0] == 1:
        return _multimap(
            lambda c, x: jnp.expand_dims(x, 0) if c else x, append_cond, f(x0)
        )

    # the original idea was to use pytrees, however for now just operate on the return value tuple
    _get_append_part = partial(_multimap, lambda c, x: x if c else None, append_cond)
    _get_op_part = partial(_multimap, lambda c, x: x if not c else None, append_cond)
    _tree_select = partial(_multimap, lambda c, t1, t2: t1 if c else t2, append_cond)

    carry_init = True, _get_op_part(zero_fun(jax.eval_shape(f, x0)))

    def f_(carry, x):
        is_first, y_carry = carry
        y = f(x)
        y_op = _get_op_part(y)
        y_append = _get_append_part(y)
        y_reduce = op(y_carry, y_op)
        return (False, y_reduce), y_append

    (_, res_op), res_append = jax.lax.scan(f_, carry_init, x, unroll=1)
    # reconstruct the result from the reduced and appended parts in the two trees
    return _tree_select(res_append, res_op)


scan_append = partial(scan_append_reduce, append_cond=True)
scan_reduce = partial(scan_append_reduce, append_cond=False)


# TODO in_axes a la vmap?
def scanmap(fun, scan_fun, argnums=0):
    """
    A helper function to wrap f with a scan_fun

    Example:
        import jax.numpy as jnp
        from functools import partial

        from netket.jax import scanmap, scan_append_reduce

        scan_fun = partial(scan_append_reduce, append_cond=(True, False, False))

        @partial(scanmap, scan_fun=scan_fun, argnums=1)
        def f(c, x):
             y = jnp.sin(x) + c
             return y, y, y**2

        N = 100
        x = jnp.linspace(0.,jnp.pi,N)
        c = 1.


        y, s, s2 = f(c, x)
        mean = s/N
        var = s2/N - mean**2
    """

    def f_(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(
            f, argnums, args, require_static_args_hashable=False
        )
        return scan_fun(lambda x: f_partial.call_wrapped(*x), dyn_args)

    return f_


def _treeify(f):
    def _f(x, *args, **kwargs):
        return jax.tree_map(lambda y: f(y, *args, **kwargs), x)

    return _f


@_treeify
def _unchunk(x):
    return x.reshape((-1,) + x.shape[2:])


@_treeify
def _chunk(x, chunk_size=None):
    # chunk_size=None -> add just a dummy chunk dimension, same as np.expand_dims(x, 0)
    n = x.shape[0]
    if chunk_size is None:
        chunk_size = n

    n_chunks, residual = divmod(n, chunk_size)
    if residual != 0:
        raise ValueError(
            "The first dimension of x must be divisible by chunk_size."
            + f"\n            Got x.shape={x.shape} but chunk_size={chunk_size}."
        )
    return x.reshape((n_chunks, chunk_size) + x.shape[1:])


def _chunk_size(x):
    b = set(map(lambda x: x.shape[:2], jax.tree_util.tree_leaves(x)))
    if len(b) != 1:
        raise ValueError(
            "The arrays in x have inconsistent chunk_size or number of chunks"
        )
    return b.pop()[1]


def unchunk(x_chunked):
    """
    Merge the first two axes of an array (or a pytree of arrays)
    Args:
        x_chunked: an array (or pytree of arrays) of at least 2 dimensions
    Returns: a pair (x, chunk_fn)
        where x is x_chunked reshaped to (-1,)+x.shape[2:]
        and chunk_fn is a function which restores x given x_chunked
    """
    return _unchunk(x_chunked), partial(_chunk, chunk_size=_chunk_size(x_chunked))



def chunk(x, chunk_size=None):
    """
    Split an array (or a pytree of arrays) into chunks along the first axis

    Args:
        x: an array (or pytree of arrays)
        chunk_size: an integer or None (default)
            The first axis in x must be a multiple of chunk_size
    Returns: a pair (x_chunked, unchunk_fn) where
        - x_chunked is x reshaped to (-1, chunk_size)+x.shape[1:]
          if chunk_size is None then it defaults to x.shape[0], i.e. just one chunk
        - unchunk_fn is a function which restores x given x_chunked
    """
    return _chunk(x, chunk_size), _unchunk



def _fun(vmapped_fun, chunk_size, argnums, *args, **kwargs):
    n_elements = jax.tree_util.tree_leaves(args[argnums[0]])[0].shape[0]
    n_chunks, n_rest = divmod(n_elements, chunk_size)

    if n_chunks == 0 or chunk_size >= n_elements:
        y = vmapped_fun(*args, **kwargs)
    else:
        # split inputs
        def _get_chunks(x):
            x_chunks = jax.tree_map(lambda x_: x_[: n_elements - n_rest, ...], x)
            x_chunks = _chunk(x_chunks, chunk_size)
            return x_chunks

        def _get_rest(x):
            x_rest = jax.tree_map(lambda x_: x_[n_elements - n_rest :, ...], x)
            return x_rest

        args_chunks = [
            _get_chunks(a) if i in argnums else a for i, a in enumerate(args)
        ]
        args_rest = [_get_rest(a) if i in argnums else a for i, a in enumerate(args)]

        y_chunks = _unchunk(
            scanmap(vmapped_fun, scan_append, argnums)(*args_chunks, **kwargs)
        )

        if n_rest == 0:
            y = y_chunks
        else:
            y_rest = vmapped_fun(*args_rest, **kwargs)
            y = jax.tree_map(lambda y1, y2: jnp.concatenate((y1, y2)), y_chunks, y_rest)
    return y


def _chunk_vmapped_function(
    vmapped_fun: Callable,
    chunk_size: Optional[int],
    argnums=0,
) -> Callable:
    """takes a vmapped function and computes it in chunks"""

    if chunk_size is None:
        return vmapped_fun

    if isinstance(argnums, int):
        argnums = (argnums,)

    # return HashablePartial(_fun, vmapped_fun, chunk_size, argnums)
    return partial(_fun, vmapped_fun, chunk_size, argnums)


def _parse_in_axes(in_axes):
    if isinstance(in_axes, int):
        in_axes = (in_axes,)

    if not set(in_axes).issubset((0, None)):
        raise NotImplementedError("Only in_axes 0/None are currently supported")

    argnums = tuple(
        map(lambda ix: ix[0], filter(lambda ix: ix[1] is not None, enumerate(in_axes)))
    )
    return in_axes, argnums


def apply_chunked(
    f: Callable,
    in_axes=0,
    *,
    chunk_size: Optional[int],
) -> Callable:
    """
    Takes an implicitly vmapped function over the axis 0 and uses scan to
    do the computations in smaller chunks over the 0-th axis of all input arguments.

    For this to work, the function `f` should be `vectorized` along the `in_axes`
    of the arguments. This means that the function `f` should respect the following
    condition:

    .. code-block:: python

        assert f(x) == jnp.concatenate([f(x_i) for x_i in x], axis=0)

    which is automatically satisfied if `f` is obtained by vmapping a function,
    such as:

    .. code-block:: python

        f = jax.vmap(f_orig)

    Args:
        f: A function that satisfies the condition above
        in_axes: The axes that should be scanned along. Only supports `0` or `None`
        chunk_size: The maximum size of the chunks to be used. If it is `None`, chunking
            is disabled

    """
    _, argnums = _parse_in_axes(in_axes)
    return _chunk_vmapped_function(f, chunk_size, argnums)



def vmap_chunked(
    f: Callable,
    in_axes=0,
    *,
    chunk_size: Optional[int],
) -> Callable:
    """
    Behaves like jax.vmap but uses scan to chunk the computations in smaller chunks.

    This function is essentially equivalent to:

    .. code-block:: python

        nk.jax.apply_chunked(jax.vmap(f, in_axes), in_axes, chunk_size)

    Some limitations to `in_axes` apply.

    Args:
        f: The function to be vectorised.
        in_axes: The axes that should be scanned along. Only supports `0` or `None`
        chunk_size: The maximum size of the chunks to be used. If it is `None`, chunking
            is disabled

    Returns:
        A vectorised and chunked function
    """
    in_axes, argnums = _parse_in_axes(in_axes)
    vmapped_fun = jax.vmap(f, in_axes=in_axes)
    return _chunk_vmapped_function(vmapped_fun, chunk_size, argnums)