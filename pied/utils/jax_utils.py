from functools import partial
from .. import deepxde as dde
import jax.numpy as jnp
import numpy as np
import jax


# https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
def vmap_mjp(f, x, M):
    y, vjp_fun = jax.vjp(f, x)
    outs, = jax.vmap(vjp_fun)(M)
    return outs

def vmap_jmp(f, W, M):
    _jvp = lambda s: jax.jvp(f, (W,), (s,))[1]
    return jax.vmap(_jvp)(M)


def jacobian_outer_product(f, x1, x2, M):
    f2 = lambda v: jax.jvp(f, [x1], [v])[1]
    J1J2T_op = lambda v: jax.vjp(f, x2)[1](f2(v))[0]
    return jax.vmap(J1J2T_op)(jnp.eye(M))



@partial(jax.jit, device=jax.devices("cpu")[0])
def to_cpu(x):
    return x

def pairwise_dist(A, B):
    C = A[:, jnp.newaxis, :] - B[jnp.newaxis, :, :]
    C = jnp.linalg.norm(C, axis=-1)
    return C

# TODO to refactor
def get_pde_residue(model: dde.Model, xs):
    f_ = lambda xs: model.net.apply(model.params[0], xs, training=True)
    return model.data.pde(xs, (f_(xs), f_))[0]

# For printing out terrible's formated dict list structures
def print_dict_structure(dictionary, level=0):
    if isinstance(dictionary,dict):
        for key, value in dictionary.items():
            print('  ' * level + f'{key}: {len(value)} items')
            if isinstance(value, dict):
                print_dict_structure(value, level + 1)
            # if the element is an array, return the shape of the array
            elif isinstance(value, (jnp.ndarray,np.ndarray)):
                print('  ' * (level + 1) + f'array of shape: {value.shape}')
            # if the element is a list, return the length of the list
            elif isinstance(value, (list,tuple)):
                print('  ' * (level + 1) + f'{type(value)} of length: {len(value)}')
                print_dict_structure(value, level + 2)

    elif isinstance(dictionary, (list,tuple)):
        # Do the same for the list object 
        for i in range(len(dictionary)):
            value = dictionary[i]
            print('  ' * level + f'element {i}:')
            if isinstance(value, dict):
                print_dict_structure(value, level + 1)
            # if the element is an array, return the shape of the array
            elif isinstance(value, (jnp.ndarray,np.ndarray)):
                print('  ' * (level + 1) + f'array of shape: {value.shape}')
            # if the element is a list, return the length of the list
            elif isinstance(value, list):
                print('  ' * (level + 1) + f'list of length: {len(value)}')
                print_dict_structure(value[i], level + 1)
    elif isinstance(dictionary, (jnp.ndarray,np.ndarray)):
        print('  ' * (level) + f'array of shape: {dictionary.shape}')
        
        
def flatten(v):
    def f(v):
        leaves, _ = jax.tree_util.tree_flatten(v)
        return jnp.concatenate([x.ravel() for x in leaves])
    out, pullback = jax.vjp(f, v)
    return out, lambda x: pullback(x)[0]


# Flatten the input points dictionary into an ndarray
def flatten_pts_dict(d):
    flattened = d['res']
    for i in range(len(d['bcs'])):
        flattened = jnp.vstack((flattened,d['bcs'][i]))
    return flattened

# Size of the flattened input points dictionary
def dict_pts_size(d):
    size = 0
    for key, value in d.items():
        if isinstance(value,(jnp.ndarray)) or isinstance(value,(np.ndarray)):
            size += len(value)
               
        if isinstance(value,list):
            for i in range(len(value)):
                size += len(value[i])
    return size


# sample from uniform distribution
def sample_from_uniform(n, bounds, sample_dim, rng=jax.random.PRNGKey(0)):
    # return bounds[None,:,0] + jax.random.uniform(key=rng, shape=(n, sample_dim)) * (bounds[None,:,1] - bounds[None,:,0])
    return jax.random.uniform(key=rng, shape=(n, sample_dim), minval=bounds[:,0], maxval=bounds[:,1])

# https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75?permalink_comment_id=4634557#gistcomment-4634557
def tree_stack(trees):
    return jax.tree_util.tree_map(lambda *v: jnp.stack(v), *trees)


# https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75?permalink_comment_id=4634557#gistcomment-4634557
def tree_unstack(tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves)]