from functools import partial
import os
import pickle as pkl
from collections.abc import MutableMapping

import numpy as np
import matplotlib.pyplot as plt
import tqdm

from scipy.interpolate import NearestNDInterpolator

import jax
import jax.numpy as jnp
import jax.experimental.host_callback as hcb
import flax
from flax import linen as nn
import optax

from . import deepxde as dde

from .utils import pairwise_dist


def generate_fixed_points_bc(comp, X, Y, seed=0):
    
    idxs = jax.random.permutation(key=jax.random.PRNGKey(seed), x=X.shape[0])
    X_shuff = X[idxs]
    Y_shuff = Y[idxs]
    
    def residue(params, net_apply, exp, inv, x_):
        i = min(X_shuff.shape[0], x_.shape[0])
        ypred = net_apply(params, X_shuff[:i])[:, comp]
        res = ypred.reshape(-1) - Y_shuff[:i].reshape(-1)
        # return jnp.mean(jnp.abs(res)).reshape(1)
        return res
    
    return residue


def generate_periodic_bc(comp, L_bound, R_bound):
    
    def residue(params, net_apply, exp, inv, xs):
        Xl = xs.at[:,0].set(L_bound)
        Xr = xs.at[:,0].set(R_bound)
        f = lambda x_: net_apply(params, x_)
        yleft = f(Xl)[:, comp:comp+1]
        yright = f(Xr)[:, comp:comp+1]
        return (yleft - yright).reshape(-1)
    
    return residue


def generate_fixed_function_bc(boundary_func, scale=1.):
    
    def residue(params, net_apply, exp, inv, xs):
        ys_pred = net_apply(params, xs)
        ys_actual = boundary_func(xs, exp, inv)
        return scale * (ys_pred - ys_actual).reshape(-1)
    
    return residue


def generate_velocity_bc(boundary_func, i=0, j=None, scale=1.):
    
    def residue(params, net_apply, exp, inv, xs):
        ys_pred = net_apply(params, xs)
        ys_pred_fn = lambda x_: net_apply(params, x_)
        deriv = dde.grad.jacobian((ys_pred, ys_pred_fn), xs, i=i, j=j)[0]
        deriv_actual = boundary_func(xs, exp, inv)
        return scale * (deriv - deriv_actual).reshape(-1)
    
    return residue


def generate_arbitrary_bc(residue_fn):
    
    def residue(params, net_apply, exp, inv, xs):
        ys = net_apply(params, xs)
        res = residue_fn(xs, ys, exp, inv)
        return res.reshape(-1)
    
    return residue


def get_corresponding_y(xs, x_pool, y_pool):
    pw = pairwise_dist(xs, x_pool)
    vect = jax.lax.stop_gradient(jax.nn.one_hot(jnp.argmin(pw, axis=1), num_classes=x_pool.shape[0]))
    return jax.lax.stop_gradient(vect @ y_pool)
    
    
def generate_residue(bc, net_apply, return_output_for_pointset=False):
    
    if isinstance(bc, dde.icbc.boundary_conditions.PeriodicBC):
        
        def residue(params, xs):
            Xl = xs.at[:,0].set(bc.geom.geometry.l)
            Xr = xs.at[:,0].set(bc.geom.geometry.r)
            f = lambda x_: net_apply(params, x_, training=False)
            if bc.derivative_order == 0:
                yleft = f(Xl)[:, bc.component : bc.component + 1]
                yright = f(Xr)[:, bc.component : bc.component + 1]
            else:
                yleft = dde.grad.jacobian((f(Xl), f), Xl, i=bc.component, j=bc.component_x)
                yright = dde.grad.jacobian((f(Xr), f), Xr, i=bc.component, j=bc.component_x)
            return (yleft - yright).reshape(-1)
        
    elif isinstance(bc, dde.icbc.boundary_conditions.NeumannBC):
        
        def b_fn(xs):
            return bc.boundary_normal(xs, 0, xs.shape[0], None)
        
        def residue(params, xs):
            values = bc.func(xs, 0, xs.shape[0], None)
            f_ = lambda x_: net_apply(params, x_, training=False)
            ys = f_(xs)
            dydx = dde.grad.jacobian((ys, f_), xs, i=bc.component, j=None)[0]
            # n = jax.lax.stop_gradient(boundary_normal_jax(xs))
            # n = jax.lax.stop_gradient(hcb.call(b_fn, xs, result_shape=xs))
            n = jax.lax.stop_gradient(jax.pure_callback(b_fn, jax.ShapeDtypeStruct(xs.shape, xs.dtype), xs))
            y = dde.backend.sum(dydx * n, 1, keepdims=True)
            return (y - values).reshape(-1)
        
    elif isinstance(bc, dde.icbc.boundary_conditions.PointSetBC):
        
        if return_output_for_pointset:
            
            def residue(params, xs):
                y_pred = net_apply(params, xs, training=False)[:, bc.component]
                return y_pred.reshape(-1)
            
        else:
            
            x_pool = np.array(bc.points)
            y_pool = np.array(bc.values[:, bc.component])
            interp = NearestNDInterpolator(x_pool, y_pool)

            @jax.jit
            def get_y_fn(xs):
                return hcb.call(interp, xs, result_shape=xs[:,0])
            
            def residue(params, xs):
                y_pred = net_apply(params, xs, training=False)[:, bc.component].reshape(-1)
                ys = jax.lax.stop_gradient(get_y_fn(xs)).reshape(-1)
                return y_pred - ys
        
    else:
    
        def residue(params, xs):
            return bc.error(
                X=xs,
                inputs=xs, 
                outputs=net_apply(params, xs, training=False), 
                beg=0, 
                end=xs.shape[0], 
                aux_var=None
            ).reshape(-1)
            
    return residue
    
# ======================================================================


def value_clip(index_arr, lb, ub):
    # Clip the values of the array index_arr to the range [lb, ub]
    return index_arr.set(jnp.clip(index_arr.get(), lb, ub))


def constrain_timedomain(points: jax.Array, timedomain: dde.geometry.TimeDomain, on_boundary: bool = False):
    # Constrain the time domain of the points. points is a jax.Array of shape (n_points, n_dims)
    if on_boundary:
        points = value_clip(points.at[:,-1], timedomain.t0, timedomain.t0)
    else:
        points = value_clip(points.at[:,-1], timedomain.t0, timedomain.t1)
    return points


def constrain_interval(points: jax.Array, geom: dde.geometry.Interval, on_boundary: bool = False):
    l, r = geom.l, geom.r
    if on_boundary:
        N = points.shape[0] // 2
        points = value_clip(points.at[:N,0], l, l)
        points = value_clip(points.at[N:,0], r, r)
    else:
        points = value_clip(points.at[:,0], l, r)
    return points


def constrain_rectangle(points: jax.Array, geom: dde.geometry.Interval, on_boundary: bool = False):
    (x0l, x1l), (x0r, x1r) = geom.bbox
    if on_boundary:
        Ns = jnp.linspace(0, points.shape[0], 5, dtype=int)
        points = value_clip(points.at[Ns[0]:Ns[1],0], x0l, x0r)
        points = value_clip(points.at[Ns[0]:Ns[1],1], x1l, x1l)
        points = value_clip(points.at[Ns[1]:Ns[2],0], x0l, x0r)
        points = value_clip(points.at[Ns[1]:Ns[2],1], x1r, x1r)
        points = value_clip(points.at[Ns[2]:Ns[3],0], x0l, x0l)
        points = value_clip(points.at[Ns[2]:Ns[3],1], x1l, x1r)
        points = value_clip(points.at[Ns[3]:Ns[4],0], x0r, x0r)
        points = value_clip(points.at[Ns[3]:Ns[4],1], x1l, x1r)
    else:
        points = value_clip(points.at[:,0], x0l, x0r)
        points = value_clip(points.at[:,1], x1l, x1r)
    return points


def constrain_geometry(points: jax.Array, geom: dde.geometry.Interval, on_boundary: bool = False):
    if isinstance(geom, dde.geometry.geometry_1d.Interval):
        return constrain_interval(points=points, geom=geom, on_boundary=on_boundary)
    if isinstance(geom, dde.geometry.geometry_2d.Rectangle):
        return constrain_rectangle(points=points, geom=geom, on_boundary=on_boundary)
    else:
        raise ValueError(f'{type(geom)} type objects not implemented yet')
    

# ===============================================================


def constrain_domain(points: jax.Array, geom: dde.geometry.Interval, timedomain: dde.geometry.TimeDomain = None):
    points = constrain_geometry(points, geom=geom, on_boundary=False)
    if timedomain is not None:
        points = constrain_timedomain(points, timedomain=timedomain, on_boundary=False)
    return points


def constrain_ic(points: jax.Array, geom: dde.geometry.Interval, timedomain: dde.geometry.TimeDomain):
    points = constrain_geometry(points, geom=geom, on_boundary=False)
    points = constrain_timedomain(points, timedomain=timedomain, on_boundary=True)
    return points


def constrain_bc(points: jax.Array, geom: dde.geometry.Interval, timedomain: dde.geometry.TimeDomain = None):
    points = constrain_geometry(points, geom=geom, on_boundary=True)
    if timedomain is not None:
        points = constrain_timedomain(points, timedomain=timedomain, on_boundary=False)
    return points
        