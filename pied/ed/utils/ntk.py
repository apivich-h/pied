from functools import partial
import os
from collections.abc import MutableMapping

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import flax

from ... import deepxde as dde

from ...icbc_patch import generate_residue


def _flatten_dict(d, parent_key='', sep='_'):
    # https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
    # needed as neural network parameters are stored in a nested dictionary, e.g. {'params': {'dense': {'kernel': ...}}}
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping) or isinstance(v, flax.core.frozen_dict.FrozenDict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

@partial(jax.jit, static_argnames=['fn'])
def _jac_params_helper(params, x, fn):
    fn2 = lambda params, x_: fn(params, x_.reshape(1, -1))[0]  # version for single dims
    # print(fn2(params, x).shape)
    f_ = lambda x_: jax.jacobian(fun=fn2, has_aux=False)(params, x_)
    dd = jax.vmap(f_)(x)
    # dd = jax.jit(jax.vmap(jax.grad(fn2), in_axes=(None, 0)))(params, x)
    return dd


def _jac_params_cleanup(dd):
    # dd = _flatten_dict(dd['params'])
    # currently only works for one-dimensional model outputs
    # return {k: dd[k].reshape(dd[k].shape[0], -1) for k in dd.keys()}
    dd = jtu.tree_flatten(dd)[0]
    dd = [jax.vmap(jnp.ravel)(d) for d in dd]
    return dd


def get_ntk_from_jac(jac1, jac2):
    prods = [j1 @ j2.T for (j1, j2) in zip(jac1, jac2)]
    return sum(prods)


def concat_list_of_arrays(list_of_arrays):
    return [jnp.vstack(arrs) for arrs in zip(*list_of_arrays)]


class NTKHelper:
    
    def __init__(self, net, pde, bcs, inverse_problem: bool = False):
        self.net = net
        self.pde = pde
        self.bcs = bcs
        self.inverse_problem = inverse_problem
        self.bc_fns = [generate_residue(bc, self.net.apply, return_output_for_pointset=True) for bc in self.bcs]
        self._output_fn = lambda params, xs: self.net.apply(params, xs, training=True)

    def _get_output_jac(self, xs, params):
        fun = lambda params, x: self._output_fn(params['net'], x)[:, 0]
        d = _jac_params_helper(params=params, x=xs, fn=fun)
        return _jac_params_cleanup(d)
        
    def _get_pde_jac(self, xs, params, forward_inv_param=None):
        
        def f2_(params, x):
            f_ = lambda x: self.net.apply(params['net'], x)
            if self.inverse_problem:
                return self.pde(x, (f_(x), f_), params['inv'])[0]
            else:
                return self.pde(x, (f_(x), f_), forward_inv_param)[0]
        
        d = _jac_params_helper(params=params, x=xs, fn=f2_)
        return _jac_params_cleanup(d)
    
    # Adding loss_w_bcs to introduce loss weights
    def _get_bc_jac(self, bc_idx, xs, params):
        d = _jac_params_helper(
            params=params, x=xs, 
            fn=lambda param, x_: self.bc_fns[bc_idx](param['net'], x_)
        )
        return _jac_params_cleanup(d)
    
    def get_jac_fn(self, code):
        if code == -2:
            # derivative wrt output only
            return lambda params, xs: self._get_output_jac(xs=xs, params=params)
        elif code == -1:
            # derivative wrt PDE residue
            return lambda params, xs: self._get_pde_jac(xs=xs, params=params)
        else:
            # derivative wrt BC error term
            assert 0 <= code < len(self.bcs)
            return lambda params, xs: self._get_bc_jac(xs=xs, params=params, bc_idx=code)
    
    def get_combined_jac(self, params, xs_anc=None, xs_pde=None, xs_bcs=None, forward_inv_param=None):
        
        jacs_list = []
        
        if xs_anc is not None:
            jacs_list.append(self.get_jac_fn(-2)(params=params, xs=xs_anc))
            
        if xs_pde is not None:
            jacs_list.append(self.get_jac_fn(-1)(params=params, xs=xs_pde))
            
        if xs_bcs is not None:
            for i, xs in enumerate(xs_bcs):
                jacs_list.append(self.get_jac_fn(i)(params=params, xs=xs))
                
        if len(jacs_list) == 0:
            raise ValueError
        elif len(jacs_list) == 1:
            return jacs_list
        else:
            return concat_list_of_arrays(jacs_list)
    
    def get_ntk_fn(self):
        return get_ntk_from_jac
    
    def get_ntk(self, jac1, jac2):
        return get_ntk_from_jac(jac1=jac1, jac2=jac2)
    
    def get_residue_fn(self, code, anc_idx=0, anc_model_param=None, forward_inv_param=None):
        
        if code == -2:
            # derivative wrt output only
            def fn(params, xs):
                ys_ref = self._output_fn(anc_model_param['net'], xs)[:, anc_idx]
                ys_pred = self._output_fn(params['net'], xs)[:, anc_idx]
                return (ys_pred - ys_ref).reshape(-1, 1)
            return fn
        
        elif code == -1:
            # derivative wrt PDE residue
            if self.inverse_problem:
                def fn(params, xs):
                    f_ = lambda x: self.net.apply(params['net'], x)
                    return self.pde(xs, (f_(xs), f_), params['inv'])[0].reshape(-1, 1)
            else:
                def fn(params, xs):
                    f_ = lambda x: self.net.apply(params['net'], x)
                    return self.pde(xs, (f_(xs), f_), forward_inv_param)[0].reshape(-1, 1)
            return fn
        
        else:
            # derivative wrt BC error term
            assert 0 <= code < len(self.bcs)
            return lambda params, xs: self.bc_fns[code](params['net'], xs).reshape(-1, 1)
        
    def get_combined_res(self, params, xs_anc=None, xs_pde=None, xs_bcs=None, anc_idx=0, anc_model_param=None, forward_inv_param=None):
        
        res_list = []
        
        if xs_anc is not None:
            res_list.append(self.get_residue_fn(-2, anc_idx=anc_idx)(params=params, xs=xs_anc))
            
        if xs_pde is not None:
            res_list.append(self.get_residue_fn(-1, forward_inv_param=forward_inv_param)(params=params, xs=xs_pde))
            
        if xs_bcs is not None:
            for i, xs in enumerate(xs_bcs):
                res_list.append(self.get_residue_fn(i)(params=params, xs=xs))
                
        if len(res_list) == 0:
            raise ValueError
        elif len(res_list) == 1:
            return res_list
        else:
            return jnp.concatenate(res_list)
