from functools import partial
import os
import pickle as pkl
from collections.abc import MutableMapping
from typing import Dict, Any, Callable, List
import logging

import tqdm

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.experimental.jax2tf import call_tf
import flax
from flax import linen as nn
import optax
import jaxopt

# import gpjax as gpx

from .. import deepxde as dde

from .model_loader import construct_net
from ..utils import to_cpu, tree_stack, tree_unstack
from ..icbc_patch import generate_residue
from ..pde_sampler import PDESampler

# logger for this file
logging.getLogger().setLevel(logging.INFO)


class PINNEnsemble:
    
    def __init__(self, pde: Callable, pde_domain, exp_design_fn, obs_design_fn, inv_embedding: Callable = None, 
                 input_transform_generator_fn: Callable = None, output_transform_generator_fn: Callable = None,
                 n_pde_collocation_pts: int = 10000, n_icbc_collocation_pts: int = 10000, pde_colloc_rand_method: str = 'Hammersley',
                 nn_construct_params: Dict = dict(), inv_problem: bool = False, maxiter: int = 10000, implicit_diff: bool = False,
                 optim_method: str = 'adam', optim_args: Dict = None, loss_thr: float = 0., loss_every: int = 1000, rng=jax.random.PRNGKey(42)):
        
        super().__init__()
        self.pde = pde
        self.exp_design_fn = exp_design_fn
        self.obs_design_fn = obs_design_fn
        self.inv_embedding = inv_embedding
        self.inv_problem = inv_problem
        
        self.pde_domain = pde_domain
        self.icbc_point_count = n_icbc_collocation_pts
        self.pde_point_count = n_pde_collocation_pts
        self.pde_colloc_rand_method = pde_colloc_rand_method
        
        if optim_args is None:
            self.optim_method = 'adam'
            self.optim_args = dict(learning_rate=0.001)
        else:
            self.optim_method = optim_method
            self.optim_args = optim_args
        self.loss_thr = loss_thr
        self.implicit_diff = implicit_diff
        self.maxiter = maxiter
        self.loss_every = loss_every
        
        self.nn_construct_params = nn_construct_params
        self.net = construct_net(**self.nn_construct_params)[0]
        self.input_transform_generator_fn = input_transform_generator_fn
        self.output_transform_generator_fn = output_transform_generator_fn
        self.rng = rng
        
        self.reset()
        
    def reset(self):
        self.exp_params = None
        self.inv_params = None
        self.inv_params_guesses = None
        self.params = None
        self.pde_collocation_pts = jnp.array(
            self.pde_domain.random_points(1, random=self.pde_colloc_rand_method),
            # dtype=jnp.zeros(1).dtype,
        )
        self.icbc_fns = None
        self.icbc_points = None
        self.steps = 0
        self.solver = None
        self.opt_state_list = None
        self.step_batch = None
        self.loss_batch = None
        self.loss_single = None
        self._param_history = []
        self._losses_steps = []
        self._losses = []
        self._losses_pde = []
        self._losses_icbc = []
                
    def _generate_solver(self, value_and_grad, unroll="auto"):
        if self.optim_method == 'adam':
            opt = optax.adam(**self.optim_args)
            solver = jaxopt.OptaxSolver(opt=opt, fun=value_and_grad, value_and_grad=True, maxiter=self.maxiter, implicit_diff=self.implicit_diff, unroll=unroll)
        elif self.optim_method == 'lbfgs':
            solver = jaxopt.LBFGS(fun=value_and_grad, value_and_grad=True, jit=True, maxiter=self.maxiter, implicit_diff=self.implicit_diff, unroll=unroll, **self.optim_args)
        else:
            raise ValueError(f'Invalid optim_method: {self.optim_method}')
        return solver

    def _generate_loss_function(self):
        
        pde_colloc_pts = jnp.array(
            self.pde_domain.random_points(self.pde_point_count, random=self.pde_colloc_rand_method),
            # dtype=jnp.zeros(1).dtype,
        )
        self.pde_collocation_pts = pde_colloc_pts
        
        bc_fns = [fn for (fn, _) in self.exp_design_fn]
        self.rng, key_ = jax.random.split(self.rng)
        bc_data = [
            jnp.array(xs[:self.icbc_point_count])[jax.random.permutation(k, xs.shape[0])[:self.icbc_point_count]]
            for (k, (_, xs)) in zip(jax.random.split(key_, num=len(self.exp_design_fn)), self.exp_design_fn)
        ]
        self.icbc_fns = bc_fns
        self.icbc_points = bc_data
        
        def _pde_residue_fn(params, xs, inv_param):
            f_ = lambda xs: self.net.apply(params, xs, training=True)
            return self.pde(xs, (f_(xs), f_), inv_param, self.exp_params)[0]
        
        self._pde_residue_fn = _pde_residue_fn
        
        if self.inv_problem:
                
            def new_loss(params, obs_design, observation, prior_inv):
                net_params = params['net']
                inv_params = params['inv']
                apply_fn = self.net.apply
                loss = 0.
                loss += jnp.mean(_pde_residue_fn(net_params, pde_colloc_pts, inv_params) ** 2)
                for bc_fn, bc_xs in zip(bc_fns, bc_data):
                    # loss += jnp.mean(bc_fn(net_params, bc_xs) ** 2)
                    loss += jnp.mean(bc_fn(net_params, apply_fn, self.exp_params, inv_params, bc_xs) ** 2)
                # y_anc_pred = apply_fn(net_params, x_anc)
                # assert y_anc_pred.shape == y_anc.shape, (y_anc_pred.shape, y_anc.shape)
                # loss += jnp.mean((y_anc_pred - y_anc) ** 2)
                f = lambda xs: apply_fn(net_params, xs)
                obs = self.obs_design_fn(f, obs_design)
                assert obs.shape == observation.shape, (obs.shape, observation.shape)
                loss += jnp.mean((obs - observation) ** 2)
                # if prior_reg > 0.:
                #     loss += prior_reg * jnp.mean((self.inv_embedding(inv_params) - self.inv_embedding(prior_inv)) ** 2)
                return loss
            
        else:
            
            def new_loss(params, inv):
                net_params = params['net']
                apply_fn = self.net.apply
                loss = jnp.mean(_pde_residue_fn(net_params, pde_colloc_pts, inv) ** 2)
                for bc_fn, bc_xs in zip(bc_fns, bc_data):
                    # loss += jnp.mean(bc_fn(net_params, bc_xs) ** 2)
                    loss += jnp.mean(bc_fn(net_params, apply_fn, self.exp_params, inv, bc_xs) ** 2)
                return loss
            
        return new_loss

    def prep_simulator(self, exp_params, inv_params=None, inv_params_guesses=None, new_nn_params=None, prior_reg=0., unroll="auto"):
        
        self.exp_params = exp_params
        self.inv_params = inv_params
        self.inv_params_guesses = inv_params_guesses
        
        assert self.exp_params is not None
        
        if self.input_transform_generator_fn is not None:
            assert self.nn_construct_params.get('arch', None) != 'fourier'
            self.net._input_transform = jax.jit(self.input_transform_generator_fn(exp_params))
        
        if self.output_transform_generator_fn is not None:
            self.net._output_transform = jax.jit(self.output_transform_generator_fn(exp_params))
            
        new_loss = self._generate_loss_function()
        self.loss_single = new_loss
        self.solver = self._generate_solver(value_and_grad=jax.jit(jax.value_and_grad(new_loss)), unroll=unroll)
        self.step_batch = jax.jit(jax.vmap(self.solver.update))
        self.loss_batch = jax.jit(jax.vmap(new_loss))
            
        if new_nn_params is not None:
            self.params = new_nn_params
        
        if self.inv_problem:
            
            if self.params is None:
                net_params = []
                for i in range(len(self.inv_params_guesses)):
                    self.rng, key_ = jax.random.split(self.rng)
                    params = self.net.init(key_, self.pde_collocation_pts[:1])
                    inv_params = self.inv_params_guesses[i]
                    net_params.append({'net': params, 'inv': inv_params})
                self.params = tree_stack(net_params)
            
        else:
            
            self.inv_params = jnp.array(self.inv_params)
               
            if self.params is None: 
                net_params = []
                for _ in range(len(self.inv_params)):
                    self.rng, key_ = jax.random.split(self.rng)
                    params = self.net.init(key_, self.pde_collocation_pts[:1])
                    net_params.append({'net': params})
                self.params = tree_stack(net_params)
                
        self.opt_state_list = jax.vmap(self.solver.init_state)(self.params)
        
    def step_opt(self, obs_design=None, observation=None):
        
        if self.steps == 0:
            
            if self.inv_problem:
                l = self.loss_batch(self.params, obs_design, observation, self.inv_params_guesses)
            else:
                l = self.loss_batch(self.params, self.inv_params)
            self._losses_steps.append(self.steps)
            self._losses.append(l)
            self._param_history.append(self.params)
            
            inv = self.params['inv'] if self.inv_problem else self.inv_params
            icbc_loss = []
            for bc_fn, bc_xs in zip(self.icbc_fns, self.icbc_points):
                # loss += jnp.mean(bc_fn(net_params, bc_xs) ** 2)
                icbc_loss.append(
                    jax.vmap(bc_fn, in_axes=(0, None, None, 0, None))(
                        self.params['net'], self.net.apply, self.exp_params, inv, bc_xs)
                )
            self._losses_icbc.append(icbc_loss)
            self._losses_pde.append(
                jax.vmap(self._pde_residue_fn, in_axes=(0, None, 0))(
                    self.params['net'], self.pde_collocation_pts, inv
                ))
        
        if self.inv_problem:
            self.params, self.opt_state_list = self.step_batch(self.params, self.opt_state_list, obs_design, observation, self.inv_params_guesses)
        else:
            self.params, self.opt_state_list = self.step_batch(self.params, self.opt_state_list, self.inv_params)
            
        self.steps += 1
        if self.steps % self.loss_every == 0:
            
            if self.inv_problem:
                l = self.loss_batch(self.params, obs_design, observation, self.inv_params_guesses)
            else:
                l = self.loss_batch(self.params, self.inv_params)
            self._losses_steps.append(self.steps)
            self._losses.append(l)
            self._param_history.append(self.params)
            
            inv = self.params['inv'] if self.inv_problem else self.inv_params
            icbc_loss = []
            for bc_fn, bc_xs in zip(self.icbc_fns, self.icbc_points):
                # loss += jnp.mean(bc_fn(net_params, bc_xs) ** 2)
                icbc_loss.append(
                    jax.vmap(bc_fn, in_axes=(0, None, None, 0, None))(
                        self.params['net'], self.net.apply, self.exp_params, inv, bc_xs)
                )
            self._losses_icbc.append(icbc_loss)
            self._losses_pde.append(
                jax.vmap(self._pde_residue_fn, in_axes=(0, None, 0))(
                    self.params['net'], self.pde_collocation_pts, inv
            ))
            
    def convert_nn_params_to_type(self, dtype=jnp.float64):
        self.params = jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=dtype), self.params)
            
    def generate_pred_function(self, i=None):
        if i is None:
            return lambda xs: jax.vmap(self.net.apply, in_axes=(0, None))(self.params['net'], xs)
        else:
            p = tree_unstack(self.params)[i]['net']
            return lambda xs: self.net.apply(p, xs)
        
    def get_param(self, i):
        return tree_unstack(self.params)[i]
    
    def generate_intermediate_info(self):
        return {
            'inv_params': self.inv_params,
            'exp_params': self.exp_params,
            'net_params': self.params,
        }
