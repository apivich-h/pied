from functools import partial
import os
import pickle as pkl
from collections.abc import MutableMapping
from typing import Dict, Any, Callable, List
import time
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

from ..models.model_loader import construct_net
from ..utils import to_cpu
from ..icbc_patch import generate_residue

# logger for this file
logging.getLogger().setLevel(logging.INFO)


class PINNBasedSampler:
    
    def __init__(self, pde: Callable, fixed_bcs: List, pde_domain, exp_design_fn: Callable, 
                 noise_kernel: str = 'rbf', noise_kernel_args: Dict = None,
                 nn_params: Dict = None, collocation_pts: int = 10000, 
                 optim_method: str = 'adam', optim_args: Dict = None, optim_steps: int = 50000, loss_thr: float = 0.):
        
        super().__init__()
        # self.pde_class = pde_class  # takes in beta, returns (pde, [list of bcs])
        self.pde = pde
        self.fixed_bcs = fixed_bcs
        self.exp_design_fn = exp_design_fn  # takes in experimental design params Dict, returns [list of bcs]
        
        self.pde_domain = pde_domain
        self.pde_collocation_pts = jnp.array(self.pde_domain.random_points(int(0.8 * collocation_pts), random='Hammersley'))
        self.ic_pts = jnp.array(self.pde_domain.random_initial_points(collocation_pts // 10))
        self.bc_pts = jnp.array(self.pde_domain.random_boundary_points(collocation_pts // 10))
                
        # if (noise_kernel is None) or ((noise_kernel == 'rbf') and (noise_kernel_args is None)):
        #     self.noise_kernel_fn = gpx.kernels.RBF(variance=0.01, lengthscale=0.01)
        # else:
        #     if noise_kernel == 'rbf':
        #         self.noise_kernel_fn = gpx.kernels.RBF(**noise_kernel_args)
        #     elif noise_kernel == 'matern52':
        #         self.noise_kernel_fn = gpx.kernels.matern52(**noise_kernel_args)
        #     else:
        #         raise ValueError(f'Invalid {noise_kernel}.')
        # self.noise_prior = gpx.Prior(mean_function=gpx.mean_functions.Zero(), kernel=self.noise_kernel_fn)
        
        self.optim_steps = optim_steps
        if optim_args is None:
            self.optim_method = 'adam'
            self.optim_args = dict(learning_rate=0.001)
        else:
            self.optim_method = optim_method
            self.optim_args = optim_args
        self.loss_thr = loss_thr
        
        self.nn_params = nn_params
        self.net = construct_net(**self.nn_params)[0]
        self.rng = jax.random.PRNGKey(np.random.randint(1000000))
        self.rng, subkey = jax.random.split(self.rng)
        self.net_params = self.net.init(subkey, self.pde_collocation_pts[:1])
        
        self._icbcs = None
        self._icbcs_data = None
        self._losses_steps = []
        self._losses = []
        self.inv_params = None
        self.exp_params = None
    
    def set_inv_params(self, inv_params):
        self.inv_params = inv_params
    
    def set_exp_params(self, exp_params):
        self.exp_params = exp_params
        
    def _generate_solver(self, value_and_grad):
        if self.optim_method == 'adam':
            opt = optax.adam(**self.optim_args)
            solver = jaxopt.OptaxSolver(opt=opt, fun=value_and_grad, value_and_grad=True)
        elif self.optim_method == 'lbfgs':
            solver = jaxopt.LBFGS(fun=value_and_grad, value_and_grad=True, jit=True, **self.optim_args)
        else:
            raise ValueError(f'Invalid optim_method: {self.optim_method}')
        return solver
    
    def generate_pinn_loss_fn(self):
        
        assert self.inv_params is not None
        assert self.exp_params is not None
        
        net = self.net
        
        # pde_fn, fixed_bcs = self.pde_class(b)
        design_bcs = self.exp_design_fn(self.exp_params)
        
        def _pde_residue_fn(params, xs):
            f_ = lambda xs: net.apply(params, xs, training=True)
            return self.pde(xs, (f_(xs), f_), self.inv_params)[0]
        
        bcs = self.fixed_bcs + design_bcs
        bc_fns = [generate_residue(bc, net_apply=net.apply) for bc in bcs]
        bc_data = []
        for bc in bcs:
            if isinstance(bc, dde.icbc.initial_conditions.IC):
                xs = self.ic_pts
            else:
                xs = self.bc_pts
            bc_data.append(xs)
        self._icbcs = bcs
        self._icbcs_data = bc_data
        
        def new_loss(params):
            loss = jnp.mean(_pde_residue_fn(params, self.pde_collocation_pts) ** 2)
            for bc_fn, bc_xs in zip(bc_fns, bc_data):
                loss += jnp.mean(bc_fn(params, bc_xs) ** 2)
            return loss
        
        return new_loss

    def prep_simulator(self):
        
        assert self.inv_params is not None
        assert self.exp_params is not None
        
        new_loss = self.generate_pinn_loss_fn()
        
        params = self.net_params
        solver = self._generate_solver(value_and_grad=jax.jit(jax.value_and_grad(new_loss)))
        opt_state = solver.init_state(params)
        
        self._losses_steps = [0]
        self._losses = [new_loss(params)]
        factor = 1000
        for s in range(self.optim_steps // factor):
            for _ in range(factor):
                params, opt_state = solver.update(params, opt_state)
            self.net_params = params
            l = new_loss(params)
            self._losses.append(l)
            self._losses_steps.append(factor * (s+1))
            if l < self.loss_thr:
                break
                
    def log_likelihood(self, xs, ys):
        mu = self.net.apply(self.net_params, xs)
        d = ys - mu
        sigma = self.noise_kernel_fn.gram(xs).matrix
        sigma_inv = jnp.linalg.inv(sigma)
        return - 0.5 * jnp.linalg.slogdet(sigma)[1] - (0.5 * (d.T @ sigma_inv @ d))[0,0]
    
    def sample(self, xs, rng):
        return self.net.apply(self.net_params, xs) # + self._noise_function(xs, rng)
    
    # def _noise_function(self, xs, rng):
    #     prior_dist = self.noise_prior.predict(xs)
    #     return prior_dist.sample(seed=rng, sample_shape=(1,)).reshape(-1, 1)
    
    def generate_intermediate_info(self):
        return {
            'inv_params': self.inv_params,
            'exp_params': self.exp_params,
            'net_params': self.net_params,
        }
