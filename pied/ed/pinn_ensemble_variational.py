from functools import partial
import os
import pickle as pkl
from collections.abc import MutableMapping
from datetime import datetime
from itertools import product
from functools import partial
from typing import Dict, Callable
import time
import logging

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import tqdm

import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit, hessian, lax
from jax.scipy.special import logsumexp
from jax.example_libraries import optimizers
from jax.nn import relu
from jax.config import config
from jax.flatten_util import ravel_pytree
import optax
import jaxopt
from jax.nn import softmax
from jax.scipy.stats.multivariate_normal import logpdf as norm_logpdf

import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound, qSimpleRegret
from botorch.optim.initializers import gen_batch_initial_conditions, initialize_q_batch_nonneg
from botorch.generation import gen_candidates_torch, get_best_candidates
from botorch.sampling.stochastic_samplers import StochasticSampler

from ..icbc_patch import generate_residue
from ..models.pinn_ensemble import PINNEnsemble
from ..utils import sample_from_uniform
from ..utils.vmap_chunked import vmap_chunked
# from .ed_loop import ExperimentalDesign
from .criterion_based import CriterionBasedAbstractMethod
from .eig_estimators.losses import generate_loss
from .utils.obs_fn_helper import get_vmap_oracle

# logger for this file
logging.getLogger().setLevel(logging.INFO)


class PINNEnsembleWithVBOEDMethod(CriterionBasedAbstractMethod):
    
    def __init__(self, simulator_xs, pde, pde_domain, exp_design_fn, obs_design_fn,
                 inv_embedding, inv_param_in_domain, exp_in_domain, obs_in_domain,
                 inv_input_dim, exp_input_dim, obs_input_dim, obs_reading_count,
                 x_input_dim, y_output_dim, 
                 ensemble_size: int = 100, pinn_ensemble_args: Dict = dict(), 
                 pinn_share_init: bool = False, pinn_init_meta_rounds: int = 0, pinn_init_meta_steps: int = 1000, pinn_meta_eps: float = 0.1,
                 ensemble_steps: int = 100000, 
                 acq_fn: str = 'ucb', ed_criterion: str = 'eig',
                 llh_function: str = 'nllh', llh_args: Dict = dict(),
                 vbed_use_pinns: bool = True, vbed_vectorise_simulator: bool = False,
                 vbed_method: str = 'marg', vbed_clusters: int = 10, 
                 vbed_optim_args: Dict = dict(learning_rate=0.01), vbed_steps: int = 1000,
                 exp_setup_rounds: int = 10, obs_setup_rounds: int = 10, obs_search_time_limit: float = 3600., min_obs_rounds: int = 3, noise_std: float = 1e-3, 
                 seed: int = 0):
        super().__init__(
            simulator_xs=simulator_xs,
            pde=pde, 
            pde_domain=pde_domain, 
            exp_design_fn=exp_design_fn, 
            obs_design_fn=obs_design_fn,
            inv_embedding=inv_embedding,
            inv_param_in_domain=inv_param_in_domain, 
            exp_in_domain=exp_in_domain, 
            obs_in_domain=obs_in_domain,
            inv_input_dim=inv_input_dim, 
            exp_input_dim=exp_input_dim, 
            obs_input_dim=obs_input_dim, 
            obs_reading_count=obs_reading_count,
            x_input_dim=x_input_dim,
            y_output_dim=y_output_dim,
            use_pinns=vbed_use_pinns,
            ensemble_size=ensemble_size,
            ensemble_steps=ensemble_steps,
            pinn_share_init=pinn_share_init,
            pinn_init_meta_rounds=pinn_init_meta_rounds,
            pinn_init_meta_steps=pinn_init_meta_steps,
            pinn_meta_eps=pinn_meta_eps,
            pinn_ensemble_args=pinn_ensemble_args,
            acq_fn=acq_fn,
            exp_setup_rounds=exp_setup_rounds,
            obs_setup_rounds=obs_setup_rounds,
            obs_search_time_limit=obs_search_time_limit,
            noise_std=noise_std,
            min_obs_rounds=min_obs_rounds,
            obs_optim_with_gd=False,
            do_jit=False,
            seed=seed,
        )
        
        self.vbed_use_pinns = vbed_use_pinns
        self.vbed_vectorise_simulator = vbed_vectorise_simulator
        self.ed_criterion = ed_criterion
        self.llh_function = llh_function
        self.llh_args = llh_args
        self.vbed_method = vbed_method
        self.vbed_clusters = vbed_clusters
        self.vbed_steps = vbed_steps
        self.vbed_optim_args = vbed_optim_args
        
    def _generate_criterion_inner(self, exp_design, true_inv_prior_samples):
                    
        if self.vbed_method == 'marg':
            
            llh_fn = generate_loss(loss=self.llh_function, **self.llh_args)
            ys_shape = (self.obs_reading_count,)
            
            def log_q(q_params, y):
                mus = q_params['means']
                covs = jax.vmap(lambda v: v.T @ v)(q_params['vs'])
                log_norm_val = jax.vmap(norm_logpdf, in_axes=(None, 0, 0))(y, mus, covs)
                weights = softmax(q_params['weights'])
                return logsumexp(a=log_norm_val, b=weights)
            
            def log_llh_on_self(ys_noisy, ys, rng):
                p0 = 0. - llh_fn(ys_noisy, ys)
                return p0
            
            def upper_bound_one_term(q_params, ys, rng):
                ys_noisy = ys + self.noise_std * jax.random.normal(key=rng, shape=ys_shape)
                return log_llh_on_self(ys_noisy, ys, rng) - log_q(q_params, ys_noisy)
            
            def upper_bound(q_params, ys_all, rng):
                rng_split = jax.random.split(rng, num=self.ensemble_size)
                s = jax.vmap(upper_bound_one_term, in_axes=(None, 0, 0))(q_params, ys_all, rng_split)
                return jnp.mean(s)
            
            if (not self.vbed_use_pinns) and (self.vbed_vectorise_simulator):
                sim = jax.jit(lambda r, inv, xs: self.simulator_xs(exp_design=exp_design, inv=inv, rng=r)(xs))
            
            def generate_variational_ub_fn(obs_design, rng):
                
                # xs = self.obs_design_fn(obs_design)
                # if self.vbed_use_pinns:
                #     ys_all = self.forward_ens.generate_pred_function()(xs).reshape(self.ensemble_size, ys_shape[0])
                # elif self.vbed_vectorise_simulator:
                #     rng, key_ = jax.random.split(rng)
                #     keys = jax.random.split(key_, num=self.ensemble_size)
                #     ys_all = jax.vmap(sim, in_axes=(0, 0, None))(keys, true_inv_prior_samples, xs).reshape(self.ensemble_size, ys_shape[0])
                # else:
                #     rng, key_ = jax.random.split(rng)
                #     keys = jax.random.split(key_, num=self.ensemble_size)
                #     ys_all = []
                #     for inv, r in zip(true_inv_prior_samples, keys):
                #         ys_all.append(self.simulator_xs(exp_design=exp_design, inv=inv, rng=r)(xs))
                #     ys_all = jnp.array(ys_all).reshape(self.ensemble_size, ys_shape[0])
                
                if self.vbed_use_pinns:
                    oracle = get_vmap_oracle(self.forward_ens, self.obs_design_fn)
                    nn_params = self.forward_ens.params['net']
                    ys_all = oracle(nn_params, obs_design)
                    # ys_all = self.forward_ens.generate_pred_function()(xs)
                elif self.vbed_vectorise_simulator:
                    rng, key_ = jax.random.split(rng)
                    keys = jax.random.split(key_, num=self.ensemble_size)
                    # ys_all = jax.vmap(sim, in_axes=(0, 0, None))(keys, true_inv_prior_samples, xs)
                    oracle = lambda r, inv, obs: self.obs_design_fn(
                        lambda xs: self.simulator_xs(exp_design=exp_design, inv=inv, rng=r)(xs),
                        obs_design
                    )
                    ys_all = jax.vmap(oracle, in_axes=(0, 0, None))(keys, true_inv_prior_samples, obs_design)
                else:
                    rng, key_ = jax.random.split(rng)
                    keys = jax.random.split(key_, num=self.ensemble_size)
                    ys_all = []
                    for inv, r in zip(true_inv_prior_samples, keys):
                    #     ys_all.append(self.simulator_xs(exp_design=exp_design, inv=inv, rng=r)(xs))
                        f = self.simulator_xs(exp_design=exp_design, inv=inv, rng=r)
                        ys_all.append(self.obs_design_fn(f, obs_design))
                    ys_all = jnp.array(ys_all)
                
                def _fn(q_params, rng):
                    return upper_bound(q_params, ys_all, rng)
                
                return _fn
        
            def criterion(obs_design, rng=jax.random.PRNGKey(0)):
                
                rng, k1, k2, k3 = jax.random.split(rng, 4)
                q_params_init = {
                    'means': jax.random.uniform(k1, shape=(self.vbed_clusters, ys_shape[0])),
                    'vs': jax.random.uniform(k2, shape=(self.vbed_clusters, ys_shape[0], ys_shape[0])),
                    'weights': jax.random.uniform(k3, shape=(self.vbed_clusters,)),
                }
                
                rng, k1 = jax.random.split(rng, 2)
                U_marg = jax.jit(generate_variational_ub_fn(obs_design, k1))
                
                # solver = jaxopt.PolyakSGD(
                solver = jaxopt.OptaxSolver(
                    opt=optax.adam(**self.vbed_optim_args),
                    maxiter=self.vbed_steps,
                    fun=U_marg, 
                    # **self.vbed_optim_args
                )
                state = solver.init_state(q_params_init)
                q_params = q_params_init
                for _ in range(self.vbed_steps):
                    rng, k_ = jax.random.split(rng)
                    q_params, state = solver.update(q_params, state, rng=k_)
                
                aux = {
                    # 'U_marg': U_marg,
                    # 'q_params_init': q_params_init,
                    'q_params_opt': q_params,
                }
                return U_marg(q_params, rng), aux
        
            helper_fns = {
                'llh_fn': llh_fn,
                'log_q': log_q,
                'log_llh_on_self': log_llh_on_self,
                'upper_bound_one_term': upper_bound_one_term,
                'upper_bound': upper_bound,
                'generate_variational_ub_fn': generate_variational_ub_fn,
                'criterion': criterion,
            }
        
        else:
            raise ValueError(f'Invalid vbed_method - {self.vbed_method}')
            
        return criterion, helper_fns
