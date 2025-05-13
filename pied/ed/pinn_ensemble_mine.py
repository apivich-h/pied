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
from ..models.model_loader import construct_net
from ..utils import sample_from_uniform
from ..utils.vmap_chunked import vmap_chunked
# from .ed_loop import ExperimentalDesign
from .criterion_based import CriterionBasedAbstractMethod
from .utils.obs_fn_helper import get_vmap_oracle

# logger for this file
logging.getLogger().setLevel(logging.INFO)


class PINNEnsembleWithMINEMethod(CriterionBasedAbstractMethod):
    
    def __init__(self, simulator_xs, pde, pde_domain, exp_design_fn, obs_design_fn,
                 inv_embedding, inv_param_in_domain, exp_in_domain, obs_in_domain,
                 inv_input_dim, exp_input_dim, obs_input_dim, obs_reading_count,
                 x_input_dim, y_output_dim, 
                 ensemble_size: int = 100, pinn_ensemble_args: Dict = dict(), 
                 pinn_share_init: bool = False, pinn_init_meta_rounds: int = 0, pinn_init_meta_steps: int = 1000, pinn_meta_eps: float = 0.1,
                 ensemble_steps: int = 100000, 
                 acq_fn: str = 'ucb', ed_criterion: str = 'eig',
                 llh_function: str = 'nllh', llh_args: Dict = dict(),
                 mine_use_pinns: bool = True, mine_vectorise_simulator: bool = False,
                 mine_use_sampled_invs_as_prior: bool = False, mine_debias_loss_alpha: float = 0.,
                 mine_nn_args: Dict = dict(hidden_layers=2, hidden_dim=8), 
                 mine_optim_args: Dict = dict(learning_rate=1e-2), mine_train_steps: int = 1000, mine_train_set_size: int = 128,
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
            use_pinns=mine_use_pinns,
            ensemble_size=ensemble_size,
            ensemble_steps=ensemble_steps,
            pinn_ensemble_args=pinn_ensemble_args,
            pinn_share_init=pinn_share_init,
            pinn_init_meta_rounds=pinn_init_meta_rounds,
            pinn_init_meta_steps=pinn_init_meta_steps,
            pinn_meta_eps=pinn_meta_eps,
            acq_fn=acq_fn,
            exp_setup_rounds=exp_setup_rounds,
            obs_setup_rounds=obs_setup_rounds,
            obs_search_time_limit=obs_search_time_limit,
            min_obs_rounds=min_obs_rounds,
            noise_std=noise_std,
            obs_optim_with_gd=False,
            do_jit=False,
            seed=seed,
        )
        
        self.mine_use_pinns = mine_use_pinns
        self.mine_vectorise_simulator = mine_vectorise_simulator
        self.ed_criterion = ed_criterion
        self.llh_function = llh_function
        self.llh_args = llh_args
        self.mine_nn_args = mine_nn_args
        self.mine_optim_args = mine_optim_args
        self.mine_debias_loss = (mine_debias_loss_alpha <= 0.)
        self.mine_debias_loss_alpha = mine_debias_loss_alpha
        self.mine_train_set_size = mine_train_set_size
        self.mine_train_steps = mine_train_steps
        self.mine_use_sampled_invs_as_prior = mine_use_sampled_invs_as_prior
        
    def _generate_criterion_inner(self, exp_design, true_inv_prior_samples):
        
        T_fn, _ = construct_net(
            input_dim=(self.y_output_dim + self.inv_input_dim),
            output_dim=1,
            **self.mine_nn_args
        )
        
        @jax.jit
        def ema(running_avg, m):
            return self.mine_debias_loss_alpha * m + (1. - self.mine_debias_loss_alpha) * running_avg
        
        @jax.jit
        def modified_apply(params, y, inv):
            combined_input = jnp.concatenate([y, inv], axis=-1)
            return T_fn.apply(params, combined_input)
                    
        @jax.jit
        def variational_ub_fn(params, y_in, beta_in, y_out, beta_out, running_avg):
            T_in = modified_apply(params, y_in, beta_in)
            T_out = modified_apply(params, y_out, beta_out)
            # we do inverse of paper since the GD algorithm performs a minimization
            return 0. - ( jnp.mean(T_in) - (logsumexp(a=T_out) - jnp.log(self.mine_train_set_size)) )
        
        @jax.jit
        def variational_ub_fn_debias(params, y_in, beta_in, y_out, beta_out, running_avg):
            T_in = modified_apply(params, y_in, beta_in)
            T_out = modified_apply(params, y_out, beta_out)
            et = jnp.exp(T_out)
            # does not give true loss, but its gradient will match that of the debiased version
            # loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))
            return 0. - ( jnp.mean(T_in) - (jnp.mean(et) / running_avg) )
            
        
        if (not self.mine_use_pinns) and (self.mine_vectorise_simulator):
            sim = jax.jit(lambda r, inv, xs: self.simulator_xs(exp_design=exp_design, inv=inv, rng=r)(xs))
            
            
        def criterion(obs_design, rng=jax.random.PRNGKey(0)):
            
            rng, key_ = jax.random.split(rng)
            T_params = T_fn.init(key_, jnp.ones(shape=(1, (self.y_output_dim * self.obs_reading_count) + self.inv_input_dim)))
            
            inv_all = true_inv_prior_samples
            # xs = self.obs_design_fn(obs_params)
            if self.mine_use_pinns:
                oracle = get_vmap_oracle(self.forward_ens, self.obs_design_fn)
                nn_params = self.forward_ens.params['net']
                ys_all = oracle(nn_params, obs_design)
                # ys_all = self.forward_ens.generate_pred_function()(xs)
            elif self.mine_vectorise_simulator:
                # rng, key_ = jax.random.split(rng)
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
            
            opt = optax.adam(**self.mine_optim_args)
            if self.mine_debias_loss:
                solver = jaxopt.OptaxSolver(opt=opt, fun=variational_ub_fn_debias, value_and_grad=False, maxiter=self.mine_train_steps)
            else:
                solver = jaxopt.OptaxSolver(opt=opt, fun=variational_ub_fn, value_and_grad=False, maxiter=self.mine_train_steps)
            opt_state = solver.init_state(T_params)
            step_fn = jax.jit(solver.update)
            
            @jax.jit
            def _generate_data(k1, k2, k3, k4, k5):
                
                y_in_idxs = jax.random.choice(k1, self.ensemble_size, shape=(self.mine_train_set_size,), replace=True)
                y_in = ys_all[y_in_idxs].reshape(self.mine_train_set_size, -1)
                y_in += self.noise_std * jax.random.normal(k2, shape=y_in.shape)
                beta_in = inv_all[y_in_idxs]
                
                y_out_idxs = jax.random.choice(k3, self.ensemble_size, shape=(self.mine_train_set_size,), replace=True)
                y_out = ys_all[y_out_idxs].reshape(self.mine_train_set_size, -1)
                y_out += self.noise_std * jax.random.normal(k4, shape=y_out.shape)
                
                if self.mine_use_sampled_invs_as_prior:
                    beta_out_idxs = jax.random.choice(k5, self.ensemble_size, shape=(self.mine_train_set_size,), replace=True)
                    beta_out = inv_all[beta_out_idxs]
                else:
                    beta_out = self.sample_inv_param(self.mine_train_set_size, rng=k5)
                    
                return y_in, beta_in, y_out, beta_out
            
            for i in range(self.mine_train_steps):
                
                rng, k1, k2, k3, k4, k5 = jax.random.split(rng, num=6)
                y_in, beta_in, y_out, beta_out = _generate_data(k1, k2, k3, k4, k5)
                
                # running average to do some debiasing
                T_out = modified_apply(T_params, y_out, beta_out)
                m = jnp.mean(jnp.exp(T_out))
                if (i == 0) or (not self.mine_debias_loss):
                    running_avg = m
                else: 
                    # exponential moving average update rule
                    running_avg = ema(running_avg, m)
                
                T_params, opt_state = step_fn(T_params, opt_state, y_in, beta_in, y_out, beta_out, running_avg)
                
            rng, k1, k2, k3, k4, k5 = jax.random.split(rng, num=6)
            y_in, beta_in, y_out, beta_out = _generate_data(k1, k2, k3, k4, k5)
            final_bound = 0. - variational_ub_fn(T_params, y_in, beta_in, y_out, beta_out, running_avg)
            
            aux = {
                'T_params': T_params,
            }
            return final_bound, aux
    
        helper_fns = {
            'T_fn': T_fn,
            'modified_apply': modified_apply,
            'ema': ema,
            'variational_ub_fn': variational_ub_fn,
            'variational_ub_fn_debias': variational_ub_fn_debias,
            'criterion': criterion,
        }
        return criterion, helper_fns
