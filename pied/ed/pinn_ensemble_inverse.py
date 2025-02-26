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
from scipy.stats import spearmanr, pearsonr
from scipy.interpolate import griddata
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
# from .ed_loop import ExperimentalDesign
from .criterion_based import CriterionBasedAbstractMethod
from .utils.obs_fn_helper import get_vmap_oracle

# logger for this file
logging.getLogger().setLevel(logging.INFO)


class PINNEnsembleInverseMethod(CriterionBasedAbstractMethod):
    
    def __init__(self, simulator_xs, pde, pde_domain, exp_design_fn, obs_design_fn, 
                 inv_embedding, inv_param_in_domain, exp_in_domain, obs_in_domain,
                 inv_input_dim, exp_input_dim, obs_input_dim, obs_reading_count,
                 x_input_dim, y_output_dim, 
                 ensemble_size: int = 100, pinn_ensemble_args: Dict = dict(), ensemble_steps: int = 100000, 
                 pinn_share_init: bool = False, pinn_init_meta_rounds: int = 0, pinn_init_meta_steps: int = 1000, pinn_meta_eps: float = 0.1,
                 acq_fn: str = 'ucb', inverse_training_steps: int = None,
                 exp_setup_rounds: int = 10, obs_setup_rounds: int = 10, obs_search_time_limit: float = 3600., noise_std: float = 1e-3, 
                 reg: float = 1e-12, seed: int = 0):
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
            use_pinns=True,
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
            noise_std=noise_std,
            obs_optim_with_gd=False,
            do_jit=False,
            seed=seed,
        )
        
        self.reg = reg
        self.inverse_ensemble_steps = (inverse_training_steps if inverse_training_steps is not None else ensemble_steps)
            
    def _generate_criterion_inner(self, exp_design, true_inv_prior_samples):
        
        def criterion(obs_design, rng=jax.random.PRNGKey(0)):
        
            k0, k1, k2 = jax.random.split(rng, 3)
        
            # xs_obs = self.obs_design_fn(obs_param=obs_design)
            # ys_obs = self.forward_ens.generate_pred_function()(xs_obs)
            # # ys_obs += self.noise_std * jax.random.normal(key=k0, shape=ys_obs.shape)
            # xs_obs_split = jnp.repeat(xs_obs[None,:], self.ensemble_size, axis=0)
            
            obs_split = jnp.repeat(obs_design[None,:], self.ensemble_size, axis=0)
            oracle = get_vmap_oracle(self.forward_ens, self.obs_design_fn)
            nn_params = self.forward_ens.params['net']
            ys_obs = oracle(nn_params, obs_design)
            
            inverse_ens = PINNEnsemble(
                pde=self.pde, 
                pde_domain=self.pde_domain, 
                exp_design_fn=self.exp_design_fn, 
                obs_design_fn=self.obs_design_fn,
                inv_embedding=self.inv_embedding,
                inv_problem=True,
                rng=k1,
                loss_every=1,
                **self.pinn_ensemble_args
            )   
            
            inv_prior_guess = self.sample_inv_param(n=self.ensemble_size, rng=k2)
            inverse_ens.reset()
            inverse_ens.prep_simulator(
                exp_params=exp_design, 
                inv_params_guesses=inv_prior_guess,
                new_nn_params=(self._generate_shared_params(self.ensemble_size, inv=inv_prior_guess) if self.pinn_share_init else None),
            )

            for i in range(self.inverse_ensemble_steps):
                # inverse_ens.step_opt(xs_obs_split, ys_obs)
                inverse_ens.step_opt(obs_split, ys_obs)
                
            # negative since requires maximisation
            score_fn = lambda inv1, inv2: self.compare_inv(inv1, inv2)
            indiv_score = jax.vmap(score_fn)(inverse_ens.params["inv"], true_inv_prior_samples)
            score = - jnp.nanmean(jnp.log(indiv_score + self.reg))
            aux = {
                'true_inv_prior_samples': true_inv_prior_samples, 
                'inv_prior_guess': inv_prior_guess,
                'inverse_ens_params': inverse_ens.params,
                'indiv_score': indiv_score,
                'losses_steps': inverse_ens._losses_steps,
                'losses_total': inverse_ens._losses,
                'losses_pde': inverse_ens._losses_pde,
                'losses_icbc': inverse_ens._losses_icbc,
                'inverse_ens': inverse_ens,
                'obs_split': obs_split,
                'ys_obs': ys_obs,
            }
            return score, aux
        
        return criterion, dict(forward_params=self.forward_ens.params)
