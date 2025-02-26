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
from .ed_loop import ExperimentalDesign
from .eig_estimators.losses import generate_loss
from .criterion_based import CriterionBasedAbstractMethod
from .utils.obs_fn_helper import get_vmap_oracle

# logger for this file
logging.getLogger().setLevel(logging.INFO)


class SimulatorEnsembleMethod(CriterionBasedAbstractMethod):
    
    def __init__(self, simulator_xs, pde, pde_domain, exp_design_fn, obs_design_fn,
                 inv_embedding, inv_param_in_domain, exp_in_domain, obs_in_domain,
                 inv_input_dim, exp_input_dim, obs_input_dim, obs_reading_count,
                 x_input_dim, y_output_dim, 
                 ensemble_size: int = 100, use_pinns: bool = True, vectorise_simulator: bool = False,
                 ensemble_steps: int = 100000, pinn_ensemble_args: Dict = dict(), 
                 pinn_share_init: bool = False, pinn_init_meta_rounds: int = 0, pinn_init_meta_steps: int = 1000, pinn_meta_eps: float = 0.1,
                 acq_fn: str = 'ucb', ed_criterion: str = 'eig',
                 llh_function: str = 'nllh', llh_args: Dict = dict(), N: int = 10, M: int = 10,
                 exp_setup_rounds: int = 10, obs_setup_rounds: int = 10, obs_search_time_limit: float = 3600., noise_std: float = 1e-3, 
                 obs_optim_gd_params: Dict = dict(stepsize=1e-2, maxiter=1000, acceleration=True), obs_optim_grad_clip: float = None, obs_optim_grad_jitter: float = None,
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
            use_pinns=use_pinns,
            ensemble_size=ensemble_size,
            pinn_ensemble_args=pinn_ensemble_args,
            ensemble_steps=ensemble_steps,
            pinn_share_init=pinn_share_init,
            pinn_init_meta_rounds=pinn_init_meta_rounds,
            pinn_init_meta_steps=pinn_init_meta_steps,
            pinn_meta_eps=pinn_meta_eps,
            acq_fn=acq_fn,
            exp_setup_rounds=exp_setup_rounds,
            obs_setup_rounds=obs_setup_rounds,
            obs_search_time_limit=obs_search_time_limit,
            noise_std=noise_std,
            obs_optim_with_gd=(vectorise_simulator or use_pinns),
            do_jit=(vectorise_simulator or use_pinns),
            obs_optim_gd_params=obs_optim_gd_params,
            obs_optim_grad_clip=obs_optim_grad_clip,
            obs_optim_grad_jitter=obs_optim_grad_jitter,
            seed=seed,
        )
        
        assert ensemble_size == N * (M + 1)  # constant for ensemble size vs MCMC estimator
        self.vectorise_simulator = vectorise_simulator
        self.ed_criterion = ed_criterion
        self.llh_function = llh_function
        self.llh_args = llh_args
        self.N = N
        self.M = M
        
    def _generate_criterion_inner(self, exp_design, true_inv_prior_samples):
        
        ys_shape = (self.obs_reading_count,)
        if self.use_pinns:
            oracle = get_vmap_oracle(self.forward_ens, self.obs_design_fn)
            nn_params = self.forward_ens.params['net']
            simulators = lambda obs: oracle(nn_params, obs)
        elif self.vectorise_simulator:
            # sim = lambda r, inv, xs: self.simulator_xs(exp_design=exp_design, inv=inv, rng=r)(xs)
            keys = self.get_rng(n=self.ensemble_size)
            # def simulators(xs):
            #     return jax.vmap(sim, in_axes=(0, 0, None))(keys, true_inv_prior_samples, xs)
            oracle = lambda r, inv, obs_design: self.obs_design_fn(
                lambda xs: self.simulator_xs(exp_design=exp_design, inv=inv, rng=r)(xs),
                obs_design
            )
            simulators = lambda obs: jax.vmap(oracle, in_axes=(0, 0, None))(keys, true_inv_prior_samples, obs)
        else:
            simulators = []
            for inv, r in zip(true_inv_prior_samples, self.get_rng(n=self.ensemble_size)):
                # simulators.append(self.simulator_xs(exp_design=exp_design, inv=inv, rng=r))
                f = self.simulator_xs(exp_design=exp_design, inv=inv, rng=r)
                simulators.append(lambda obs: self.obs_design_fn(f, obs))

        llh_fn = generate_loss(loss=self.llh_function, **self.llh_args)
            
        def _one_mcmc_round(y_samples, rng):
            k1, k2 = jax.random.split(rng)
            y0 = y_samples[0] + self.noise_std * jax.random.normal(key=k1, shape=ys_shape)
            y_priors = y_samples + self.noise_std * jax.random.normal(key=k2, shape=((self.M+1,) + ys_shape))
            p0 = 0. - llh_fn(y_priors[0], y0)
            evidence = 0. - jax.vmap(llh_fn, in_axes=(0, None))(y_priors[1:], y0)
            return p0 - logsumexp(evidence, b=1./self.M)
            
        def criterion(obs_design, rng=jax.random.PRNGKey(0)):
            
            if self.use_pinns or self.vectorise_simulator:
                ys = simulators(obs_design)
            else:
                sims = [simulator(obs_design) for simulator in simulators]
                ys = jnp.array(sims)
                
            if self.ed_criterion == 'eig':
                ys_divided = ys.reshape((self.N, self.M+1) + ys_shape)
                p = jax.vmap(_one_mcmc_round, in_axes=(0, 0))(ys_divided, jax.random.split(rng, num=self.N))
                return jnp.nanmean(p), dict(ys_divided=ys_divided, p=p)
            else:
                raise ValueError
            
        if (not self.use_pinns) and self.vectorise_simulator:
            criterion = jax.jit(criterion)
            
        return criterion, dict(criterion=criterion, one_mcmc_round=_one_mcmc_round, simulators=simulators)
