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
from ..utils.jax_utils import flatten, vmap_mjp, vmap_jmp, jacobian_outer_product
from ..utils.vmap_chunked import vmap_chunked
# from .ed_loop import ExperimentalDesign
from .criterion_based import CriterionBasedAbstractMethod
from .eig_estimators.losses import generate_loss

# logger for this file
logging.getLogger().setLevel(logging.INFO)


class RandomMethod(CriterionBasedAbstractMethod):
    
    def __init__(self, simulator_xs, pde, pde_domain, exp_design_fn, obs_design_fn,
                 inv_embedding, inv_param_in_domain, exp_in_domain, obs_in_domain,
                 inv_input_dim, exp_input_dim, obs_input_dim, obs_reading_count,
                 x_input_dim, y_output_dim, noise_std: float = 1e-3,
                 ensemble_size: int = 100, pinn_ensemble_args: Dict = dict(), 
                 pinn_share_init: bool = False, pinn_init_meta_rounds: int = 0, pinn_init_meta_steps: int = 1000, pinn_meta_eps: float = 0.1,
                 ensemble_steps: int = 100000, obs_search_time_limit: float = 3600., seed: int = 0):
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
            use_pinns=pinn_share_init,
            ensemble_size=ensemble_size,
            ensemble_steps=ensemble_steps,
            pinn_ensemble_args=pinn_ensemble_args,
            pinn_share_init=pinn_share_init,
            pinn_init_meta_rounds=pinn_init_meta_rounds,
            pinn_init_meta_steps=pinn_init_meta_steps,
            pinn_meta_eps=pinn_meta_eps,
            exp_setup_rounds=1,
            obs_setup_rounds=1,
            obs_search_time_limit=obs_search_time_limit,
            noise_std=noise_std,
            obs_optim_with_gd=False,
            do_jit=False,
            seed=seed,
        )
        
        # set so that no actual forward ensemble training is done
        self.forward_ensemble_steps = 0
        
    def _generate_criterion_inner(self, exp_design, true_inv_prior_samples):
        
        def criterion(obs_design, k=jax.random.PRNGKey(0)):
            return 0., dict()
        
        return criterion, dict()
