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
from jax.scipy.stats import multivariate_normal

from ..models.pinn_ensemble import PINNEnsemble
from ..utils import sample_from_uniform
from ..utils.jax_utils import flatten, vmap_mjp, vmap_jmp, jacobian_outer_product
from ..utils.vmap_chunked import vmap_chunked
# from .ed_loop import ExperimentalDesign
from .criterion_based import CriterionBasedAbstractMethod
from .utils.obs_fn_helper import get_vmap_oracle

# logger for this file
logging.getLogger().setLevel(logging.INFO)



class GPMutualInformationMethod(CriterionBasedAbstractMethod):
    
    def __init__(self, simulator_xs, pde, pde_domain, exp_design_fn, obs_design_fn,
                 inv_embedding, inv_param_in_domain, exp_in_domain, obs_in_domain,
                 inv_input_dim, exp_input_dim, obs_input_dim, obs_reading_count,
                 x_input_dim, y_output_dim, noise_std: float = 1e-3, 
                 ensemble_size: int = 100, use_pinns: bool = True, pinn_ensemble_args: Dict = dict(), ensemble_steps: int = 100000, 
                 pinn_share_init: bool = False, pinn_init_meta_rounds: int = 0, pinn_init_meta_steps: int = 1000, pinn_meta_eps: float = 0.1,
                 pool_size: int = 100, acq_fn: str = 'ucb', exp_setup_rounds: int = 10, obs_setup_rounds: int = 10, obs_search_time_limit: float = 3600.,
                 min_obs_rounds: int = 3, seed: int = 0):
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
            min_obs_rounds=min_obs_rounds,
            do_jit=True,
            seed=seed,
        )
        
        self.pool_size = pool_size
        
    def _generate_criterion_inner(self, exp_design, true_inv_prior_samples):
        if self.use_pinns:
            pred_fn = self.forward_ens.generate_pred_function()

        else:
            oracles = [self.simulator_xs(exp_design, inv) for inv in true_inv_prior_samples]
            def pred_fn(xs):
                return jnp.array([fn(xs) for fn in oracles])

        single_cov = lambda y1, y2: jnp.cov(y1, y2)[0,1]

        def cov_fn(y1, y2):
            return jax.vmap(lambda y: jax.vmap(lambda y_: single_cov(y, y_))(y2))(y1)

        xs_pool = jnp.array(
            self.pde_domain.random_points(self.pool_size, random='Hammersley'),
            # dtype=jnp.zeros(1).dtype,
        )   
        if self.use_pinns:
            ys_pool = self.forward_ens.generate_pred_function()(xs_pool).reshape(self.ensemble_size, -1).T
            oracle = get_vmap_oracle(self.forward_ens, self.obs_design_fn)
            nn_params = self.forward_ens.params['net']
        else:
            ys_pool = pred_fn(xs_pool).reshape(self.ensemble_size, -1).T
        K_pp = cov_fn(ys_pool, ys_pool)
        
        
        def criterion(obs_design, rng=jax.random.PRNGKey(0)):
            if self.use_pinns:
                ys_obs = oracle(nn_params, obs_design).reshape(self.ensemble_size, -1).T
            else:
                ys_obs = jnp.array([self.obs_design_fn(fn, obs_design) for fn in oracles]).reshape(self.ensemble_size, -1).T
            K_po = cov_fn(ys_pool, ys_obs)
            K_oo = cov_fn(ys_obs, ys_obs) + self.noise_std**2 * jnp.eye(self.obs_reading_count)
            posterior = K_pp - (K_po @ jnp.linalg.inv(K_oo) @ K_po.T) + self.noise_std**2 * jnp.eye(K_pp.shape[0])
            return - jnp.linalg.slogdet(posterior)[1], dict()
        
        aux = {
            'K_fn': cov_fn,
            'xs_pool': xs_pool,
            #'oracle': oracle,
            #'nn_params': nn_params,
        }
        return criterion, aux
