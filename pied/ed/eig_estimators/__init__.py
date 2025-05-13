from functools import partial
import os
import pickle as pkl
from collections.abc import MutableMapping
from datetime import datetime
from itertools import product
from functools import partial

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

from .losses import generate_loss


def generate_naive_mcmc_estimator(
    noisy_sim_xs, obs_design_fn, sample_inv_param_prior, 
    N=100, M=100, loss='mse', jit=True, **loss_kwargs):
    
    """
    noisy_simulator(exp_params, obs_params, inv_param, rng) -> observed y
    log_cost(y_obs, exp_params, obs_params, inv_param, rng) -> score of the y_obs wrt other params
    sample_inv_param_prior(n, rng) -> n samples of the inverse param
    """
    
    log_cost = jax.jit(generate_loss(noisy_sim_xs, obs_design_fn, loss=loss))

    def _one_mcmc_round(exp_params, obs_params, rng=jax.random.PRNGKey(0)):
        rng, r_ = jax.random.split(rng)
        betas_samples = sample_inv_param_prior(M, r_)
        b0 = betas_samples[0]
        rng, r_ = jax.random.split(rng)
        y0 = noisy_sim_xs(exp_params, b0, r_)(obs_design_fn(obs_params))
        betas_prior = betas_samples[1:]
        rng, rng1 = jax.random.split(rng)
        rngs = jax.random.split(rng, M - 1)
        p0 = log_cost(y0, exp_params, obs_params, b0, rng1)
        evidence = jax.vmap(log_cost, in_axes=(None, None, None, 0, 0))(
            y0, exp_params, obs_params, betas_prior, rngs)
        return p0 - logsumexp(evidence, b=1./(M - 1.))
    
    if jit:
        _one_mcmc_round = jax.jit(_one_mcmc_round)

    def naive_mcmc(exp_params, obs_params, rng=jax.random.PRNGKey(0)):
        rngs = jax.random.split(rng, N)
        terms = jax.vmap(_one_mcmc_round, in_axes=(None, None, 0))(exp_params, obs_params, rngs)
        return jnp.mean(terms)
    
    if jit:
        naive_mcmc = jax.jit(naive_mcmc)
    
    aux = {
        'one_mcmc_round': _one_mcmc_round,
        'log_cost': log_cost,
    }
    return naive_mcmc, aux
