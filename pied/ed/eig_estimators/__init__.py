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
# import gpjax as gpx

from ...models.deeponet_modified import DeepONetModified, train_deeponet, train_deeponet_with_pde, \
    generate_train_set_for_deeponet, generate_collocation_train_set_for_deeponet, generate_bc_train_set_for_deeponet
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


def generate_deeponet_based_mcmc_estimator(
    noisy_sim_xs, obs_design_fn, sample_inv_param_prior,
    exp_design_fns, exp_setup_list, pde_domain, pde, 
    inv_input_dim, exp_input_dim, x_input_dim, 
    deeponet_hidden_layers=4, deeponet_hidden_dim=128, deeponet_arch=None, learned_params=None,
    n_inv_sim=100, n_xs_sim=1000, n_inv_colloc=1000, n_xs_colloc=10000,
    steps=200000, batch_size=4096, optim_type='adam', optim_args=None, 
    noise_variance=0.001, noise_lengthscale=0.01,
    N=100, M=100, loss='mse', jit=True, rng=jax.random.PRNGKey(0), **loss_kwargs):
    
    deeponet = DeepONetModified(
        inv_input_dim=inv_input_dim, exp_input_dim=exp_input_dim, x_input_dim=x_input_dim, 
        hidden_layers=deeponet_hidden_layers, hidden_dim=deeponet_hidden_dim, arch=deeponet_arch)
    
    if learned_params is None:
    
        rng, r_ = jax.random.split(rng)
        dset = generate_train_set_for_deeponet(
            noisy_simulator_xs=noisy_sim_xs, 
            exp_setup_list=exp_setup_list, 
            inv_param_sampler=sample_inv_param_prior, 
            pde_domain=pde_domain,
            n_inv=n_inv_sim, 
            n_xs=n_xs_sim, 
            batch_size=batch_size, 
            rng=r_,
        )
        
        rng, r_ = jax.random.split(rng)
        dset_colloc = generate_collocation_train_set_for_deeponet(
            exp_setup_list=exp_setup_list, 
            inv_param_sampler=sample_inv_param_prior, 
            pde_domain=pde_domain,
            n_inv=n_inv_colloc, 
            n_xs=n_xs_colloc, 
            batch_size=batch_size, 
            rng=r_,
        )

        dset_bcs = []
        for bc_fn in exp_design_fns:
            rng, r_ = jax.random.split(rng)
            dset_bcs.append(generate_bc_train_set_for_deeponet(
                exp_setup_list=exp_setup_list, 
                inv_param_sampler=sample_inv_param_prior, 
                pde_domain=pde_domain,
                bc=bc_fn(exp_param=exp_setup_list[0]),
                n_inv=n_inv_colloc, 
                n_xs=n_xs_colloc, 
                batch_size=batch_size, 
                rng=r_,
            ))
        
        rng, r_ = jax.random.split(rng)
        params, aux_training = train_deeponet_with_pde(
            deeponet=deeponet,
            dset=dset,
            colloc_dset=dset_colloc,
            pde=pde,
            bcs_dset=dset_bcs,
            bcs_gen=exp_design_fns,
            steps=steps,
            optim_type=optim_type, 
            optim_args=optim_args, 
            rng=r_,
        )
        aux_don = {
           'dset': dset,
           'dset_pde': dset_colloc,
           'dset_bcs': dset_bcs,
           'training': aux_training, 
        }
        
    else:
        
        params = learned_params
        aux_don = dict()
        
    noise_prior = gpx.Prior(
        mean_function=gpx.mean_functions.Zero(), 
        kernel=gpx.kernels.RBF(variance=noise_variance, lengthscale=noise_lengthscale)
    )

    def noisy_sim_xs_estimate(exp_design, inv, rng=jax.random.PRNGKey(42)):
        
        def _fn(xs):
            prior_dist = noise_prior.predict(xs)
            ys = deeponet.apply_single_branch(params, inv, exp_design, xs)
            noise = prior_dist.sample(seed=rng, sample_shape=(1,)).reshape(ys.shape)
            assert ys.shape == noise.shape, (ys.shape, noise.shape)
            return ys + noise
        
        return _fn
    
    def generate_mcmc_est_from_deeponet(N_, M_, loss_, jit_, **loss_kwargs_):
        return generate_naive_mcmc_estimator(
            noisy_sim_xs=noisy_sim_xs_estimate, 
            obs_design_fn=obs_design_fn,
            sample_inv_param_prior=sample_inv_param_prior, 
            N=N_, M=M_, loss=loss_, jit=jit_, **loss_kwargs_
        )
        
    mcmc_est, aux_estimate = generate_mcmc_est_from_deeponet(N_=N, M_=M, loss_=loss, jit_=jit, **loss_kwargs)
    
    aux = {
        'deeponet': deeponet,
        'params': params,
        'apply_fn': lambda inv, exp, xs: deeponet.apply_single_branch(params, inv, exp, xs),
        'noisy_sim_xs_estimate': noisy_sim_xs_estimate,
        'aux_deeponet_training': aux_don,
        'aux_mcmc_est': aux_estimate,
        'generate_mcmc_est_from_deeponet': generate_mcmc_est_from_deeponet,
    }
    return mcmc_est, aux
