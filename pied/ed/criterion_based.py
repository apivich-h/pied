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

from ..models.pinn_ensemble import PINNEnsemble
from ..utils import sample_from_uniform
from ..utils.jax_utils import tree_stack
from ..utils.vmap_chunked import vmap_chunked
from .ed_loop import ExperimentalDesign
from .eig_estimators.losses import generate_loss

# logger for this file
logging.getLogger().setLevel(logging.INFO)


class CriterionBasedAbstractMethod(ExperimentalDesign):
    
    def __init__(self, simulator_xs, pde, pde_domain, exp_design_fn, obs_design_fn,
                 inv_embedding, inv_param_in_domain, exp_in_domain, obs_in_domain,
                 inv_input_dim, exp_input_dim, obs_input_dim, obs_reading_count,
                 x_input_dim, y_output_dim, use_pinns: bool = True, 
                 pinn_share_init: bool = False, pinn_init_meta_rounds: int = 0, pinn_init_meta_steps: int = 1000, pinn_meta_eps: float = 0.1,
                 ensemble_size: int = 100, pinn_ensemble_args: Dict = dict(), ensemble_steps: int = 100000, ensemble_inv_reg: float = 0.,
                 exp_setup_rounds: int = 10, obs_setup_rounds: int = 10, obs_search_time_limit: float = 3600., noise_std: float = 1e-3, 
                 acq_fn: str = 'ucb', obs_optim_with_gd: bool = False, do_jit: bool = False, min_obs_rounds: int = 3,
                 obs_optim_use_lbfgs: bool = False, obs_optim_gd_params: Dict = dict(stepsize=1e-2, maxiter=10000, acceleration=True), 
                 obs_optim_grad_clip: float = None, obs_optim_grad_jitter: float = None, obs_optim_grad_zero_rate: float = None,
                 obs_optim_gd_use_jacfwd: bool = False, seed: int = 0):
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
            seed=seed,
        )
        
        # ensemble parameters
        self.use_pinns = use_pinns
        self.ensemble_size = ensemble_size
        self.forward_ensemble_steps = ensemble_steps
        self.inverse_ensemble_steps = ensemble_steps
        self.pinn_ensemble_args = pinn_ensemble_args
        self.ensemble_inv_reg = ensemble_inv_reg
        
        # criterion parameters
        self.noise_std = noise_std
        self.acq_fn = acq_fn
        self.exp_setup_rounds = exp_setup_rounds
        self.obs_setup_rounds = obs_setup_rounds
        self.obs_search_time_limit = obs_search_time_limit
        self.do_jit = do_jit
        self.min_obs_rounds = min_obs_rounds
        
        # obs param GD params
        self.obs_optim_with_gd = obs_optim_with_gd
        self.obs_optim_gd_params = obs_optim_gd_params
        self.obs_optim_grad_clip = obs_optim_grad_clip
        self.obs_optim_grad_jitter = obs_optim_grad_jitter
        self.obs_optim_grad_zero_rate = obs_optim_grad_zero_rate
        self.obs_optim_use_lbfgs = obs_optim_use_lbfgs
        self.obs_optim_gd_use_jacfwd = obs_optim_gd_use_jacfwd
        
        
        
        self.log_prior_pdf = lambda inv: 1.
        if self.use_pinns:
            self.forward_ens = PINNEnsemble(
                pde=self.pde, 
                pde_domain=self.pde_domain, 
                exp_design_fn=self.exp_design_fn, 
                obs_design_fn=self.obs_design_fn,
                inv_problem=False,
                rng=self.get_rng(),
                **self.pinn_ensemble_args
            )
            self.pinn_share_init = pinn_share_init
            self.pinn_init_meta_rounds = pinn_init_meta_rounds
            self.pinn_init_meta_steps = pinn_init_meta_steps
            self.pinn_meta_eps = pinn_meta_eps
            self.pinn_shared_init_params = None
            self._pinn_shared_init_params_records = []
        else:
            self.forward_ens = None
            self.pinn_shared_init_params = None
            self._pinn_shared_init_params_records = None
        self.inverse_ens = None
        
    def _generate_shared_params(self, n, inv=None):
        init_params = {'net': tree_stack([self.pinn_shared_init_params for _ in range(n)])}
        if inv is not None:
            init_params['inv'] = inv
        return init_params
        
    def _inner_sample_inv_param(self, n, rng):
        raise ValueError
        
    def _inner_experiment_round(self, given_exp_design=None, given_obs_design=None):
                
        ran_exp_params = []
        ran_exp_scores = []
        ran_exp_coresponding_obs = []
        ran_exp_auxs = []
        
        t = time.time()
        inv_prior_samples = self.sample_inv_param(n=self.ensemble_size, rng=self.get_rng())
        t = time.time() - t
        logging.info(f'[TIMING] Sampling inverse params (s) : {t:.6f}')
            
        for i in range(self.exp_setup_rounds):
            
            logging.info(f'[OUTER_LOOP] Running outer loop {i+1} of {self.exp_setup_rounds}.')
            
            if given_exp_design is not None:
                
                exp_design_candidate = given_exp_design
            
            elif i < 2:
                
                exp_design_candidate = sample_from_uniform(
                    n=1, 
                    bounds=self.exp_in_domain, 
                    sample_dim=self.exp_input_dim, 
                    rng=self.get_rng()
                )[0]
            
            else:
                
                train_X = torch.tensor(np.array(ran_exp_params))
                train_Y = torch.tensor(np.array(ran_exp_scores).reshape(-1, 1))
                train_Y = (train_Y - torch.mean(train_Y)) / torch.std(train_Y)

                model = SingleTaskGP(train_X, train_Y)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
                
                sampler = StochasticSampler(sample_shape=torch.Size([128]))
                if self.acq_fn == 'ucb':
                    q_fn = qUpperConfidenceBound(model, beta=1., sampler=sampler)
                elif self.acq_fn == 'ei':
                    q_fn = qExpectedImprovement(model, best_f=train_Y.max(), sampler=sampler)
                else:
                    raise ValueError(f'Invalid self.acq_fn {self.acq_fn}')
                
                exp_domain_np = np.array(self.exp_in_domain)
                Xinit = gen_batch_initial_conditions(q_fn, torch.tensor(exp_domain_np.T), q=1, num_restarts=25, raw_samples=500)
                batch_candidates, batch_acq_values = gen_candidates_torch(
                    initial_conditions=Xinit,
                    acquisition_function=q_fn,
                    lower_bounds=torch.tensor(exp_domain_np[:,0]),
                    upper_bounds=torch.tensor(exp_domain_np[:,1]),
                )
                exp_design_candidate = jnp.array(get_best_candidates(batch_candidates, batch_acq_values)[0].cpu().detach().numpy())
            
            if self.exp_input_dim <= 10:
                logging.info(f'[OUTER_LOOP] Candidate for round {i+1} of {self.exp_setup_rounds} is {exp_design_candidate}.')
            
            if given_obs_design is not None:
                
                criterion, _ = self.generate_criterion(
                    exp_design=exp_design_candidate,
                    true_inv_prior_samples=inv_prior_samples,
                )
                score, aux = criterion(given_obs_design)
                best_obs_param = given_obs_design
            
            else:
            
                t = time.time()
                score, best_obs_param, aux = self._process_exp_design(
                    exp_design=exp_design_candidate,
                    true_inv_prior_samples=inv_prior_samples,
                )
                t = time.time() - t
                logging.info(f'[TIMING] [OUTER_LOOP] Computing score for outer loop {i+1} of {self.exp_setup_rounds} (s) : {t:.6f}')
            
            logging.info(f'[OUTER_LOOP] Score for outer loop {i+1} of {self.exp_setup_rounds} is {score:.10f}.')
            logging.info(f'[OUTER_LOOP] Finished outer loop {i+1} of {self.exp_setup_rounds}.')
            if self.obs_input_dim <= 10:
                logging.info(f'[OUTER_LOOP] Chosen obs_param is {best_obs_param}.')
            if not jnp.isnan(score):
                ran_exp_params.append(exp_design_candidate)
                ran_exp_scores.append(score)
                ran_exp_coresponding_obs.append(best_obs_param)
                ran_exp_auxs.append(aux)
            
        best_i = np.nanargmax(ran_exp_scores)
        best_exp = ran_exp_params[best_i]
        best_obs = ran_exp_coresponding_obs[best_i]
        aux = {
            'best_exp': best_exp,
            'best_obs': best_obs,
            'ran_exp_params': ran_exp_params,
            'ran_exp_scores': ran_exp_scores,
            'ran_exp_coresponding_obs': ran_exp_coresponding_obs,
            'ran_exp_auxs': ran_exp_auxs,
            'inv_prior_samples': inv_prior_samples,
        }
        return best_exp, best_obs, aux
        
    def generate_criterion(self, exp_design, true_inv_prior_samples):
        
        if self.use_pinns:
            t = time.time()
        
            if self.pinn_share_init:
                
                self.pinn_shared_init_params = self.forward_ens.net.init(self.get_rng(), self.forward_ens.pde_collocation_pts[:1])
                self._pinn_shared_init_params_records.append(self.pinn_shared_init_params)
                
                for i in range(self.pinn_init_meta_rounds):
                    init_params = self._generate_shared_params(self.ensemble_size)
                    self.forward_ens.reset()
                    self.forward_ens.prep_simulator(
                        exp_params=exp_design, 
                        inv_params=true_inv_prior_samples,
                        new_nn_params=init_params,
                    )
                    for _ in tqdm.trange(self.pinn_init_meta_steps, desc=f'Meta round {i+1}', mininterval=2):
                        self.forward_ens.step_opt()
                    avg_init = jax.tree_map(lambda x: jnp.mean(x, axis=0), self.forward_ens.params['net'])
                    self.pinn_shared_init_params = jax.tree_map(
                        lambda x, y: x + self.pinn_meta_eps * (y - x), 
                        self.pinn_shared_init_params,
                        avg_init
                    )
                    self._pinn_shared_init_params_records.append(self.pinn_shared_init_params)
                
                init_params = self._generate_shared_params(self.ensemble_size)
                self.forward_ens.reset()
                self.forward_ens.prep_simulator(
                    exp_params=exp_design, 
                    inv_params=true_inv_prior_samples,
                    new_nn_params=init_params,
                )
            
            else:
                self.forward_ens.reset()
                self.forward_ens.prep_simulator(exp_params=exp_design, inv_params=true_inv_prior_samples)
        
            for _ in tqdm.trange(self.forward_ensemble_steps, desc='Forward training', mininterval=2):
                self.forward_ens.step_opt()
            t = time.time() - t
            logging.info(f'[TIMING] [OUTER_LOOP] Training forward ensemble (s) : {t:.6f}')
            
        t = time.time()
        criterion, helper_fns = self._generate_criterion_inner(exp_design=exp_design, true_inv_prior_samples=true_inv_prior_samples)
        t = time.time() - t
        logging.info(f'[TIMING] [OUTER_LOOP] Generate criterion function (s) : {t:.6f}')
        
        return criterion, helper_fns
        
    def _generate_criterion_inner(self, exp_design, true_inv_prior_samples):
        # IMPLEMENT THIS FOR EACH EXAMPLES
        # ASSUMES THAT THE CRITERION IS BEING MAXIMISED!
        raise NotImplementedError
        
    def _process_exp_design(self, exp_design, true_inv_prior_samples):
        
        exp_design_start_time = time.time()
        
        criterion, helper_fns = self.generate_criterion(exp_design=exp_design, true_inv_prior_samples=true_inv_prior_samples)
        
        obs_transform = lambda obs: (obs - self.obs_in_domain[:,0]) / (self.obs_in_domain[:,1] - self.obs_in_domain[:,0])
        obs_rev_transform = lambda x: self.obs_in_domain[:,0] + x * (self.obs_in_domain[:,1] - self.obs_in_domain[:,0])
        hypercube = jnp.array([[0., 1.] for _ in range(self.obs_input_dim)])
        
        if self.obs_optim_with_gd:
        
            obs_param_candidates = []
            
            # k = self.get_rng()
            # add negative sign since the GD algorithm performs minimisation
            criterion_min = lambda x, k=jax.random.PRNGKey(0): 0. - criterion(obs_rev_transform(x), rng=k)[0]
            
            if self.obs_optim_gd_use_jacfwd: 
                def criterion_val_and_grad(x, k=jax.random.PRNGKey(0)):
                    # https://github.com/google/jax/pull/762
                    f = lambda x: criterion_min(x, k=k).reshape(1)
                    pushfwd = partial(jax.jvp, f, (x,))
                    basis = jnp.eye(self.obs_input_dim, dtype=self.obs_in_domain.dtype)
                    y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
                    return y[0], jac[0]
            else:
                criterion_val_and_grad = jax.value_and_grad(criterion_min)
                
            # if self.obs_optim_grad_clip is not None:
            #     def criterion_val_and_grad_modified(x, k=jax.random.PRNGKey(0)):
            #         val, grad = criterion_val_and_grad(x, k)
            #         return val, jnp.clip(grad, a_min=-self.obs_optim_grad_clip, a_max=self.obs_optim_grad_clip)
            # else:
            #     criterion_val_and_grad_modified = criterion_val_and_grad
                
            def criterion_val_and_grad_modified(x, k=jax.random.PRNGKey(0)):
                k1, k2, k3 = jax.random.split(k, num=3)
                val, grad = criterion_val_and_grad(x, k1)
                if self.obs_optim_grad_jitter is not None:
                    grad = grad + self.obs_optim_grad_jitter * jax.random.normal(k2, shape=grad.shape)
                if self.obs_optim_grad_clip is not None:
                    grad = jnp.clip(grad, a_min=-self.obs_optim_grad_clip, a_max=self.obs_optim_grad_clip)
                if self.obs_optim_grad_zero_rate is not None:
                    grad = grad * jax.random.bernoulli(k3, p=(1. - self.obs_optim_grad_zero_rate), shape=grad.shape).astype(grad.dtype)
                return val, grad
            
            if self.do_jit:
                criterion_val_and_grad_modified = jax.jit(criterion_val_and_grad_modified)
                
            if self.obs_optim_use_lbfgs:
                pg = jaxopt.ScipyBoundedMinimize(
                    fun=lambda x, k=jax.random.PRNGKey(0): criterion(x, k)[0], 
                    jit=self.do_jit,
                    # value_and_grad=True,
                    method="l-bfgs-b",
                    **self.obs_optim_gd_params
                )
                gd_paths = None
            else:
                pg = jaxopt.ProjectedGradient(
                    fun=criterion_val_and_grad_modified, 
                    projection=jaxopt.projection.projection_box,
                    value_and_grad=True,
                    jit=self.do_jit,
                    # unroll=True,
                    **self.obs_optim_gd_params
                )
                update_fn = pg.update
                if self.do_jit:
                    update_fn = jax.jit(update_fn)
                gd_paths = []
            
            t = time.time()
            
            for r in range(self.obs_setup_rounds):
                
                t1 = time.time()
                
                obs_candidate = sample_from_uniform(
                    n=1, 
                    bounds=hypercube, 
                    sample_dim=self.obs_input_dim, 
                    rng=self.get_rng()
                )[0]
                gd_path = [obs_rev_transform(obs_candidate)]
                store_sz = max(1, self.obs_optim_gd_params['maxiter'] // 100)
                
                if self.obs_optim_use_lbfgs:
                    logging.info(f'[INNER_LOOP] Starting LBFGS Round {r+1}')
                    k = self.get_rng()
                    best_obs_param = pg.run(obs_candidate, hypercube.T, k).params
                else:
                    logging.info(f'[INNER_LOOP] Starting GD Round {r+1}')
                    best_obs_param = obs_candidate
                    opt_state = pg.init_state(best_obs_param)
                    for i in tqdm.trange(self.obs_optim_gd_params['maxiter'], mininterval=2):
                        k = self.get_rng()
                        b, opt_state = update_fn(best_obs_param, opt_state, hypercube.T, k)
                        if jnp.isnan(b).any():
                            logging.info(f'[INNER_LOOP] Encountered nan in GD!')
                            break
                        best_obs_param = b
                        if (i+1) % store_sz == 0:
                            gd_path.append(obs_rev_transform(best_obs_param))
                    gd_paths.append(gd_path)
                    
                best_obs_param = obs_rev_transform(best_obs_param)
                obs_param_candidates.append(best_obs_param)
                    
                t1 = time.time() - t1
                logging.info(f'[TIMING] [INNER_LOOP] Finding obs param candidate {r+1} (s) : {t1:.6f}')
                if self.obs_input_dim <= 10:
                    logging.info(f'[INNER_LOOP] Candidate for round {r+1} of {self.obs_setup_rounds} is {best_obs_param}.')
                
                time_elapsed = time.time() - exp_design_start_time
                if ((r + 1) >= self.min_obs_rounds) and (time_elapsed > self.obs_search_time_limit):
                    logging.info(f'[INNER_LOOP] Search time at {time_elapsed:6f}s, over limit of {self.obs_search_time_limit}s - terminating loop.')
                    break
                            
            obs_param_scores = []
            current_best_i = 0
            k_ = self.get_rng()
            
            for r, obs in enumerate(obs_param_candidates):
                s = criterion(obs, k_)[0]
                if jnp.isnan(s):
                    s = -jnp.inf
                obs_param_scores.append(s)
                logging.info(f'[INNER_LOOP] Score for inner loop {r+1} of {self.obs_setup_rounds} is {obs_param_scores[r]:.10f}.')
                if obs_param_scores[current_best_i] < s:
                    current_best_i = r
            
            t = time.time() - t
            logging.info(f'[TIMING] [OUTER_LOOP] iterate to find best obs param (s) e: {t:.6f}')
                            
            best_score = obs_param_scores[current_best_i]
            best_obs_param = obs_param_candidates[current_best_i]
            aux = {
                'exp_param': exp_design,
                'best_score': best_score,
                'best_obs_param': best_obs_param,
                'inv_prior_samples': true_inv_prior_samples,
                'obs_param_candidates': obs_param_candidates,
                'obs_param_scores': obs_param_scores,
                'inv_prior_samples': true_inv_prior_samples,
                'gd_paths': gd_paths,
                'forward_ensemble_nn_params': (self.forward_ens.params if self.use_pinns else None),
                'helper_fns': helper_fns,
                'criterion': criterion,
                'neg_criterion_val_and_grad': criterion_val_and_grad_modified,
                'exp_round_time_elapsed': time_elapsed,
            }
            
        else:
            
            ran_obs_params_untransformed = []
            ran_obs_params = []
            ran_obs_scores = []
            ran_obs_auxs = []
                
            for i in range(self.obs_setup_rounds):
                
                if (i+1) % 50 == 0:
                    logging.info(f'[INNER_LOOP] Running inner loop round {i+1} of {self.obs_setup_rounds}.')
                t = time.time()
                
                if i < 2:
                    
                    obs_design_candidate_untransformed = sample_from_uniform(
                        n=1, 
                        bounds=hypercube, 
                        sample_dim=self.obs_input_dim, 
                        rng=self.get_rng()
                    )[0]
                
                else:
                
                    train_X = torch.tensor(np.array(ran_obs_params_untransformed))
                    train_Y = torch.tensor(np.array(ran_obs_scores).reshape(-1, 1))
                    train_Y = (train_Y - torch.mean(train_Y)) / torch.std(train_Y)

                    model = SingleTaskGP(train_X, train_Y)
                    mll = ExactMarginalLogLikelihood(model.likelihood, model)
                    fit_gpytorch_mll(mll)
                    
                    sampler = StochasticSampler(sample_shape=torch.Size([128]))
                    if self.acq_fn == 'ucb':
                        q_fn = qUpperConfidenceBound(model, beta=1., sampler=sampler)
                    elif self.acq_fn == 'ei':
                        q_fn = qExpectedImprovement(model, best_f=train_Y.max(), sampler=sampler)
                    else:
                        raise ValueError(f'Invalid self.acq_fn {self.acq_fn}')
                    
                    obs_domain_np = np.array(hypercube)
                    Xinit = gen_batch_initial_conditions(q_fn, torch.tensor(obs_domain_np.T), q=1, num_restarts=25, raw_samples=500)
                    batch_candidates, batch_acq_values = gen_candidates_torch(
                        initial_conditions=Xinit,
                        acquisition_function=q_fn,
                        lower_bounds=torch.tensor(obs_domain_np[:,0]),
                        upper_bounds=torch.tensor(obs_domain_np[:,1]),
                    )
                    obs_design_candidate_untransformed = jnp.array(get_best_candidates(batch_candidates, batch_acq_values)[0].cpu().detach().numpy())
                
                obs_design_candidate = obs_rev_transform(obs_design_candidate_untransformed)
                score, aux = criterion(obs_design=obs_design_candidate)
                
                t = time.time() - t
                if (i+1) % 50 == 0:
                    if self.obs_input_dim <= 10:
                        logging.info(f'[INNER_LOOP] Candidate for round {i+1} of {self.obs_setup_rounds} is {obs_design_candidate}.')
                    logging.info(f'[INNER_LOOP] Score for inner loop {i+1} of {self.obs_setup_rounds} is {score:.10f}.')
                    logging.info(f'[TIMING] [INNER_LOOP] Running inner loop {i+1} of {self.exp_setup_rounds} (s) : {t:.6f}')  
                
                ran_obs_params_untransformed.append(obs_design_candidate_untransformed)
                ran_obs_params.append(obs_design_candidate)
                ran_obs_scores.append(score)
                ran_obs_auxs.append(aux)
                
                time_elapsed = time.time() - exp_design_start_time
                if ((i + 1) >= self.min_obs_rounds) and (time_elapsed > self.obs_search_time_limit):
                    logging.info(f'[INNER_LOOP] Search time at {time_elapsed:6f}s, over limit of {self.obs_search_time_limit}s after {i+1} rounds - terminating loop.')
                    break
                            
            best_obs_i = np.argmax(ran_obs_scores)
            best_score = ran_obs_scores[best_obs_i]
            best_obs_param = ran_obs_params[best_obs_i]
            aux = {
                'exp_param': exp_design,
                'best_score': best_score,
                'best_obs_param': best_obs_param,
                'ran_obs_params': ran_obs_params,
                'ran_obs_params_untransformed': ran_obs_params_untransformed,
                'ran_obs_scores': ran_obs_scores,
                'ran_obs_auxs': ran_obs_auxs,
                'inv_prior_samples': true_inv_prior_samples,
                'forward_ensemble_nn_params': (self.forward_ens.params if self.use_pinns else None),
                'helper_fns': helper_fns,
                'criterion': criterion,
                'exp_round_time_elapsed': time_elapsed,
            }
        
        return best_score, best_obs_param, aux
    
    def _inner_process_obs(self, best_exp, best_obs, observation, n_inv_ens=None):
        
        self.inverse_ens = PINNEnsemble(
            pde=self.pde, 
            pde_domain=self.pde_domain, 
            exp_design_fn=self.exp_design_fn, 
            obs_design_fn=self.obs_design_fn,
            inv_embedding=self.inv_embedding,
            inv_problem=True,
            rng=self.get_rng(),
            **self.pinn_ensemble_args
        )
                
        n_inv_ens = self.ensemble_size if n_inv_ens is None else n_inv_ens
        inv_prior_guess = self.sample_inv_param(n=n_inv_ens, rng=self.get_rng())
        
        if self.pinn_share_init:
            init_params = self._generate_shared_params(n=n_inv_ens, inv=inv_prior_guess)
            self.inverse_ens.reset()
            self.inverse_ens.prep_simulator(
                exp_params=best_exp, 
                inv_params_guesses=inv_prior_guess,
                new_nn_params=init_params,
                prior_reg=(self.ed_round * self.ensemble_inv_reg),
            )
            
        else:
            self.inverse_ens.reset()
            self.inverse_ens.prep_simulator(
                exp_params=best_exp, 
                inv_params_guesses=inv_prior_guess,
                prior_reg=(self.ed_round * self.ensemble_inv_reg),
            )

        xs_obs_split = jnp.repeat(best_obs[None,:], n_inv_ens, axis=0)
        ys_obs_split = jnp.repeat(observation[None,:], n_inv_ens, axis=0)
        for _ in tqdm.trange(self.inverse_ensemble_steps, mininterval=2):
            self.inverse_ens.step_opt(xs_obs_split, ys_obs_split)
            
        self.inv_samples = self.inverse_ens.params['inv']
        aux = {
            'best_exp': best_exp,
            'best_obs': best_obs,
            'observation': observation,
            'inverse_ens_params': self.inverse_ens.params,
        }
        return self.inv_samples, aux
        
