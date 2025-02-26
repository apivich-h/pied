from functools import partial
import os
import pickle as pkl
from collections.abc import MutableMapping
from datetime import datetime
from itertools import product
from functools import partial
from typing import Dict
import time
import logging

import jax
import jax.numpy as jnp

from ..utils import sample_from_uniform

# logger for this file
logging.getLogger().setLevel(logging.INFO)


class ExperimentalDesign:
    
    def __init__(self, simulator_xs, pde, pde_domain, exp_design_fn, obs_design_fn, 
                 inv_embedding, inv_param_in_domain, exp_in_domain, obs_in_domain,
                 inv_input_dim, exp_input_dim, obs_input_dim, obs_reading_count,
                 x_input_dim, y_output_dim, seed=0):
        self.simulator_xs = simulator_xs
        self.pde = pde
        self.pde_domain = pde_domain
        self.inv_embedding = inv_embedding
        self.exp_design_fn = exp_design_fn
        self.obs_design_fn = obs_design_fn
        self.inv_input_dim = inv_input_dim
        self.exp_input_dim = exp_input_dim
        self.obs_input_dim = obs_input_dim
        self.obs_reading_count = obs_reading_count
        self.x_input_dim = x_input_dim
        self.y_output_dim = y_output_dim
        self.inv_param_in_domain = inv_param_in_domain
        self.exp_in_domain = exp_in_domain
        self.obs_in_domain = obs_in_domain
        
        self.ed_round = 0
        self.awaiting_experiment = False
        self.ran_experiment_setup = []
        self.ran_experiment_setup_aux = []
        self.ran_experiment_obs = []
        self.ran_experiment_inverse_guesses = []
        self.ran_experiment_inverse_aux = []
        self.rng = jax.random.PRNGKey(seed)
        
    def get_rng(self, n=None):
        self.rng, r_ = jax.random.split(self.rng)
        if n is None:
            return r_
        else:
            return jax.random.split(r_, num=n)
        
    def sample_inv_param(self, n, rng=None):
        if rng is None:
            rng = self.get_rng()
        if self.ed_round <= 0:
            return sample_from_uniform(
                n=n, 
                bounds=self.inv_param_in_domain, 
                sample_dim=self.inv_input_dim, 
                rng=rng
            )
        else:
            return self._inner_sample_inv_param(n, rng)
    
    def _inner_sample_inv_param(self, n, rng):
        raise NotImplementedError
        
    def run_experiment_round(self, given_exp_design=None, given_obs_design=None):
        
        assert not self.awaiting_experiment
        t_overall = time.time()
        best_exp, best_obs, aux = self._inner_experiment_round(given_exp_design=given_exp_design, given_obs_design=given_obs_design)
        t_overall = time.time() - t_overall
        logging.info(f'[TIMING] ED loop round {self.ed_round} (s) : {t_overall:.6f}')
        
        self.ran_experiment_setup.append((best_exp, best_obs))
        self.ran_experiment_setup_aux.append(aux)
        self.ran_experiment_obs.append(None)
        self.awaiting_experiment = True
        
        return best_exp, best_obs, aux
    
    def _inner_experiment_round(self, given_exp_design=None, given_obs_design=None):
        raise NotImplementedError
    
    def process_observation(self, observation, n_inv=None):
        
        t = time.time()
        assert self.awaiting_experiment, 'Need to call run_experiment_round() first.'
        
        self.ran_experiment_obs[self.ed_round] = observation
        best_exp, best_obs = self.ran_experiment_setup[self.ed_round]
        
        inv_samples, aux = self._inner_process_obs(best_exp=best_exp, best_obs=best_obs, observation=observation, n_inv_ens=n_inv)
        t = time.time() - t
        logging.info(f'[TIMING] [INNER_LOOP] Training inverse ensemble (s) : {t:.6f}')
        self.ran_experiment_inverse_guesses.append(inv_samples)
        self.ran_experiment_inverse_aux.append(aux)
        self.ed_round += 1
        self.awaiting_experiment = False
        
        return inv_samples, aux
    
    def _inner_process_obs(self, best_exp, best_obs, observation, n_inv_ens=None):
        raise NotImplementedError
    
    def compare_inv(self, inv1, inv2):
        return jnp.linalg.norm(self.inv_embedding(inv1) - self.inv_embedding(inv2))




# from functools import partial
# import os
# import pickle as pkl
# from collections.abc import MutableMapping
# from typing import Dict, Any, Callable, List, Type
# import asyncio
# import time
# import logging

# import matplotlib.pyplot as plt
# import matplotlib.tri as tri
# import numpy as np
# import tqdm

# import jax
# import jax.numpy as jnp
# from jax.scipy.special import logsumexp
# from jax.experimental.jax2tf import call_tf
# import flax
# from flax import linen as nn
# import optax
# import jaxopt

# import emcee
# # from gpflow.kernels import SquaredExponential, Matern12, Matern32, Matern52
# # import gpjax as gpx
# from ax.service.ax_client import AxClient, ObjectiveProperties
# from ax.modelbridge.registry import Cont_X_trans, Models, Y_trans
# from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
# from ax.models.torch.botorch_modular.model import BoTorchModel
# from ax.models.torch.botorch_modular.surrogate import Surrogate
# from botorch.models.gp_regression import FixedNoiseGP
# from botorch.acquisition import UpperConfidenceBound, qExpectedImprovement, qNoisyExpectedImprovement

# from . import deepxde as dde

# # from .gpflow_sampling.sampling import priors

# from .pde_sampler import PDESampler, PINNEnsemble
# from .icbc_patch import generate_residue, get_corresponding_y
# from .utils import to_cpu, tree_stack, tree_unstack, sample_from_uniform

# from torch.utils.tensorboard import SummaryWriter

# # logger for this file
# logging.getLogger().setLevel(logging.INFO)


# class BEDTrainLoop:
    
#     def __init__(self, simulator_type: str, simulator_setup: Dict,
#                  obs_design_fn: Callable, obs_in_domain: jnp.ndarray, exp_param_setup: List[Dict], 
#                  inv_param_in_domain: jnp.ndarray, inv_param_prior_log_pdf: Callable = None, 
#                  sim_count: int = 100, bo_rounds: int = 20, bo_surrogate_kwargs: Dict = None, bo_acq_fn: Type = qNoisyExpectedImprovement,
#                  posterior_estimation_loops: int = 5, posterior_surrogate_kwargs: Dict = None, posterior_acq_fn: Type = qNoisyExpectedImprovement,
#                  scoring_function: str = 'infogain', scoring_function_reps: int = 100, nwalkers_beta: int = 10,
#                  obs_optim_method: str = 'adam', obs_optim_args: Dict = None, obs_optim_steps: int = 5000, obs_optim_rounds: int = 10,
#                  seed: int = 42):
        
#         self.simulator_type = simulator_type
#         self.simulator_setup = simulator_setup
        
#         self.obs_design_fn = obs_design_fn  # takes in observation design params array, returns points to observe functions at
#         self.obs_in_domain = obs_in_domain
#         self.obs_in_dim = obs_in_domain.shape[0]
#         self.exp_param_setup = exp_param_setup
        
#         self.inv_param_prior_log_pdf = (lambda b: jnp.array(0.)) if (inv_param_prior_log_pdf is None) else inv_param_prior_log_pdf
#         self.inv_param_in_domain = inv_param_in_domain
                
#         self.sim_count = sim_count
#         self.nwalkers_beta = nwalkers_beta
#         self.scoring_function = scoring_function
#         self.scoring_function_reps = scoring_function_reps
#         self.bo_rounds = bo_rounds
#         self.bo_surrogate_kwargs = {'botorch_model_class': FixedNoiseGP} if (bo_surrogate_kwargs is None) else bo_surrogate_kwargs
#         self.bo_acq_fn = bo_acq_fn
#         self.posterior_estimation_loops = posterior_estimation_loops
#         self.posterior_surrogate_kwargs = {'botorch_model_class': FixedNoiseGP} if (posterior_surrogate_kwargs is None) else posterior_surrogate_kwargs
#         self.posterior_acq_fn = posterior_acq_fn
            
#         self.obs_optim_steps = obs_optim_steps
#         self.obs_optim_rounds = obs_optim_rounds
#         if obs_optim_args is None:
#             self.obs_optim_method = 'adam'
#             self.obs_optim_args = dict(learning_rate=0.01)
#         else:
#             self.obs_optim_method = obs_optim_method
#             self.obs_optim_args = obs_optim_args
            
#         self.rng = jax.random.PRNGKey(seed)
#         self.rounds_elapsed = 0
#         self.sim_ensemble: PDESampler = None
#         self.round_trials = None
#         self.past_snapshots = []
#         self.exp_trials = []
#         self.current_prior = self.inv_param_prior_log_pdf
#         self.current_prior_bounded = None
#         self.current_prior_sample_method = 'uniform' if (inv_param_prior_log_pdf is None) else 'mcmc'
#         self.awaiting_experiment = False
#         self._bo_loop_bed = None
#         self._bo_loop_posterior = None
        
#     def sample_inv_param(self, n):
        
#         if self.current_prior_sample_method == 'uniform':
#             # if uniform sampling, then don't need fancy MCMC methods
#             ndim = self.inv_param_in_domain.shape[0]
#             return self.inv_param_in_domain[None,:,0] + np.random.rand(n, ndim) * (self.inv_param_in_domain[None,:,1] - self.inv_param_in_domain[None,:,0])
            
#         def log_prob(beta):
#             if (self.inv_param_in_domain[:,0] <= beta).all() and (beta <= self.inv_param_in_domain[:,1]).all():
#                 return self.current_prior(beta)
#             else:
#                 return - np.inf
            
#         self.current_prior_bounded = log_prob
        
#         ndim = self.inv_param_in_domain.shape[0]
#         sampler = emcee.EnsembleSampler(self.nwalkers_beta, ndim, log_prob)
#         p0 = self.inv_param_in_domain[None,:,0] + np.random.rand(self.nwalkers_beta, ndim) * (self.inv_param_in_domain[None,:,1] - self.inv_param_in_domain[None,:,0])

#         # burn-in
#         burn_factor = 100
#         state = sampler.run_mcmc(p0, burn_factor * n, skip_initial_state_check=True)
#         sampler.reset()
        
#         # actual samples 
#         sampler.run_mcmc(state, burn_factor * (1 + n // self.nwalkers_beta), skip_initial_state_check=True)
#         s = sampler.get_chain(thin=burn_factor, flat=True)[-n:,:]
#         return s
            
#     def _generate_solver(self, value_and_grad):
#         if self.obs_optim_method == 'adam':
#             opt = optax.adam(**self.obs_optim_args)
#             solver = jaxopt.OptaxSolver(opt=opt, fun=value_and_grad, value_and_grad=True)
#         elif self.obs_optim_method == 'lbfgs':
#             solver = jaxopt.LBFGS(fun=value_and_grad, value_and_grad=True, jit=True, **self.obs_optim_args)
#         else:
#             raise ValueError(f'Invalid obs_optim_method: {self.obs_optim_method}')
#         return solver
            
#     def _reset_sim_ensembles(self, beta_list):
#         if self.simulator_type == 'pinn':
#             self.sim_ensemble = PINNEnsemble(**self.simulator_setup)
#         else:
#             raise ValueError(f'Invalid simulator type - {self.simulator_type}')
#         self.sim_ensemble.set_inv_params(beta_list)
            
#     def _prep_ensemble(self, inv_param, exp_design_param):
#         t = time.time()
#         self.sim_ensemble.reset()
#         self.sim_ensemble.set_inv_params(inv_param)
#         self.sim_ensemble.set_exp_params(exp_design_param)
#         self.sim_ensemble.prep_simulator()
#         t = time.time() - t
#         logging.info(f'[TIMING] Prep ensemble for given exp_design_param : {t:.6f}s.')
        
#     def _generate_criterion(self):
        
#         if 'infogain' in self.scoring_function:
#             # define the loss function for the round
#             def criterion(obs_design_param, rng):
#                 n = self.sim_count
#                 xs = self.obs_design_fn(obs_design_param)
#                 ys = self.sim_ensemble.sample(xs, rng)
#                 llhs = jax.vmap(self.sim_ensemble.log_likelihood, in_axes=(None, 0))(xs, ys)
#                 return - (jnp.mean(jnp.diag(llhs)) - jnp.mean(logsumexp(llhs, axis=1)))  # negate because GD module will minimise function
#                 # original, non-array version
#                 # vals = 0.
#                 # for i in range(n):
#                 #     # ignore some dependency on n in likelihood term
#                 #     # ignore 1/M factor in front of prior estimate because treated as constant anyway
#                 #     vals += self.sim_ensemble[i][1].log_likelihood(xs, ys[i])
#                 #     vals -= logsumexp(jnp.stack([self.sim_ensemble[j][1].log_likelihood(xs, ys[i]) for j in range(n)]))
#                 # return - vals
        
#         else:
#             raise ValueError(f'Invalid {self.scoring_function}')
        
#         return criterion
                
#     def _eval_function(self, inv_param_sample, exp_design_param, given_obs_design=None):
        
#         # train the ensemble on the experiment design first
#         self._prep_ensemble(inv_param=inv_param_sample, exp_design_param=exp_design_param)
#         criterion = jax.jit(self._generate_criterion())
#         criterion_repeat = jax.jit(lambda design, rngs: jnp.mean(jax.vmap(criterion, in_axes=(None, 0))(design, rngs)))  # repeat criterion multiple times
        
#         if given_obs_design is None:
#             obs_guess = (self.obs_in_domain[:,0])[None, :] + jnp.array(np.random.rand(self.obs_optim_rounds, self.obs_in_dim)) * (self.obs_in_domain[:,1] - self.obs_in_domain[:,0])[None, :]
#             solver = self._generate_solver(value_and_grad=jax.jit(jax.value_and_grad(criterion_repeat)))
#             # opt_state = tree_stack([solver.init_state(guess) for guess in obs_guess])
#             opt_state = jax.vmap(solver.init_state)(obs_guess)
            
#             step_batch = jax.jit(jax.vmap(solver.update, in_axes=(0, 0, None)))
            
#             logging.info(f'Finding optimal observation for exp design')
#             for _ in tqdm.trange(self.obs_optim_steps):
#                 self.rng, key_ = jax.random.split(self.rng)
#                 key_ = jax.random.split(key_, num=self.scoring_function_reps)
#                 obs_guess, opt_state = step_batch(obs_guess, opt_state, key_)
#                 obs_guess = jax.vmap(lambda g_: jnp.clip(g_, self.obs_in_domain[:,0], self.obs_in_domain[:,1]))(obs_guess)
                
#             self.rng, key_ = jax.random.split(self.rng)
#             key_ = jax.random.split(key_, num=self.scoring_function_reps)
#             final_scores = [criterion_repeat(design, key_) for design in obs_guess]
#             best_final_score = np.argmin(final_scores)
#             obs_guess = obs_guess[best_final_score]
                
#         # if given_obs_design is None:
#         #     obs_guess = self.obs_in_domain[:,0] + jnp.array(np.random.rand(self.obs_in_dim)) * (self.obs_in_domain[:,1] - self.obs_in_domain[:,0])
#         #     solver = self._generate_solver(value_and_grad=jax.jit(jax.value_and_grad(criterion_repeat)))
#         #     opt_state = solver.init_state(obs_guess)
#         #     logging.info(f'Finding optimal observation for exp design')
#         #     for _ in tqdm.trange(self.obs_optim_steps):
#         #         self.rng, key_ = jax.random.split(self.rng)
#         #         key_ = jax.random.split(key_, num=self.scoring_function_reps)
#         #         obs_guess, opt_state = solver.update(obs_guess, opt_state, rng=key_)
#         #         obs_guess = jnp.clip(obs_guess, self.obs_in_domain[:,0], self.obs_in_domain[:,1])
            
#         else:
#             logging.info(f'Observation design fixed at {given_obs_design}.')
#             obs_guess = given_obs_design
        
#         self.rng, key_ = jax.random.split(self.rng)
#         criterion_trials = jax.vmap(criterion, in_axes=(None, 0))(obs_guess, jax.random.split(key_, num=self.scoring_function_reps))
#         return {
#             'crit_mean': float(jnp.mean(criterion_trials)),
#             'crit_std': float(jnp.std(criterion_trials)),
#             'crit_trials': criterion_trials,
#             'best_obs': obs_guess,
#         }
        
#     def run_experiment_round(self, given_exp_design=None, given_obs_design=None):
        
#         assert not self.awaiting_experiment, 'Need to call process_prev_experiment(...) first.'
        
#         # initialise PINN ensembles
#         logging.info(f'Sampling {self.sim_count} inverse params for ensemble...')
#         t = time.time()
#         inv_param_sample = self.sample_inv_param(n=self.sim_count)
#         t = time.time() - t
#         logging.info(f'[TIMING] Sampling inverse params : {t:.6f}s.')
#         logging.info(f'Done sampling {self.sim_count} inverse params for ensemble.')
        
#         logging.info(f'Preparing ensemble...')
#         t = time.time()
#         self._reset_sim_ensembles(inv_param_sample)
#         t = time.time() - t
#         logging.info(f'[TIMING] Setting inverse params for ensemble : {t:.6f}s.')
#         logging.info(f'Done preparing ensemble.')
        
#         self.round_trials = dict()
        
#         if given_exp_design is None:
            
#             gs = GenerationStrategy(steps=[
#                 GenerationStep(  # Initialization step - to generate just one random point
#                     model=Models.SOBOL,
#                     num_trials=1,
#                     min_trials_observed=1,
#                 ),
#                 GenerationStep(
#                     model=Models.BOTORCH_MODULAR,
#                     num_trials=-1,
#                     model_kwargs={
#                         "surrogate": Surrogate(**self.bo_surrogate_kwargs),
#                         "botorch_acqf_class": self.bo_acq_fn,
#                     }
#                 )
#             ])
            
#             ax_client = AxClient(generation_strategy=gs)
#             ax_client.create_experiment(
#                 name=f'ED_round_{self.rounds_elapsed}',  # The name of the experiment.
#                 parameters=self.exp_param_setup,
#                 objectives={'criterion': ObjectiveProperties(minimize=False)},  # The objective name and minimization setting.
#             )
                        
#             for i in range(self.bo_rounds):
#                 parameters, trial_index = ax_client.get_next_trial()
#                 crit_scores = self._eval_function(inv_param_sample=inv_param_sample, exp_design_param=parameters, given_obs_design=given_obs_design)
#                 self.round_trials[trial_index] = {
#                     'exp_design': parameters, 
#                     'obs_design': crit_scores['best_obs'], 
#                     'crit_scores': crit_scores,
#                     'sim_inv_param': self.sim_ensemble.inv_params,
#                     'simulator': self.sim_ensemble.generate_intermediate_info(),
#                 }
#                 ax_client.complete_trial(trial_index=trial_index, raw_data={'criterion': (crit_scores['crit_mean'], crit_scores['crit_std'])})
            
#             best_exp_design, _ = ax_client.get_best_parameters()
#             self._bo_loop_bed = ax_client
        
#         else:
#             # use the given exp_design directly
#             logging.info(f'Experiment design fixed at {given_exp_design}.')
#             best_exp_design = given_exp_design
        
#         best_obs_design = self._eval_function(inv_param_sample=inv_param_sample, exp_design_param=best_exp_design)['best_obs']
        
#         self.exp_trials.append({
#             'exp_design': best_exp_design, 
#             'obs_design': best_obs_design,
#             'observations': None,
#             'inv_param_posterior': [],
#         })
#         self.awaiting_experiment = True
    
#     def get_last_design(self):
#         return self.exp_trials[-1]['exp_design'], self.exp_trials[-1]['obs_design']
    
#     def _get_llh(self, exp_design, obs_design, ys, inv_params):
#         xs = self.obs_design_fn(obs_design)
#         t = time.time()
#         self._prep_ensemble(inv_param=inv_params, exp_design_param=exp_design)
#         t = time.time() - t
#         logging.info(f'[TIMING] Prep ensemble for computing inverse param : {t:.6f}s.')
#         return self.sim_ensemble.log_likelihood(xs, ys)
    
#     def _get_posterior(self, exp_design, obs_design, ys, inv_params):
#         llh = self._get_llh(exp_design, obs_design, ys, inv_params)
#         prior = jax.vmap(self.current_prior)(inv_params)
#         assert llh.shape == prior.shape
#         return prior + llh
    
#     def process_prev_experiment(self, observations):
        
#         # store values
#         assert self.awaiting_experiment, 'Need to call run_experiment_round() first.'
#         self.exp_trials[-1]['observations'] = observations
#         self.exp_trials[-1]['inv_param_posterior'] = []
        
#         # update distribution of inverse param
#         exp_design, obs_design = self.get_last_design()
        
#         gs = GenerationStrategy(steps=[
#             GenerationStep(
#                 model=Models.SOBOL,
#                 num_trials=self.sim_count,
#             ),
#             GenerationStep(
#                 model=Models.BOTORCH_MODULAR,
#                 num_trials=-1,
#                 model_kwargs={
#                     "surrogate": Surrogate(**self.posterior_surrogate_kwargs),
#                     "botorch_acqf_class": self.posterior_acq_fn,
#                 }
#             )
#         ])
        
#         inv_bo_params = [
#             {
#                 'name': f'inv_{i}', 
#                 'type': 'range', 
#                 'value_type': 'float', 
#                 'bounds': [float(b) for b in self.inv_param_in_domain[i]],
#             } for i in range(self.inv_param_in_domain.shape[0])
#         ]
        
#         def inv_param_to_dict(invps):
#             return [{f'inv_{i}': float(p[i]) for i in range(len(p))} for p in invps]
        
#         def dict_to_inv_param(dicts):
#             return jnp.array([[d[f'inv_{i}'] for i in range(len(d))] for d in dicts])
        
#         ax_client = AxClient(generation_strategy=gs)
#         ax_client.create_experiment(
#             name=f'Posterior_round_{self.rounds_elapsed}',  # The name of the experiment.
#             parameters=inv_bo_params,
#             objectives={'posterior': ObjectiveProperties(minimize=False)},  # The objective name and minimization setting.
#         )
        
#         inv_params_round = []
#         llh_computed_round = []
        
#         # for first round, just use the current set of inv params
#         existing_inv_param = self.sim_ensemble.inv_params
#         llh_score = self._get_posterior(exp_design=exp_design, obs_design=obs_design, ys=observations, inv_params=existing_inv_param)
#         for r, inv_param_dict in enumerate(inv_param_to_dict(existing_inv_param)):
#             ax_client.attach_trial(parameters=inv_param_dict)
#             ax_client.complete_trial(trial_index=r, raw_data={'posterior': (float(llh_score[r]), 0.)})
#             self.exp_trials[-1]['inv_param_posterior'].append((existing_inv_param[r], float(llh_score[r])))
#             inv_params_round.append(existing_inv_param[r])
#             llh_computed_round.append(float(llh_score[r]))
        
#         for i in range(self.posterior_estimation_loops):
            
#             # get plausible MAP inv param from BO loop
#             inv_guesses = ax_client.get_next_trials(self.sim_count)[0]
#             inv_param_array = dict_to_inv_param(list(inv_guesses.values()))
#             llh_score = self._get_posterior(exp_design=exp_design, obs_design=obs_design, ys=observations, inv_params=inv_param_array)
            
#             for r, trial_index in enumerate(inv_guesses.keys()):
#                 ax_client.complete_trial(trial_index=trial_index, raw_data={'posterior': (float(llh_score[r]), 0.)})
#                 self.exp_trials[-1]['inv_param_posterior'].append((inv_param_array[r], float(llh_score[r])))
#                 inv_params_round.append(inv_param_array[r])
#                 llh_computed_round.append(float(llh_score[r]))
        
#         # computing the posterior based on the GP
#         # TODO: make this part more customisable probably
#         inv_params_round = jnp.array(inv_params_round)
#         llh_computed_round = jnp.array(llh_computed_round).reshape(-1, 1)
#         llh_min = jnp.min(llh_computed_round)
#         llh_computed_round = llh_computed_round - llh_min
        
#         # mean = gpx.mean_functions.Zero()
#         # kernel = gpx.kernels.Matern52()
#         # prior = gpx.Prior(mean_function=mean, kernel=kernel)
#         # D = gpx.Dataset(X=inv_params_round, y=llh_computed_round)
#         # likelihood = gpx.Gaussian(num_datapoints=D.n).replace_trainable()
#         # no_opt_posterior = prior * likelihood
#         # negative_mll = gpx.objectives.ConjugateMLL(negative=True)
#         # opt_posterior, history = gpx.fit(
#         #     model=no_opt_posterior,
#         #     objective=negative_mll,
#         #     train_data=D,
#         #     optim=optax.adam(learning_rate=1.),
#         #     num_iters=10000,
#         #     safe=True,
#         #     key=self.rng,
#         # )
        
#         # def get_posterior(b):
#         #     opt_latent_dist = opt_posterior.predict(b.reshape(-1, 1), train_data=D)
#         #     opt_predictive_dist = opt_posterior.likelihood(opt_latent_dist)
#         #     return opt_predictive_dist.mean()[0] + llh_min
        
#         # self.current_prior = jax.jit(get_posterior)
#         # self.current_prior_bounded = None
#         # self.current_prior_sample_method == 'mcmc'
        
#         # self._bo_loop_posterior = ax_client
#         # self.awaiting_experiment = False
#         # self.rounds_elapsed += 1
