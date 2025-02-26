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
from ..utils.vmap_chunked import vmap_chunked
# from .ed_loop import ExperimentalDesign
from .criterion_based import CriterionBasedAbstractMethod
from .utils.obs_fn_helper import get_vmap_oracle
from ..utils import to_cpu, tree_stack, tree_unstack

# logger for this file
logging.getLogger().setLevel(logging.INFO)


class PINNFewStepInverseSolverTraining(CriterionBasedAbstractMethod):
    
    def __init__(self, simulator_xs, pde, pde_domain, exp_design_fn, obs_design_fn, 
                 inv_embedding, inv_param_in_domain, exp_in_domain, obs_in_domain,
                 inv_input_dim, exp_input_dim, obs_input_dim, obs_reading_count,
                 x_input_dim, y_output_dim, 
                 ensemble_size: int = 100, pinn_ensemble_args: Dict = dict(), 
                 ensemble_steps: int = 100000, do_fresh_pretraining: bool = False, net_perturb: bool = False, inv_perturb: bool = False, inv_perturb_val: float = 0.1,
                 pinn_share_init: bool = False, pinn_init_meta_rounds: int = 0, pinn_init_meta_steps: int = 1000, pinn_meta_eps: float = 0.1,
                 inverse_ensemble_pretraining_steps: int = 1000, inverse_ensemble_training_steps: int = 1,
                 pde_colloc_sample_num: int = 1000, icbc_colloc_sample_num: int = 100, use_implicit_diff: bool = False,
                 obs_optim_with_gd: bool = True, do_jit: bool = False, acq_fn: str = 'ucb', chunk_size: int = 100,
                 exp_setup_rounds: int = 10, obs_setup_rounds: int = 10, obs_search_time_limit: float = 3600., noise_std: float = 1e-3, 
                 obs_optim_use_lbfgs: bool = False, obs_optim_gd_params: Dict = dict(stepsize=1e-2, maxiter=1000, acceleration=True), 
                 obs_optim_grad_clip: float = None, obs_optim_grad_jitter: float = None, obs_optim_grad_zero_rate: float = None, min_obs_rounds: int = 3, 
                 gd_reps: int = 1, reg: float = 1e-12, seed: int = 0):
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
            min_obs_rounds=min_obs_rounds,
            noise_std=noise_std,
            obs_optim_with_gd=obs_optim_with_gd,
            obs_optim_gd_params=obs_optim_gd_params,
            obs_optim_grad_clip=obs_optim_grad_clip,
            obs_optim_grad_jitter=obs_optim_grad_jitter,
            obs_optim_grad_zero_rate=obs_optim_grad_zero_rate,
            obs_optim_use_lbfgs=obs_optim_use_lbfgs,
            obs_optim_gd_use_jacfwd=True,
            do_jit=do_jit,
            seed=seed,
        )
        
        self.reg = reg
        self.use_implicit_diff = use_implicit_diff
        self.do_fresh_pretraining = do_fresh_pretraining
        self.net_perturb = net_perturb
        self.inv_perturb = inv_perturb
        self.inv_perturb_val = inv_perturb_val
        self.inverse_ensemble_pretraining_steps = inverse_ensemble_pretraining_steps
        self.inverse_ensemble_training_steps = inverse_ensemble_training_steps
        self.pde_colloc_sample_num = pde_colloc_sample_num
        self.icbc_colloc_sample_num = icbc_colloc_sample_num
        self.chunk_size = chunk_size
        self.gd_reps = gd_reps
        
        assert not (self.obs_optim_with_gd and (self.inverse_ensemble_training_steps < 2))
            
    def _generate_criterion_inner(self, exp_design, true_inv_prior_samples):
        
        oracle_fn = jax.jit(self.forward_ens.generate_pred_function())
            
        updated_ensemble_args = {k: self.pinn_ensemble_args[k] for k in self.pinn_ensemble_args.keys()}
        updated_ensemble_args['n_pde_collocation_pts'] = self.pde_colloc_sample_num
        updated_ensemble_args['n_icbc_collocation_pts'] = self.icbc_colloc_sample_num
        inverse_ens = PINNEnsemble(
            pde=self.pde, 
            pde_domain=self.pde_domain, 
            exp_design_fn=self.exp_design_fn, 
            obs_design_fn=self.obs_design_fn,
            inv_embedding=self.inv_embedding,
            inv_problem=True,
            # maxiter=self.inverse_ensemble_training_steps,
            implicit_diff=self.use_implicit_diff,
            rng=self.get_rng(),
            **updated_ensemble_args
        )   
        
        inv_params_guesses = self.sample_inv_param(self.ensemble_size)
        if not self.do_fresh_pretraining:
            new_nn_params = {
                'net': self.forward_ens.params['net'],
                'inv': inv_params_guesses,
            }
        elif self.pinn_share_init:
            new_nn_params = self._generate_shared_params(n=self.ensemble_size, inv=inv_params_guesses)
        else:
            new_nn_params = None
        
        inverse_ens.reset()
        inverse_ens.prep_simulator(
            exp_params=exp_design, 
            inv_params_guesses=inv_params_guesses, 
            new_nn_params=new_nn_params,
            prior_reg=0.,
        )
                
        inverse_params_0_global = inverse_ens.params
        inverse_params_0_global = jax.tree_map(lambda x: jnp.array(x, dtype=jnp.zeros(1).dtype), inverse_params_0_global)
        # state_0 = jax.vmap(inverse_ens.solver.init_state)(inverse_params_0)
        # inverse_opt_states_0 = inverse_ens.opt_state_list
        # step_batch_fn = inverse_ens.step_batch
        loss_batch_fn = inverse_ens.loss_batch
        score_fn = jax.vmap(lambda inv1, inv2: self.compare_inv(inv1, inv2))
        oracle = get_vmap_oracle(self.forward_ens, self.obs_design_fn)
        
        @jax.jit
        def body_fun_all(val, ys_obs, obs_split):
            j, p, state = val
            # p = jax.lax.stop_gradient(p)
            # state = jax.lax.stop_gradient(state)
            j += 1
            if self.chunk_size > self.ensemble_size:
                p1, s1 = jax.vmap(inverse_ens.solver.update)(
                    p, state, obs_split, ys_obs, p['inv'])
            else:
                p1, s1 = vmap_chunked(inverse_ens.solver.update, in_axes=(0, 0, 0, 0, 0), chunk_size=self.chunk_size)(
                    p, state, obs_split, ys_obs, p['inv'])
            return (j, p1, s1)
        
        UNROLL = True if (self.obs_optim_with_gd or self.do_jit) else False
        
        def criterion_single(obs_design, rng=jax.random.PRNGKey(0)):
            
            if not self.do_fresh_pretraining:
                
                k1, k2 = jax.random.split(rng)
                
                if self.inv_perturb:
                    inv_guess = true_inv_prior_samples + self.inv_perturb_val * jax.random.normal(
                        key=k1,
                        shape=true_inv_prior_samples.shape,
                    )
                else:
                    inv_guess = sample_from_uniform(
                        n=self.ensemble_size, 
                        bounds=self.inv_param_in_domain, 
                        sample_dim=self.inv_input_dim, 
                        rng=rng
                    )
                    
                if self.net_perturb:
                    vector, unravel = ravel_pytree(self.forward_ens.params['net'])
                    noisy_vector = vector + self.inv_perturb_val * jax.random.normal(k2, vector.shape)
                    nn_param = unravel(vector)
                else:
                    nn_param = self.forward_ens.params['net']
                
                inverse_params_0 = {
                    'net': nn_param,
                    'inv': inv_guess,
                }
                
            elif self.pinn_share_init:
                inverse_params_0 = {
                    'net': inverse_params_0_global['net'],
                    'inv': sample_from_uniform(
                        n=self.ensemble_size, 
                        bounds=self.inv_param_in_domain, 
                        sample_dim=self.inv_input_dim, 
                        rng=rng
                    ),
                }
            else:
                invs = sample_from_uniform(
                    n=self.ensemble_size, 
                    bounds=self.inv_param_in_domain, 
                    sample_dim=self.inv_input_dim, 
                    rng=rng
                )
                net_params = []
                for i in range(self.ensemble_size):
                    rng, key_ = jax.random.split(rng)
                    params = inverse_ens.net.init(key_, inverse_ens.pde_collocation_pts[:1])
                    inv_params = invs[i]
                    net_params.append({'net': params, 'inv': inv_params})
                inverse_params_0 = tree_stack(net_params)
            state_0 = jax.vmap(inverse_ens.solver.init_state)(inverse_params_0)
            
            ys_obs = oracle(self.forward_ens.params['net'], obs_design)
            obs_split = jnp.repeat(obs_design[None,:], self.ensemble_size, axis=0)
            
            # add some noise
            rng, k_ = jax.random.split(rng)
            ys_obs = ys_obs + self.noise_std * jax.random.normal(key=k_, shape=ys_obs.shape)
            
            # separate body loop so gradient is stopped properly
            # ys_obs_1 = jax.lax.stop_gradient(oracle(self.forward_ens.params['net'], obs_design))
            # obs_split_1 = jax.lax.stop_gradient(jnp.repeat(obs_design[None,:], self.ensemble_size, axis=0))
            body_fun_1 = lambda val: body_fun_all(
                val, 
                jax.lax.stop_gradient(ys_obs), 
                jax.lax.stop_gradient(obs_split),
            )
            
            if (self.inverse_ensemble_pretraining_steps > 0) and self.do_jit:
                # use while loop to prevent unrolling
                _, diff_nn_params, diff_state = jax.lax.while_loop(
                    cond_fun=lambda val: val[0] < self.inverse_ensemble_pretraining_steps,
                    body_fun=body_fun_1,
                    init_val=(0, inverse_params_0, state_0),
                )
            else:
                diff_nn_params, diff_state = inverse_params_0, state_0
                for _ in range(self.inverse_ensemble_pretraining_steps):
                    _, diff_nn_params, diff_state = body_fun_1((_, diff_nn_params, diff_state))
                
            # dont consider gradient of first part
            diff_nn_params = jax.lax.stop_gradient(diff_nn_params)
            diff_state = jax.lax.stop_gradient(diff_state)
            
            # allow gradient tracking for second bit
            # ys_obs_2 = oracle(self.forward_ens.params['net'], obs_design)
            # obs_split_2 = jnp.repeat(obs_design[None,:], self.ensemble_size, axis=0)
            body_fun_2 = lambda val: body_fun_all(val, ys_obs, obs_split)
            
            if (self.inverse_ensemble_training_steps > 0) and UNROLL:
                _, nn_params, _ = jax.lax.fori_loop(
                    lower=0, 
                    upper=self.inverse_ensemble_training_steps, 
                    body_fun=lambda i, val: body_fun_2(val), 
                    init_val=(0, diff_nn_params, diff_state)
                )
                
                # def f_(carry, x):
                #     ans = body_fun_2(carry)
                #     return ans, ans[1]['inv'] - carry[1]['inv']
                
                # (_, nn_params, _), dinv_traj = jax.lax.scan(
                #     f=f_,
                #     init=(0, diff_nn_params, diff_state),
                #     xs=None,
                #     length=self.inverse_ensemble_training_steps,
                #     unroll=True,
                # )
                
            else:
                nn_params, state = diff_nn_params, diff_state
                for _ in range(self.inverse_ensemble_training_steps):
                    _, nn_params, state = body_fun_2((_, nn_params, state))
                
            # indiv_score = score_fn(nn_params["inv"], inverse_params_0["inv"])
            indiv_score = score_fn(nn_params["inv"], true_inv_prior_samples)
            
            # negative since requires maximisation
            score = - jnp.nanmean(indiv_score)
            # score = - jnp.nansum(jnp.log(indiv_score + self.reg))
            
            aux = {
                'true_inv_prior_samples': true_inv_prior_samples, 
                'nn_params_init': inverse_params_0,
                'nn_params_pt': diff_nn_params,
                'nn_params_final': nn_params,
                'indiv_score': indiv_score,
                # 'dinv_traj': dinv_traj,
            }
            return score, aux
        
        def criterion(obs_design, rng=jax.random.PRNGKey(0)):
            rng_split = jax.random.split(rng, self.gd_reps)
            scores, aux = jax.vmap(criterion_single, in_axes=(None, 0))(obs_design, rng_split)
            return jnp.nanmean(scores), {'scores': scores, 'train_aux': aux}
        
        return criterion, dict(
            forward_params=self.forward_ens.params,
            nn_params_init=inverse_params_0_global,
            criterion_single=criterion_single,
        )







# from functools import partial
# import os
# import pickle as pkl
# from collections.abc import MutableMapping
# from datetime import datetime
# from itertools import product
# from functools import partial
# from typing import Dict, Callable
# import time
# import logging

# import matplotlib.pyplot as plt
# import matplotlib.tri as tri
# import numpy as np
# import tqdm

# import jax
# import jax.numpy as jnp
# from jax import random, grad, vmap, jit, hessian, lax
# from jax.scipy.special import logsumexp
# from jax.example_libraries import optimizers
# from jax.nn import relu
# from jax.config import config
# from jax.flatten_util import ravel_pytree
# import optax
# import jaxopt
# from scipy.stats import spearmanr, pearsonr
# from scipy.interpolate import griddata
# import torch

# from gpytorch.mlls import ExactMarginalLogLikelihood
# from botorch.fit import fit_gpytorch_mll
# from botorch.models import SingleTaskGP
# from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound, qSimpleRegret
# from botorch.optim.initializers import gen_batch_initial_conditions, initialize_q_batch_nonneg
# from botorch.generation import gen_candidates_torch, get_best_candidates
# from botorch.sampling.stochastic_samplers import StochasticSampler

# from ..icbc_patch import generate_residue
# from ..models.pinn_ensemble import PINNEnsemble
# from ..utils import sample_from_uniform
# from ..utils.vmap_chunked import vmap_chunked
# # from .ed_loop import ExperimentalDesign
# from .criterion_based import CriterionBasedAbstractMethod
# from .utils.obs_fn_helper import get_vmap_oracle
# from ..utils import to_cpu, tree_stack, tree_unstack

# # logger for this file
# logging.getLogger().setLevel(logging.INFO)


# class PINNEnsembleDifferentiableInverseMethod(CriterionBasedAbstractMethod):
    
#     def __init__(self, simulator_xs, pde, pde_domain, exp_design_fn, obs_design_fn, 
#                  inv_embedding, inv_param_in_domain, exp_in_domain, obs_in_domain,
#                  inv_input_dim, exp_input_dim, obs_input_dim, obs_reading_count,
#                  x_input_dim, y_output_dim, 
#                  ensemble_size: int = 100, pinn_ensemble_args: Dict = dict(), 
#                  ensemble_steps: int = 100000, do_fresh_pretraining: bool = False,
#                  pinn_share_init: bool = False, pinn_init_meta_rounds: int = 0, pinn_init_meta_steps: int = 1000, pinn_meta_eps: float = 0.1,
#                  inverse_ensemble_pretraining_steps: int = 1000, inverse_ensemble_training_steps: int = 1,
#                  pde_colloc_sample_num: int = 1000, icbc_colloc_sample_num: int = 100, use_implicit_diff: bool = False,
#                  obs_optim_with_gd: bool = True, do_jit: bool = False, acq_fn: str = 'ucb', chunk_size: int = 100,
#                  exp_setup_rounds: int = 10, obs_setup_rounds: int = 10, obs_search_time_limit: float = 3600., noise_std: float = 1e-3, 
#                  obs_optim_use_lbfgs: bool = True, obs_optim_gd_params: Dict = dict(stepsize=1e-2, maxiter=1000, acceleration=True), obs_optim_grad_clip: float = None, min_obs_rounds: int = 3,
#                  gd_reps: int = 5, reg: float = 1e-12, seed: int = 0):
#         super().__init__(
#             simulator_xs=simulator_xs,
#             pde=pde, 
#             pde_domain=pde_domain, 
#             exp_design_fn=exp_design_fn, 
#             obs_design_fn=obs_design_fn,
#             inv_embedding=inv_embedding, 
#             inv_param_in_domain=inv_param_in_domain, 
#             exp_in_domain=exp_in_domain, 
#             obs_in_domain=obs_in_domain,
#             inv_input_dim=inv_input_dim, 
#             exp_input_dim=exp_input_dim, 
#             obs_input_dim=obs_input_dim, 
#             obs_reading_count=obs_reading_count,
#             x_input_dim=x_input_dim,
#             y_output_dim=y_output_dim,
#             use_pinns=True,
#             ensemble_size=ensemble_size,
#             ensemble_steps=ensemble_steps,
#             pinn_ensemble_args=pinn_ensemble_args,
#             pinn_share_init=pinn_share_init,
#             pinn_init_meta_rounds=pinn_init_meta_rounds,
#             pinn_init_meta_steps=pinn_init_meta_steps,
#             pinn_meta_eps=pinn_meta_eps,
#             acq_fn=acq_fn,
#             exp_setup_rounds=exp_setup_rounds,
#             obs_setup_rounds=obs_setup_rounds,
#             obs_search_time_limit=obs_search_time_limit,
#             min_obs_rounds=min_obs_rounds,
#             noise_std=noise_std,
#             obs_optim_with_gd=obs_optim_with_gd,
#             obs_optim_gd_params=obs_optim_gd_params,
#             obs_optim_grad_clip=obs_optim_grad_clip,
#             obs_optim_use_lbfgs=obs_optim_use_lbfgs,
#             obs_optim_gd_use_jacfwd=True,
#             do_jit=do_jit,
#             seed=seed,
#         )
        
#         self.reg = reg
#         self.use_implicit_diff = use_implicit_diff
#         self.do_fresh_pretraining = do_fresh_pretraining
#         self.inverse_ensemble_pretraining_steps = inverse_ensemble_pretraining_steps
#         self.inverse_ensemble_training_steps = inverse_ensemble_training_steps
#         self.pde_colloc_sample_num = pde_colloc_sample_num
#         self.icbc_colloc_sample_num = icbc_colloc_sample_num
#         self.chunk_size = chunk_size
#         self.gd_reps = gd_reps
        
#         assert not (self.obs_optim_with_gd and (self.inverse_ensemble_training_steps < 2))
            
#     def _generate_criterion_inner(self, exp_design, true_inv_prior_samples):
        
#         oracle_fn = jax.jit(self.forward_ens.generate_pred_function())
        
#         # if self.do_fresh_pretraining:
        
#         #     inverse_ens_pretrain = PINNEnsemble(
#         #         pde=self.pde, 
#         #         pde_domain=self.pde_domain, 
#         #         exp_design_fn=self.exp_design_fn, 
#         #         obs_design_fn=self.obs_design_fn,
#         #         inv_embedding=self.inv_embedding,
#         #         inv_problem=True,
#         #         rng=self.get_rng(),
#         #         **self.pinn_ensemble_args
#         #     )   
            
#         #     inv_prior_guess = self.sample_inv_param(n=self.ensemble_size, rng=self.get_rng())
#         #     inverse_ens_pretrain.reset()
#         #     inverse_ens_pretrain.prep_simulator(exp_params=exp_design, inv_params_guesses=inv_prior_guess)
            
#         #     # xs_obs = inverse_ens_pretrain.pde_collocation_pts[:self.pretraining_anchor_num]
#         #     # ys_obs = oracle_fn(xs_obs)
#         #     # xs_obs_split = jnp.repeat(xs_obs[None,:], self.ensemble_size, axis=0)
            
#         #     obs_samples_pretrain = sample_from_uniform(
#         #         n=self.ensemble_size,
#         #         bounds=self.obs_in_domain,
#         #         sample_dim=self.obs_input_dim,
#         #         rng=self.get_rng(),
#         #     )
#         #     ys_obs_pretrain = jnp.array([self.obs_design_fn(
#         #         self.forward_ens.generate_pred_function(i),
#         #         obs
#         #     ) for i, obs in enumerate(obs_samples_pretrain)])
            
            
#         #     for _ in range(self.inverse_ensemble_pretraining_steps):
#         #         inverse_ens_pretrain.step_opt(obs_samples_pretrain, ys_obs_pretrain)
            
#         #     new_nn_params = inverse_ens_pretrain.params
            
#         # else:
            
#         obs_samples_pretrain = None
#         ys_obs_pretrain = None
#         new_nn_params = {'net': self.forward_ens.params['net'], 'inv': true_inv_prior_samples}
#         new_nn_params = jax.tree_util.tree_map(lambda x: jax.random.permutation(key=self.get_rng(), x=x, axis=0), new_nn_params)
            
#         updated_ensemble_args = {k: self.pinn_ensemble_args[k] for k in self.pinn_ensemble_args.keys()}
#         updated_ensemble_args['n_pde_collocation_pts'] = self.pde_colloc_sample_num
#         updated_ensemble_args['n_icbc_collocation_pts'] = self.icbc_colloc_sample_num
#         inverse_ens = PINNEnsemble(
#             pde=self.pde, 
#             pde_domain=self.pde_domain, 
#             exp_design_fn=self.exp_design_fn, 
#             obs_design_fn=self.obs_design_fn,
#             inv_embedding=self.inv_embedding,
#             inv_problem=True,
#             # maxiter=self.inverse_ensemble_training_steps,
#             implicit_diff=self.use_implicit_diff,
#             rng=self.get_rng(),
#             **updated_ensemble_args
#         )   
        
#         inverse_ens.reset()
#         inverse_ens.prep_simulator(
#             exp_params=exp_design, 
#             inv_params_guesses=new_nn_params['inv'], 
#             new_nn_params=new_nn_params,
#             prior_reg=0.,
#         )
                
#         inverse_params_0_global = inverse_ens.params
#         inverse_params_0_global = jax.tree_map(lambda x: jnp.array(x, dtype=jnp.zeros(1).dtype), inverse_params_0_global)
#         # state_0 = jax.vmap(inverse_ens.solver.init_state)(inverse_params_0)
#         # inverse_opt_states_0 = inverse_ens.opt_state_list
#         # step_batch_fn = inverse_ens.step_batch
#         loss_batch_fn = inverse_ens.loss_batch
#         score_fn = jax.vmap(lambda inv1, inv2: self.compare_inv(inv1, inv2))
#         oracle = get_vmap_oracle(self.forward_ens, self.obs_design_fn)
        
#         @jax.jit
#         def body_fun_all(val, ys_obs, obs_split):
#             j, p, state = val
#             # p = jax.lax.stop_gradient(p)
#             # state = jax.lax.stop_gradient(state)
#             j += 1
#             if self.chunk_size > self.ensemble_size:
#                 p1, s1 = jax.vmap(inverse_ens.solver.update)(
#                     p, state, obs_split, ys_obs, p['inv'])
#             else:
#                 p1, s1 = vmap_chunked(inverse_ens.solver.update, in_axes=(0, 0, 0, 0, 0), chunk_size=self.chunk_size)(
#                     p, state, obs_split, ys_obs, p['inv'])
#             return (j, p1, s1)
        
#         UNROLL = True if (self.obs_optim_with_gd or self.do_jit) else False
        
#         def criterion_single(obs_design, rng=jax.random.PRNGKey(0)):
            
#             if self.pinn_share_init:
#                 inverse_params_0 = {
#                     'net': inverse_params_0_global['net'],
#                     'inv': sample_from_uniform(
#                         n=self.ensemble_size, 
#                         bounds=self.inv_param_in_domain, 
#                         sample_dim=self.inv_input_dim, 
#                         rng=rng
#                     ),
#                 }
#             else:
#                 invs = sample_from_uniform(
#                     n=self.ensemble_size, 
#                     bounds=self.inv_param_in_domain, 
#                     sample_dim=self.inv_input_dim, 
#                     rng=rng
#                 )
#                 net_params = []
#                 for i in range(self.ensemble_size):
#                     rng, key_ = jax.random.split(rng)
#                     params = inverse_ens.net.init(key_, inverse_ens.pde_collocation_pts[:1])
#                     inv_params = invs[i]
#                     net_params.append({'net': params, 'inv': inv_params})
#                 inverse_params_0 = tree_stack(net_params)
#             state_0 = jax.vmap(inverse_ens.solver.init_state)(inverse_params_0)
            
#             ys_obs = oracle(self.forward_ens.params['net'], obs_design)
#             obs_split = jnp.repeat(obs_design[None,:], self.ensemble_size, axis=0)
            
#             # separate body loop so gradient is stopped properly
#             # ys_obs_1 = jax.lax.stop_gradient(oracle(self.forward_ens.params['net'], obs_design))
#             # obs_split_1 = jax.lax.stop_gradient(jnp.repeat(obs_design[None,:], self.ensemble_size, axis=0))
#             body_fun_1 = lambda val: body_fun_all(
#                 val, 
#                 jax.lax.stop_gradient(ys_obs), 
#                 jax.lax.stop_gradient(obs_split),
#             )
            
#             if (self.inverse_ensemble_pretraining_steps > 0) and self.do_jit:
#                 # use while loop to prevent unrolling
#                 _, diff_nn_params, diff_state = jax.lax.while_loop(
#                     cond_fun=lambda val: val[0] < self.inverse_ensemble_pretraining_steps,
#                     body_fun=body_fun_1,
#                     init_val=(0, inverse_params_0, state_0),
#                 )
#             else:
#                 diff_nn_params, diff_state = inverse_params_0, state_0
#                 for _ in range(self.inverse_ensemble_pretraining_steps):
#                     _, diff_nn_params, diff_state = body_fun_1((_, diff_nn_params, diff_state))
                
#             # dont consider gradient of first part
#             diff_nn_params = jax.lax.stop_gradient(diff_nn_params)
#             diff_state = jax.lax.stop_gradient(diff_state)
            
#             # allow gradient tracking for second bit
#             # ys_obs_2 = oracle(self.forward_ens.params['net'], obs_design)
#             # obs_split_2 = jnp.repeat(obs_design[None,:], self.ensemble_size, axis=0)
#             body_fun_2 = lambda val: body_fun_all(val, ys_obs, obs_split)
            
#             if (self.inverse_ensemble_training_steps > 0) and UNROLL:
#                 _, nn_params, _ = jax.lax.fori_loop(
#                     lower=0, 
#                     upper=self.inverse_ensemble_training_steps, 
#                     body_fun=lambda i, val: body_fun_2(val), 
#                     init_val=(0, diff_nn_params, diff_state)
#                 )
                
#                 # def f_(carry, x):
#                 #     ans = body_fun_2(carry)
#                 #     return ans, ans[1]['inv'] - carry[1]['inv']
                
#                 # (_, nn_params, _), dinv_traj = jax.lax.scan(
#                 #     f=f_,
#                 #     init=(0, diff_nn_params, diff_state),
#                 #     xs=None,
#                 #     length=self.inverse_ensemble_training_steps,
#                 #     unroll=True,
#                 # )
                
#             else:
#                 nn_params, state = diff_nn_params, diff_state
#                 for _ in range(self.inverse_ensemble_training_steps):
#                     _, nn_params, state = body_fun_2((_, nn_params, state))
                
#             # indiv_score = score_fn(nn_params["inv"], inverse_params_0["inv"])
#             indiv_score = score_fn(nn_params["inv"], true_inv_prior_samples)
            
#             # negative since requires maximisation
#             # score = - jnp.nanmean(indiv_score)
#             score = - jax.nn.logsumexp(indiv_score, return_sign=False)
#             # score = - jnp.nansum(jnp.log(indiv_score + self.reg))
            
#             aux = {
#                 'true_inv_prior_samples': true_inv_prior_samples, 
#                 'nn_params_init': inverse_params_0,
#                 'nn_params_pt': diff_nn_params,
#                 'nn_params_final': nn_params,
#                 'indiv_score': indiv_score,
#                 # 'dinv_traj': dinv_traj,
#             }
#             return score, aux
        
#         def criterion(obs_design, rng=jax.random.PRNGKey(0)):
#             rng_split = jax.random.split(rng, self.gd_reps)
#             scores, aux = jax.vmap(criterion_single, in_axes=(None, 0))(obs_design, rng_split)
#             return jnp.mean(scores), {'scores': scores, 'train_aux': aux}
        
#         if self.do_jit:
#             criterion = jax.jit(criterion)
        
#         return criterion, dict(
#             forward_params=self.forward_ens.params,
#             nn_params_init=inverse_params_0_global,
#             criterion_single=criterion_single,
#         )
