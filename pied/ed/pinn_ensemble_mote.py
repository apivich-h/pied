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
import flax

from ..icbc_patch import generate_residue
from ..models.pinn_ensemble import PINNEnsemble
from ..utils import sample_from_uniform
from ..utils.jax_utils import flatten, vmap_mjp, vmap_jmp, jacobian_outer_product
from ..utils.vmap_chunked import vmap_chunked
from .utils.obs_fn_helper import get_vmap_oracle
# from .ed_loop import ExperimentalDesign
from .criterion_based import CriterionBasedAbstractMethod
from ..utils import to_cpu, tree_stack, tree_unstack

# logger for this file
logging.getLogger().setLevel(logging.INFO)


class PINNModelTrainingEstimation(CriterionBasedAbstractMethod):
    
    def __init__(self, simulator_xs, pde, pde_domain, exp_design_fn, obs_design_fn,
                 inv_embedding, inv_param_in_domain, exp_in_domain, obs_in_domain,
                 inv_input_dim, exp_input_dim, obs_input_dim, obs_reading_count,
                 x_input_dim, y_output_dim, 
                 ensemble_size: int = 100, pinn_ensemble_args: Dict = dict(), 
                 pinn_share_init: bool = False, pinn_init_meta_rounds: int = 0, pinn_init_meta_steps: int = 1000, pinn_meta_eps: float = 0.1,
                 ensemble_steps: int = 100000, acq_fn: str = 'ucb', reg: float = 1e-6, 
                 inverse_ens_use_forward_params: bool = False, inverse_ensemble_pretraining_steps: int = 1000, 
                 inverse_ens_do_perturb: bool = False, inverse_ens_perturb_std: float = 0.1,
                 pde_colloc_sample_num: int = 1000, icbc_colloc_sample_num: int = 100, 
                 exp_setup_rounds: int = 10, obs_setup_rounds: int = 10, obs_search_time_limit: float = 3600., noise_std: float = 1e-3, 
                 obs_optim_gd_params: Dict = dict(stepsize=1e-2, maxiter=100, acceleration=True), obs_optim_use_lbfgs: bool = False,
                 obs_optim_grad_clip: float = None, obs_optim_grad_jitter: float = None, obs_optim_grad_zero_rate: float = None, min_obs_rounds: int = 3,
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
            obs_optim_with_gd=True,
            do_jit=True,
            obs_optim_use_lbfgs=obs_optim_use_lbfgs,
            obs_optim_gd_params=obs_optim_gd_params,
            obs_optim_grad_clip=obs_optim_grad_clip,
            obs_optim_grad_jitter=obs_optim_grad_jitter,
            obs_optim_grad_zero_rate=obs_optim_grad_zero_rate,
            min_obs_rounds=min_obs_rounds,
            seed=seed,
        )
        
        self.inverse_ensemble_pretraining_steps = inverse_ensemble_pretraining_steps
        self.inverse_ens_use_forward_params = inverse_ens_use_forward_params
        self.inverse_ens_do_perturb = inverse_ens_do_perturb
        self.inverse_ens_perturb_std = inverse_ens_perturb_std
        self.pde_colloc_sample_num = pde_colloc_sample_num
        self.icbc_colloc_sample_num = icbc_colloc_sample_num
        self.reg = reg
        
    def _generate_criterion_inner(self, exp_design, true_inv_prior_samples):
        
        # mock_param = {
        #     'net': inverse_ens.net.init(jax.random.PRNGKey(0), inverse_ens.pde_collocation_pts[:1]), 
        #     'inv': sample_from_uniform(
        #         n=1, 
        #         bounds=self.inv_param_in_domain, 
        #         sample_dim=self.inv_input_dim, 
        #         rng=jax.random.PRNGKey(0),
        #     )[0]
        # }
        # _, nn_param_unflatten_fn = flatten(mock_param)
        
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
            rng=self.get_rng(),
            **updated_ensemble_args
        )   
        
        inv_params_guesses = self.sample_inv_param(self.ensemble_size)
        if self.inverse_ens_use_forward_params:
            new_nn_params = {
                'net': self.forward_ens.params['net'],
                'inv': (true_inv_prior_samples if self.inverse_ens_do_perturb else inv_params_guesses),
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
        )
        
        modified_apply = inverse_ens.net.apply
        
        idxs_sample = jnp.array(np.random.choice(
            self.forward_ens.pde_collocation_pts.shape[0], 
            size=min(self.forward_ens.pde_collocation_pts.shape[0], self.pde_colloc_sample_num), 
            replace=False
        ))
        xs_colloc = self.forward_ens.pde_collocation_pts[idxs_sample]
        
        bc_fns = [fn for (fn, _) in self.forward_ens.exp_design_fn]
        bc_data = [xs[:min(xs.shape[0], self.icbc_colloc_sample_num)] for (_, xs) in self.forward_ens.exp_design_fn]
        
        @jax.jit
        def colloc_residues(params):
            nn_params = params['net']
            beta = params['inv']
            f_ = lambda xs: modified_apply(nn_params, xs)
            return {
                'pde': self.pde(xs_colloc, (f_(xs_colloc), f_), beta, exp_design)[0][:,0].reshape(-1) / (xs_colloc.shape[0] ** 0.5),
                'bcs': [
                    bc_fn(nn_params, modified_apply, self.forward_ens.exp_params, beta, bc_xs).reshape(-1) / (bc_xs.shape[0] ** 0.5)
                    for bc_fn, bc_xs in zip(bc_fns, bc_data)
                ]
            }
            
        @jax.jit
        def colloc_residue_stacked(params):
            res = colloc_residues(params)
            return jnp.concatenate([res['pde']] + res['bcs'])
        
        @jax.jit
        def anc_residue(params, obs, forward_params):
            reading_fn = lambda p: self.obs_design_fn((lambda xs: modified_apply(p, xs)), obs)
            return (reading_fn(params['net']) - reading_fn(forward_params['net'])).reshape(-1)
        
        def get_ntk_from_jac(jac1, jac2):
            # prods = [jnp.einsum('ijk,ljk->jil', jac1[k], jac2[k]) for k in jac1.keys()]
            # return sum(prods)
            prods = None
            for k in jac1.keys():
                m = jac1[k] @ jac2[k].T
                prods = m if (prods is None) else (prods + m)
            return prods

        def _flatten_dict(d, parent_key='', sep='_'):
            # https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
            # needed as neural network parameters are stored in a nested dictionary, e.g. {'params': {'dense': {'kernel': ...}}}
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, MutableMapping) or isinstance(v, flax.core.frozen_dict.FrozenDict):
                    items.extend(_flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        def get_jac(fn, p):
            dd = jax.jacrev(fn)(p)
            dd = _flatten_dict(dd)
            # currently only works for one-dimensional model outputs
            return {k: dd[k].reshape(dd[k].shape[0], -1) for k in dd.keys()}
        
        def jac_mult_fn(jac1, jac2):
            # prods = [jnp.einsum('ijk,ljk->jil', jac1[k], jac2[k]) for k in jac1.keys()]
            # return sum(prods)
            prods = None
            for k in jac1.keys():
                m = jac1[k] @ jac2[k].T
                prods = m if (prods is None) else (prods + m)
            return prods

        @jax.jit
        def body_fun_all(val, ys_obs, obs_split):
            j, p, state = val
            j += 1
            p1, s1 = jax.vmap(inverse_ens.solver.update)(p, state, obs_split, ys_obs, p['inv'])
            return (j, p1, s1)

        def single_network_criterion(obs_param, p, pntk, colloc_ntk, colloc_jac, colloc_res, forward_p, true_inv, rng):
            
            if self.inverse_ens_do_perturb:
                vector, unravel = flatten(p)
                noisy_vector = vector + self.inverse_ens_perturb_std * jax.random.normal(rng, vector.shape)
                p = unravel(vector)
                colloc_res = colloc_residue_stacked(p)
            
            anc_fn = lambda p_: anc_residue(params=p_, obs=obs_param, forward_params=forward_p)
            anc_jac = get_jac(anc_fn, pntk)
            anc_res = anc_fn(p)
            
            ntk_CC = colloc_ntk
            ntk_CA = jac_mult_fn(colloc_jac, anc_jac)
            ntk_AA = jac_mult_fn(anc_jac, anc_jac)
            ntk_matrix = jnp.block([[ntk_CC, ntk_CA], [ntk_CA.T, ntk_AA]])
            
            residue = jnp.block([[colloc_res.reshape(-1, 1)], [anc_res.reshape(-1, 1)]])
            ntk_inv = jnp.linalg.inv(ntk_matrix + self.reg * jnp.eye(ntk_matrix.shape[0]))
            residue_jac_invparam = jnp.block([[colloc_jac['inv']], [anc_jac['inv']]])
            
            inv0 = p['inv'].reshape(-1)
            dinv = - (residue_jac_invparam.T @ ntk_inv @ residue).reshape(-1)
            inv_converged = inv0 + dinv
                        
            aux = {
                'anc_res': anc_res,
                'anc_jac': anc_jac,
                'ntk_matrix': ntk_matrix,
                'residue': residue,
                'ntk_inv': ntk_inv,
                'residue_jac_invparam': residue_jac_invparam,
                'dinv': dinv.reshape(-1),
                'inv_converged': inv_converged.reshape(-1),
            }
            s = jnp.linalg.norm(self.inv_embedding(inv_converged) - self.inv_embedding(true_inv))
            return s, aux
        
        FOR_PARAMS_ALL = self.forward_ens.params
        INV_PARAMS_ALL = inverse_ens.params
        TRUE_INV_ALL = true_inv_prior_samples
        
        oracle = get_vmap_oracle(self.forward_ens, self.obs_design_fn)
        
        inverse_params_0 = inverse_ens.params
        state_0 = jax.vmap(inverse_ens.solver.init_state)(inverse_params_0)
        
        if self.inverse_ens_use_forward_params:
            ntk_nn_params = {
                'net': self.forward_ens.params['net'],
                'inv': true_inv_prior_samples,
            }
                
        def criterion(obs_design, rng=jax.random.PRNGKey(0)):
            
            ys_obs = oracle(self.forward_ens.params['net'], obs_design)
            obs_split = jnp.repeat(obs_design[None,:], self.ensemble_size, axis=0)
            
            # add some noise
            rng, k_ = jax.random.split(rng)
            ys_obs = ys_obs + self.noise_std * jax.random.normal(key=k_, shape=ys_obs.shape)
            
            if (not self.inverse_ens_use_forward_params) and (self.inverse_ensemble_pretraining_steps > 0):
            
                body_fun_1 = lambda val: body_fun_all(
                    val, 
                    jax.lax.stop_gradient(ys_obs), 
                    jax.lax.stop_gradient(obs_split),
                )
                
                if self.do_jit:
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
                        
            else:
                diff_nn_params = inverse_params_0
                    
            if self.inverse_ens_use_forward_params:
                ntk_params = ntk_nn_params
            else:
                ntk_params = diff_nn_params
                    
            COLLOC_JAC_ALL = jax.vmap(lambda p_: get_jac(colloc_residue_stacked, p_))(ntk_params)
            COLLOC_NTK_ALL = jax.vmap(lambda j_: jac_mult_fn(j_, j_))(COLLOC_JAC_ALL)
            COLLOC_RES_ALL = jax.vmap(colloc_residue_stacked)(diff_nn_params)
            rng_split = jax.random.split(rng, num=self.ensemble_size)
            
            scores_indiv, aux_indiv = jax.vmap(single_network_criterion, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0))(
                obs_design, diff_nn_params, ntk_params, COLLOC_NTK_ALL, COLLOC_JAC_ALL, COLLOC_RES_ALL, FOR_PARAMS_ALL, TRUE_INV_ALL, rng_split,
            )
            
            aux = {
                'indiv_scores': scores_indiv,
                'indiv_aux': aux_indiv,
                'indiv_param': diff_nn_params,
                'colloc_jacs': COLLOC_JAC_ALL,
                'colloc_ntks': COLLOC_NTK_ALL,
                'colloc_res': COLLOC_RES_ALL,
            }
            return - jnp.mean(scores_indiv), aux
        
        helper_fns = {
            'colloc_residues': colloc_residues,
            'colloc_residue_stacked': colloc_residue_stacked,
            'anc_residue': anc_residue,
            'single_network_criterion': single_network_criterion,
            'for_params': FOR_PARAMS_ALL,
            'inv_params': INV_PARAMS_ALL,
            'true_inv_all': true_inv_prior_samples,
        }
        
        return criterion, helper_fns


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

# # logger for this file
# logging.getLogger().setLevel(logging.INFO)


# class PINNEnsemblePerturbedInverseMethod(CriterionBasedAbstractMethod):
    
#     def __init__(self, simulator_xs, pde, pde_domain, exp_design_fn, obs_design_fn, 
#                  inv_embedding, inv_param_in_domain, exp_in_domain, obs_in_domain,
#                  inv_input_dim, exp_input_dim, obs_input_dim, obs_reading_count,
#                  x_input_dim, y_output_dim, 
#                  ensemble_size: int = 100, pinn_ensemble_args: Dict = dict(), 
#                  pinn_share_init: bool = False, pinn_init_meta_rounds: int = 0, pinn_init_meta_steps: int = 1000, pinn_meta_eps: float = 0.1,
#                  ensemble_steps: int = 100000, inverse_ensemble_training_steps: int = 1, added_noise: float = 0.1,
#                  pde_colloc_sample_num: int = 1000, icbc_colloc_sample_num: int = 100, use_implicit_diff: bool = False,
#                  obs_optim_with_gd: bool = True, do_jit: bool = False, acq_fn: str = 'ucb', chunk_size: int = 100,
#                  exp_setup_rounds: int = 10, obs_setup_rounds: int = 10, obs_search_time_limit: float = 3600., noise_std: float = 1e-3, 
#                  obs_optim_gd_params: Dict = dict(stepsize=1e-2, maxiter=1000, acceleration=True), obs_optim_grad_clip: float = None,
#                  reg: float = 1e-12, seed: int = 0):
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
#             noise_std=noise_std,
#             obs_optim_with_gd=obs_optim_with_gd,
#             obs_optim_gd_params=obs_optim_gd_params,
#             obs_optim_grad_clip=obs_optim_grad_clip,
#             do_jit=do_jit,
#             seed=seed,
#         )
        
#         self.reg = reg
#         self.use_implicit_diff = use_implicit_diff
#         self.inverse_ensemble_training_steps = inverse_ensemble_training_steps
#         self.pde_colloc_sample_num = pde_colloc_sample_num
#         self.icbc_colloc_sample_num = icbc_colloc_sample_num
#         self.chunk_size = chunk_size
#         self.added_noise = added_noise
            
#     def _generate_criterion_inner(self, exp_design, true_inv_prior_samples):
        
#         oracle_fn = jax.jit(self.forward_ens.generate_pred_function())
            
#         new_nn_params = {
#             'net': self.forward_ens.params['net'], 
#             'inv': true_inv_prior_samples,
#         }
                    
#         updated_ensemble_args = {k: self.pinn_ensemble_args[k] for k in self.pinn_ensemble_args.keys()}
#         updated_ensemble_args['n_pde_collocation_pts'] = self.pde_colloc_sample_num
#         updated_ensemble_args['n_icbc_collocation_pts'] = self.icbc_colloc_sample_num
#         updated_ensemble_args['maxiter'] = self.inverse_ensemble_training_steps
#         # updated_ensemble_args['optim_method'] = 'lbfgs'
#         # updated_ensemble_args['optim_args'] = None
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
        
#         inverse_ens.prep_simulator(
#             exp_params=exp_design, 
#             inv_params_guesses=new_nn_params['inv'], 
#             # new_nn_params=new_nn_params,
#             # prior_reg=0.,
#         )
                
#         inverse_params_0 = new_nn_params  #inverse_ens.params
#         # inverse_params_0 = jax.tree_map(lambda x: jnp.array(x, dtype=jnp.zeros(1).dtype), inverse_params_0)
#         # inverse_opt_states_0 = inverse_ens.opt_state_list
#         # step_batch_fn = inverse_ens.step_batch
#         loss_batch_fn = inverse_ens.loss_batch
#         score_fn = jax.vmap(lambda inv1, inv2: self.compare_inv(inv1, inv2))
        
#         def noisy_tree(tree_to_noise, eps, rng):
#             leaves, tree = jax.tree_util.tree_flatten(tree_to_noise)
#             subkeys = jax.random.split(rng, len(leaves))
#             subkeys = jax.tree_util.tree_unflatten(tree, subkeys)

#             def add_noise(val, key, eps=0.1):
#                 return val + eps * jax.random.normal(key, val.shape, val.dtype)

#             return jax.tree_util.tree_map(add_noise, tree_to_noise, subkeys)
        
#         def criterion(obs_design, rng=jax.random.PRNGKey(0)):
            
#             noisy_init = noisy_tree(
#                 tree_to_noise=inverse_params_0, 
#                 eps=self.added_noise, 
#                 rng=rng
#             )
            
#             oracle = get_vmap_oracle(self.forward_ens, self.obs_design_fn)
#             ys_obs = oracle(self.forward_ens.params['net'], obs_design)
#             obs_split = jnp.repeat(obs_design[None,:], self.ensemble_size, axis=0)
#             if self.chunk_size >= self.ensemble_size:
#                 nn_params = jax.vmap(inverse_ens.solver.run)(noisy_init, obs_design=obs_split, observation=ys_obs, prior_inv=noisy_init['inv']).params
#             else:
#                 nn_params = vmap_chunked(inverse_ens.solver.run, in_axes=(0, 0, 0, 0), chunk_size=self.chunk_size)(
#                     noisy_init, obs_design=obs_split, observation=ys_obs, prior_inv=noisy_init['inv']).params
                
#             # negative since requires maximisation
#             indiv_score = score_fn(nn_params["inv"], true_inv_prior_samples)
#             score = - jnp.nanmean(jnp.log(indiv_score + self.reg))
#             aux = {
#                 'true_inv_prior_samples': true_inv_prior_samples, 
#                 'nn_params_init': noisy_init,
#                 'nn_params': nn_params,
#                 # 'nn_opt_states': nn_opt_states,
#                 'indiv_score': indiv_score,
#             }
#             return score, aux
        
#         if self.do_jit:
#             criterion = jax.jit(criterion)
        
#         return criterion, dict(
#             forward_params=self.forward_ens.params,
#             nn_params_init=inverse_params_0
#         )
