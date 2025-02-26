import time
from functools import partial
import tqdm
import numpy as np
import jax
import jax.numpy as jnp

import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound, qSimpleRegret
from botorch.optim.initializers import gen_batch_initial_conditions, initialize_q_batch_nonneg
from botorch.generation import gen_candidates_torch, get_best_candidates
from botorch.sampling.stochastic_samplers import StochasticSampler

from pied.utils import sample_from_uniform


def solve_with_numerical_sim(
        numerical_solver, exp_design, obs_design, ys_obs,
        inv_param_in_domain, inv_input_dim, pinn_ensemble_training_steps,
        bo_rounds=100, rng=jax.random.PRNGKey(0),
    ):
                
    inv_samples = []
    
    for inv_case in tqdm.trange(len(ys_obs), desc='Inverse problems'):
                            
        YS_CHECK = ys_obs[inv_case]
        
        inv_transform = lambda inv: (inv - inv_param_in_domain[:,0]) / (inv_param_in_domain[:,1] - inv_param_in_domain[:,0])
        inv_rev_transform = lambda x: inv_param_in_domain[:,0] + x * (inv_param_in_domain[:,1] - inv_param_in_domain[:,0])
        hypercube = jnp.array([[0., 1.] for _ in range(inv_param_in_domain.shape[0])])
    
        ran_inv_params_untransformed = []
        ran_inv_params = []
        ran_inv_scores = []
                    
        for rr in range(bo_rounds):
            
            if rr < 2:
                
                rng, k_ = jax.random.split(rng)
                inv_design_candidate_untransformed = sample_from_uniform(
                    n=1, 
                    bounds=hypercube, 
                    sample_dim=hypercube.shape[0], 
                    rng=k_
                )[0]
            
            else:
            
                train_X = torch.tensor(np.array(ran_inv_params_untransformed))
                train_Y = torch.tensor(np.array(ran_inv_scores).reshape(-1, 1))
                train_Y = (train_Y - torch.mean(train_Y)) / torch.std(train_Y)
                
                model = SingleTaskGP(train_X, train_Y)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
                
                sampler = StochasticSampler(sample_shape=torch.Size([128]))
                q_fn = qUpperConfidenceBound(model, beta=1., sampler=sampler)
                
                inv_domain_np = np.array(hypercube)
                Xinit = gen_batch_initial_conditions(q_fn, torch.tensor(inv_domain_np.T), q=1, num_restarts=25, raw_samples=500)
                batch_candidates, batch_acq_values = gen_candidates_torch(
                    initial_conditions=Xinit,
                    acquisition_function=q_fn,
                    lower_bounds=torch.tensor(inv_domain_np[:,0]),
                    upper_bounds=torch.tensor(inv_domain_np[:,1]),
                )
                inv_design_candidate_untransformed = jnp.array(get_best_candidates(batch_candidates, batch_acq_values)[0].cpu().detach().numpy())
            
            inv_design_candidate = inv_rev_transform(inv_design_candidate_untransformed)
            ys_pred = numerical_solver(exp_design, inv_design_candidate)(obs_design)
            assert ys_pred.shape == YS_CHECK.shape
            score = - jnp.log(jnp.linalg.norm(ys_pred - YS_CHECK))
            
            ran_inv_params_untransformed.append(inv_design_candidate_untransformed)
            ran_inv_params.append(inv_design_candidate)
            ran_inv_scores.append(score)
                                    
        best_obs_i = np.argmax(ran_inv_scores)
        inv_score_round = ran_inv_scores[best_obs_i]
        inv_guess_round = ran_inv_params[best_obs_i]
        
        inv_samples.append(inv_guess_round)
        
    inv_samples = jnp.array(inv_samples)
    aux = dict()
    
    return inv_samples, aux
