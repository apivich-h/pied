from functools import partial
import tqdm
import jax
import jax.numpy as jnp

from pied.utils import sample_from_uniform
from pied.models.pinn_ensemble import PINNEnsemble


def solve_with_inverse_pinn(
        pde, pde_domain, exp_design_fn, obs_design_fn, exp_design, obs_design, ys_obs,
        inv_param_in_domain, inv_input_dim, pinn_ensemble_training_steps, shared_nn_params=None,
        rng=jax.random.PRNGKey(0), **pinn_ensemble_args
    ):
    
    n_inv = ys_obs.shape[0]
    
    rng, k_ = jax.random.split(rng)
    ensinv = PINNEnsemble(
        pde=pde,
        pde_domain=pde_domain,
        exp_design_fn=exp_design_fn,
        obs_design_fn=obs_design_fn,
        inv_problem=True,
        rng=k_,
        **pinn_ensemble_args,
    )
    obs_design_rep = jnp.repeat(obs_design[None,:], repeats=n_inv, axis=0).reshape(n_inv, -1)
    
    rng, k_ = jax.random.split(rng)
    inv_params_guesses = sample_from_uniform(
        n=n_inv,
        bounds=inv_param_in_domain,
        sample_dim=inv_input_dim,
        rng=k_,
    )
    
    ensinv.prep_simulator(exp_params=exp_design, inv_params_guesses=inv_params_guesses, new_nn_params=shared_nn_params)

    for _ in tqdm.trange(pinn_ensemble_training_steps, mininterval=10):
        ensinv.step_opt(obs_design=obs_design_rep, observation=ys_obs)
        
    inv_samples = ensinv.params['inv']
    aux = dict()
    
    return inv_samples, aux
