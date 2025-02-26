import jax.numpy as jnp


def generate_loss(loss='nllh', **kwargs):
    if loss == 'nllh':
        return generate_nllh_loss(**kwargs)
    else:
        raise ValueError


# def generate_mse_loss(scaling=1.):
#     return lambda y_obs, y_true: jnp.log(scaling * jnp.sum((y_obs - y_true)**2))


def generate_nllh_loss(noise_std=0.1):
    return lambda y_obs, y_true: jnp.sum((y_obs - y_true)**2 / (2. * noise_std**2))


# def generate_loss(noisy_simulator, obs_design_fn, loss='mse', **kwargs):
#     if loss == 'mse':
#         return generate_mse_loss(noisy_simulator, obs_design_fn, **kwargs)
#     else:
#         raise ValueError


# def generate_mse_loss(noisy_simulator, obs_design_fn, scaling=1.):

#     def log_cost(y_obs, exp_params, obs_params, inv, rng):
#         y = noisy_simulator(exp_params, inv, rng)(obs_design_fn(obs_params))
#         return - scaling * jnp.sum((y_obs - y)**2)
    
#     return log_cost
