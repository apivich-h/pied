import jax
import jax.numpy as jnp

from ...models.pinn_ensemble import PINNEnsemble


def get_vmap_oracle(ensenble: PINNEnsemble, obs_design_fn):
    
    apply_fn = ensenble.net.apply
    
    def single_oracle(params, obs_param):
        f = lambda xs: apply_fn(params, xs)
        return obs_design_fn(f, obs_param)
    
    def vmap_oracle(params, obs_param):
        return jax.vmap(single_oracle, in_axes=(0, None))(params, obs_param)
    
    return vmap_oracle
