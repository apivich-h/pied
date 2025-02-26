import jax.numpy as jnp


def rbf_kernel(x1, x2, gamma):
    sq_distances = jnp.sum((x1[:, None, :] - x2[None, :, :])**2, axis=-1)
    return jnp.exp(-gamma * sq_distances)
