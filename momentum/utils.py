import numpy as np

import jax


def jax_prng_key():
    return jax.random.PRNGKey(np.random.randint(int(1e5)))


def categorical(key, p, shape=()):
    s = jax.numpy.cumsum(p)
    r = jax.random.uniform(key, shape=shape + (1,))
    return jax.numpy.sum(s < r, axis=-1)
