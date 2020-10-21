import numpy as np

import jax
from momentum.hmc.hmc_within_gibbs import hmc_within_gibbs
from momentum.potential.variable_selection import generate_variable_selection_potential
from momentum.utils import jax_prng_key


def draw_samples_hmc_within_gibbs(n_samples, X, y, sigma, epsilon, L):
    N, p = X.shape
    potential = generate_variable_selection_potential(X, y, sigma)
    labels_for_discrete = jax.device_put(np.tile(np.arange(2), (p, 1)))
    beta0 = jax.device_put(np.random.rand(p))
    gamma0 = jax.device_put(np.random.binomial(1, 0.5, size=(p,)))
    key = jax_prng_key()
    gamma_samples, beta_samples, accept_array = hmc_within_gibbs(
        q0_discrete=gamma0,
        q0_continuous=beta0,
        n_samples=n_samples,
        epsilon=epsilon,
        L=L,
        key=key,
        labels_for_discrete=labels_for_discrete,
        potential=potential,
        mode='GB',
    )
    return gamma_samples, beta_samples, accept_array
