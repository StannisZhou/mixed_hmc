import numpy as np

import jax
from momentum.hmc.dhmc import dhmc_on_joint
from momentum.potential.variable_selection import generate_variable_selection_potential
from momentum.utils import jax_prng_key


def draw_samples_dhmc(n_samples, X, y, sigma, epsilon, L, progbar=True):
    N, p = X.shape
    potential = generate_variable_selection_potential(X, y, sigma, use_dhmc=True)
    beta0 = jax.device_put(np.random.rand(p))
    gamma0 = jax.device_put(2 * np.random.rand(p))
    key = jax_prng_key()
    gamma_samples, beta_samples, accept_array = dhmc_on_joint(
        q0_embedded=gamma0,
        q0_continuous=beta0,
        n_samples=n_samples,
        epsilon_range=epsilon,
        L_range=L,
        key=key,
        potential=potential,
        progbar=progbar,
    )
    gamma_samples = np.floor(gamma_samples).astype(np.int32)
    return gamma_samples, beta_samples, accept_array
