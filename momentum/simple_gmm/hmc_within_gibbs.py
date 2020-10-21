import numpy as onp

import jax
import jax.numpy as np
from momentum.hmc.hmc_within_gibbs import hmc_within_gibbs
from momentum.potential.simple_gmm import generate_simple_gmm_potential
from momentum.utils import jax_prng_key


def draw_samples_hmc_within_gibbs(n_samples, pi, mu_list, Sigma_list, epsilon, L):
    potential = generate_simple_gmm_potential(pi, mu_list, Sigma_list)
    z0 = np.array([onp.random.randint(0, pi.shape[0])])
    x0 = jax.device_put(onp.random.randn(mu_list.shape[1]))
    labels_for_discrete = np.arange(pi.shape[0]).reshape((1, -1))
    key = jax_prng_key()
    z_samples, x_samples, accept_array = hmc_within_gibbs(
        q0_discrete=z0,
        q0_continuous=x0,
        n_samples=n_samples,
        key=key,
        epsilon=epsilon,
        L=L,
        labels_for_discrete=labels_for_discrete,
        potential=potential,
        mode='GB',
    )
    return z_samples, x_samples, accept_array
