import numpy as np

import jax
from momentum.hmc.hmc_within_gibbs import hmc_within_gibbs
from momentum.potential.correlated_topic_models import (
    generate_correlated_topic_models_potential,
)
from momentum.utils import jax_prng_key


def draw_samples_hmc_within_gibbs(n_samples, w, mu, Sigma, beta, epsilon, L, mode='GB'):
    potential = generate_correlated_topic_models_potential(
        w, mu, Sigma, beta, use_jax=True, use_dhmc=False
    )
    K, V = beta.shape
    z0 = jax.device_put(np.random.randint(0, K, size=w.shape))
    eta0 = jax.device_put(np.random.randn(K - 1))
    labels_for_discrete = jax.device_put(np.tile(np.arange(K), (w.shape[0], 1)))
    key = jax_prng_key()
    z_samples, eta_samples, accept_array = hmc_within_gibbs(
        q0_discrete=z0,
        q0_continuous=eta0,
        n_samples=n_samples,
        epsilon=epsilon,
        L=L,
        key=key,
        labels_for_discrete=labels_for_discrete,
        potential=potential,
        mode=mode,
    )
    return z_samples, eta_samples, accept_array
