import numpy as np

import jax
from momentum.hmc.dhmc import dhmc_on_joint
from momentum.potential.correlated_topic_models import (
    generate_correlated_topic_models_potential,
)
from momentum.utils import jax_prng_key


def draw_samples_dhmc(
    n_samples, w, mu, Sigma, beta, epsilon, L, progbar=True, adaptive_step_size=None
):
    potential = generate_correlated_topic_models_potential(
        w, mu, Sigma, beta, use_jax=True, use_dhmc=True
    )
    K, V = beta.shape
    z0 = jax.device_put(np.random.randint(0, K, size=w.shape))
    eta0 = jax.device_put(np.random.randn(K - 1))
    key = jax_prng_key()
    z_samples, eta_samples, accept_array = dhmc_on_joint(
        q0_embedded=z0,
        q0_continuous=eta0,
        n_samples=n_samples,
        epsilon_range=epsilon,
        L_range=L,
        key=key,
        potential=potential,
        progbar=progbar,
        adaptive_step_size=adaptive_step_size,
    )
    return z_samples, eta_samples, accept_array
