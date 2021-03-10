import numpy as np

import jax
from momentum.hmc.mixed_hmc_jax import mixed_hmc_on_joint
from momentum.potential.variable_selection import \
    generate_variable_selection_potential
from momentum.utils import jax_prng_key


def draw_samples_mixed_hmc(
    n_samples,
    X,
    y,
    sigma,
    epsilon,
    total_travel_time,
    L,
    n_discrete_to_update,
    adaptive_step_size=None,
    progbar=True,
    mode='RW'
):
    N, p = X.shape
    potential = generate_variable_selection_potential(X, y, sigma)
    labels_for_discrete = jax.device_put(np.tile(np.arange(2), (p, 1)))
    beta0 = jax.device_put(np.random.rand(p))
    gamma0 = jax.device_put(np.random.binomial(1, 0.5, size=(p,)))
    key = jax_prng_key()
    gamma_samples, beta_samples, accept_array = mixed_hmc_on_joint(
        q0_discrete=gamma0,
        q0_continuous=beta0,
        n_samples=n_samples,
        epsilon=epsilon,
        total_travel_time=total_travel_time,
        L=L,
        key=key,
        n_discrete_to_update=n_discrete_to_update,
        labels_for_discrete=labels_for_discrete,
        potential=potential,
        mode=mode,
        adaptive_step_size=adaptive_step_size,
        progbar=progbar,
    )
    return gamma_samples, beta_samples, accept_array
