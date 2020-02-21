import numpy as onp

import jax
import jax.numpy as np
from momentum.hmc.mixed_hmc_jax import mixed_hmc_on_joint
from momentum.potential.simple_gmm import generate_simple_gmm_potential
from momentum.utils import jax_prng_key


def draw_samples_mixed_hmc(
    n_samples, pi, mu_list, Sigma_list, epsilon, L, n_discrete_to_update
):
    potential = generate_simple_gmm_potential(pi, mu_list, Sigma_list)
    z0 = np.array([onp.random.randint(0, pi.shape[0])])
    x0 = jax.device_put(onp.random.randn(mu_list.shape[1]))
    labels_for_discrete = np.arange(pi.shape[0]).reshape((1, -1))
    key = jax_prng_key()
    z_samples, x_samples, accept_array = mixed_hmc_on_joint(
        q0_discrete=z0,
        q0_continuous=x0,
        n_samples=n_samples,
        key=key,
        epsilon=epsilon,
        L=L,
        n_discrete_to_update=n_discrete_to_update,
        labels_for_discrete=labels_for_discrete,
        potential=potential,
        mode='GB',
        progbar=False,
    )
    return z_samples, x_samples, accept_array
