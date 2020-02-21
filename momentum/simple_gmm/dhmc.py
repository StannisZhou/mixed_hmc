import numpy as onp

import jax
import jax.numpy as np
from momentum.hmc.dhmc import dhmc_on_joint
from momentum.potential.simple_gmm import generate_simple_gmm_potential
from momentum.utils import jax_prng_key


def draw_samples_dhmc(n_samples, pi, mu_list, Sigma_list, epsilon_range, L_range):
    potential = generate_simple_gmm_potential(pi, mu_list, Sigma_list, use_dhmc=True)
    z0 = jax.device_put(onp.array([pi.shape[0] * onp.random.rand()]))
    x0 = jax.device_put(onp.random.randn(mu_list.shape[1]))
    key = jax_prng_key()
    z_samples, x_samples, accept_array = dhmc_on_joint(
        q0_embedded=z0,
        q0_continuous=x0,
        n_samples=n_samples,
        key=key,
        epsilon_range=epsilon_range,
        L_range=L_range,
        potential=potential,
        progbar=False,
    )
    z_samples = np.floor(z_samples).astype(np.int32)
    return z_samples, x_samples, accept_array
