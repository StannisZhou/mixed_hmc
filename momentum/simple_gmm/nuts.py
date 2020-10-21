import numpy as np

import jax
from momentum.potential.simple_gmm import generate_simple_gmm_marginalized_potential
from numpyro.infer.mcmc import hmc
from numpyro.util import fori_collect


def draw_samples_nuts(n_warm_up_samples, n_samples, pi, mu_list, Sigma_list):
    pi = jax.device_put(pi)
    mu_list = jax.device_put(mu_list)
    Sigma_list = jax.device_put(Sigma_list)
    potential = generate_simple_gmm_marginalized_potential(pi, mu_list, Sigma_list)
    init_params = {'x': jax.device_put(10 * np.random.randn(mu_list.shape[1]))}
    init_kernel, sample_kernel = hmc(potential, algo='NUTS')
    hmc_state = init_kernel(
        init_params, num_warmup=n_warm_up_samples, target_accept_prob=0.6
    )
    samples = fori_collect(0, n_samples, sample_kernel, hmc_state, progbar=False)
    return samples.z['x']
