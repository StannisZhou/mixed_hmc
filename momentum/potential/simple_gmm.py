import numpy as np

import jax
from jax.scipy.special import logsumexp


def generate_simple_gmm_potential(
    pi, mu_list, Sigma_list, use_jax=True, use_dhmc=False
):
    if use_jax:
        import jax.numpy as np

        pi = jax.device_put(pi)
        mu_list = jax.device_put(mu_list)
        Sigma_list = jax.device_put(Sigma_list)
    else:
        import numpy as np

    def map_embedded_to_discrete(z_embedded):
        return np.floor(z_embedded).astype(np.int32)

    def simple_gmm_potential(z, x):
        n_components = pi.shape[0]
        if use_dhmc:
            z = map_embedded_to_discrete(z)
            boundary_potential = (np.sum(z < 0) + np.sum(z >= n_components)) * np.exp(
                80
            )
        else:
            boundary_potential = 0

        z = z[0]
        mean_deviation = x - mu_list[z]
        Sigma_inv = np.linalg.inv(Sigma_list[z])
        potential = (
            -np.log(pi[z])
            + 0.5
            * np.log(
                np.prod(2 * np.pi * np.ones_like(x)) * np.linalg.det(Sigma_list[z])
            )
            + 0.5 * np.dot(np.dot(mean_deviation, Sigma_inv), mean_deviation)
            + boundary_potential
        )
        return potential

    return simple_gmm_potential


def generate_simple_gmm_marginalized_potential(pi, mu_list, Sigma_list):
    import jax.numpy as np

    def simple_gmm_marginalized_potential(param):
        x = param['x']

        def get_potential_for_component(ii):
            mean_deviation = x - mu_list[ii]
            Sigma_inv = np.linalg.inv(Sigma_list[ii])
            potential = (
                -np.log(pi[ii])
                + 0.5
                * np.log(
                    np.prod(2 * np.pi * np.ones_like(x)) * np.linalg.det(Sigma_list[ii])
                )
                + 0.5 * np.dot(np.dot(mean_deviation, Sigma_inv), mean_deviation)
            )
            return potential

        potential = -logsumexp(
            -np.array([get_potential_for_component(ii) for ii in range(pi.shape[0])])
        )
        return potential

    return simple_gmm_marginalized_potential


def get_mixture_density(x, pi, mu_list, sigma_list):
    mixture_density = np.zeros_like(x)
    for ii in range(pi.shape[0]):
        mixture_density += (
            pi[ii]
            * np.exp(-0.5 * (x - mu_list[ii]) ** 2 / sigma_list[ii] ** 2)
            / np.sqrt(2 * np.pi * sigma_list[ii] ** 2)
        )

    return mixture_density
