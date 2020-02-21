import jax


def generate_variable_selection_potential(X, y, sigma, use_jax=True, use_dhmc=False):
    if use_jax:
        import jax.numpy as np

        X = jax.device_put(X)
        y = jax.device_put(y)
        sigma = jax.device_put(sigma)
    else:
        import numpy as np

    def map_embedded_to_discrete(z_embedded):
        return np.floor(z_embedded).astype(np.int32)

    def variable_selection_potential(gamma, beta):
        if use_dhmc:
            gamma = map_embedded_to_discrete(gamma)
            boundary_potential = (np.sum(gamma < 0) + np.sum(gamma >= 2)) * np.exp(80)
        else:
            boundary_potential = 0

        beta_prior_potential = np.sum(
            0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * beta ** 2 / sigma ** 2
        )
        probs = 1 / (
            1 + np.exp(-np.dot(np.dot(X, np.diag(gamma).astype(np.float32)), beta))
        )
        likelihood_potential = -np.sum(
            y * np.log(probs + 1e-12) + (1 - y) * np.log(1 - probs + 1e-12)
        )
        potential = beta_prior_potential + likelihood_potential + boundary_potential
        return potential

    return variable_selection_potential
