import numba
import numpy as np

import jax


def generate_correlated_topic_models_potential(
    w, mu, Sigma, beta, use_jax=True, use_dhmc=False
):
    """generate_correlated_topic_models_potential

    Parameters
    ----------
    w : np.array
        An np array of shape (N,), where N is the number of words in
        the document. Each element takes values from 0 to V-1
    mu : np.array
        Shape (K-1,)
    Sigma : np.array
        Shape (K-1, K-1). Positive definite. Covariance matrix
    beta : np.array
        Shape (K, V). K number of topics, V size of vocabulary.
    use_jax :

    use_dhmc :

    Returns
    -------
    """
    if use_jax:
        import jax.numpy as np

        w = [jax.device_put(w[ii]) for ii in range(len(w))]
        mu = jax.device_put(mu)
        Sigma = jax.device_put(Sigma)
        beta = jax.device_put(beta)
    else:
        import numpy as np

    def map_embedded_to_discrete(z_embedded):
        return np.floor(z_embedded).astype(np.int32)

    def softmax(x):
        expx = np.exp(x)
        return expx / np.sum(expx)

    def correlated_topic_models_potential(z, eta):
        K, V = beta.shape
        if use_dhmc:
            z = map_embedded_to_discrete(z)
            boundary_potential = (np.sum(z < 0) + np.sum(z >= K)) * np.exp(80)
        else:
            boundary_potential = 0

        mean_deviation = eta - mu
        Sigma_inv = np.linalg.inv(Sigma)
        gaussian_potential = 0.5 * np.log(
            np.prod(2 * np.pi * np.ones_like(eta)) * np.linalg.det(Sigma)
        ) + 0.5 * np.dot(np.dot(mean_deviation, Sigma_inv), mean_deviation)
        theta = softmax(np.concatenate([np.zeros(1), eta]))
        categorical_potential = -np.sum(np.log(theta[z])) - np.sum(np.log(beta[z, w]))
        potential = gaussian_potential + categorical_potential + boundary_potential
        return potential

    return correlated_topic_models_potential


@numba.jit(nopython=True, cache=True)
def softmax(x):
    x -= np.max(x)
    expx = np.exp(x)
    return expx / np.sum(expx)


@numba.jit(nopython=True, cache=True)
def potential(z, eta, w, mu, Sigma, beta):
    K, V = beta.shape
    mean_deviation = eta - mu
    gaussian_potential = 0.5 * np.log(
        np.prod(2 * np.pi * np.ones_like(eta)) * np.linalg.det(Sigma)
    ) + 0.5 * np.dot(np.linalg.solve(Sigma, mean_deviation), mean_deviation)
    theta = softmax(np.concatenate((np.zeros(1), eta)))
    categorical_potential = -np.sum(np.log(theta[z]))
    for ii in range(len(z)):
        categorical_potential -= np.log(beta[z[ii], w[ii]])

    potential = gaussian_potential + categorical_potential
    return potential
