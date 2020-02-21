import numba
import numpy as np
from tqdm import tqdm


def naive_mixed_hmc(z0, q0, n_samples, epsilon, L, pi, mu_list, sigma_list, use_k=True):
    """Function for comparing mixed HMC and naive Metropolis updates within HMC

    Parameters
    ----------
    z0 : int
        Discrete variable for the mixture component
    q0 : float
        Continuous variable for the state of GMM
    n_samples : int
        Number of samples to draw
    epsilon : float
        Step size
    L : int
        Number of steps
    pi : np.array
        Array of shape (n_components,). The probabilities for different components
    mu_list : np.array
        Array of shape (n_components,). Means of different components
    sigma_list : np.array
        Array of shape (n_components,). Standard deviations of different components
    use_k : bool
        True if we use mixed HMC. False if we make naive Metropolis updates within HMC

    Returns
    -------
    z_samples : np.array
        Array of shape (n_samples,). Samples for z
    x_samples : np.array
        Array of shape (n_samples,). Samples for x
    accept_list : np.array
        Array of shape (n_samples,). Records whether we accept or reject at each step
    """

    @numba.jit(nopython=True)
    def potential(z, q):
        potential = (
            -np.log(pi[z])
            + 0.5 * np.log(2 * np.pi * sigma_list[z] ** 2)
            + 0.5 * (q - mu_list[z]) ** 2 / sigma_list[z] ** 2
        )
        return potential

    @numba.jit(nopython=True)
    def grad_potential(z, q):
        grad_potential = (q - mu_list[z]) / sigma_list[z] ** 2
        return grad_potential

    @numba.jit(nopython=True)
    def take_naive_mixed_hmc_step(z0, q0, epsilon, L, n_components):
        # Resample momentum
        p0 = np.random.randn()
        k0 = np.random.exponential()
        # Initialize q, k
        z = z0
        q = q0
        p = p0
        k = k0
        # Take L steps
        for ii in range(L):
            q, p = leapfrog_step(z=z, q=q, p=p, epsilon=epsilon)
            z, k = update_discrete(z0=z, k0=k, q=q, n_components=n_components)

        # Accept or reject
        current_U = potential(z0, q0)
        current_K = k0 + 0.5 * p0 ** 2
        proposed_U = potential(z, q)
        proposed_K = k + 0.5 * p ** 2
        accept = np.random.rand() < np.exp(
            current_U - proposed_U + current_K - proposed_K
        )
        if not accept:
            z, q = z0, q0

        return z, q, accept

    @numba.jit(nopython=True)
    def leapfrog_step(z, q, p, epsilon):
        p -= 0.5 * epsilon * grad_potential(z, q)
        q += epsilon * p
        p -= 0.5 * epsilon * grad_potential(z, q)
        return q, p

    @numba.jit(nopython=True)
    def update_discrete(z0, k0, q, n_components):
        z = z0
        k = k0
        distribution = np.ones(n_components)
        distribution[z] = 0
        distribution /= np.sum(distribution)
        proposal_for_ind = np.argmax(np.random.multinomial(1, distribution))
        z = proposal_for_ind
        delta_E = potential(z, q) - potential(z0, q)
        # Decide whether to accept or reject
        if use_k:
            accept = k > delta_E
            if accept:
                k -= delta_E
            else:
                z = z0
        else:
            accept = np.random.exponential() > delta_E
            assert k == k0
            if not accept:
                z = z0

        return z, k

    z, q = z0, q0
    z_samples, x_samples, accept_list = [], [], []
    for _ in tqdm(range(n_samples)):
        z, q, accept = take_naive_mixed_hmc_step(
            z0=z, q0=q, epsilon=epsilon, L=L, n_components=pi.shape[0]
        )
        z_samples.append(z)
        x_samples.append(q)
        accept_list.append(accept)

    z_samples = np.array(z_samples)
    x_samples = np.array(x_samples)
    accept_list = np.array(accept_list)
    return z_samples, x_samples, accept_list
