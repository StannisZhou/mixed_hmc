import numba
import numpy as np
from tqdm import tqdm


def naive_mixed_hmc(x0, q0, n_samples, epsilon, L, pi, mu_list, sigma_list, use_k=True):
    """Function for comparing mixed HMC and naive Metropolis updates within HMC

    Parameters
    ----------
    x0 : int
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
    x_samples : np.array
        Array of shape (n_samples,). Samples for x
    q_samples : np.array
        Array of shape (n_samples,). Samples for x
    accept_list : np.array
        Array of shape (n_samples,). Records whether we accept or reject at each step
    """

    @numba.jit(nopython=True)
    def potential(x, q):
        potential = (
            -np.log(pi[x])
            + 0.5 * np.log(2 * np.pi * sigma_list[x] ** 2)
            + 0.5 * (q - mu_list[x]) ** 2 / sigma_list[x] ** 2
        )
        return potential

    @numba.jit(nopython=True)
    def grad_potential(x, q):
        grad_potential = (q - mu_list[x]) / sigma_list[x] ** 2
        return grad_potential

    @numba.jit(nopython=True)
    def take_naive_mixed_hmc_step(x0, q0, epsilon, L, n_components):
        # Resample momentum
        p0 = np.random.randn()
        k0 = np.random.exponential()
        # Initialize q, k, delta_U
        x = x0
        q = q0
        p = p0
        k = k0
        delta_U = 0.0
        # Take L steps
        for ii in range(L):
            q, p = leapfrog_step(x=x, q=q, p=p, epsilon=epsilon)
            x, k, delta_U = update_discrete(
                x0=x, k0=k, q=q, delta_U=delta_U, n_components=n_components
            )

        # Accept or reject
        current_E = potential(x0, q0) + 0.5 * p0 ** 2
        proposed_E = potential(x, q) + 0.5 * p ** 2
        accept = np.random.rand() < np.exp(current_E + delta_U - proposed_E)
        if not accept:
            x, q = x0, q0

        return x, q, accept

    @numba.jit(nopython=True)
    def leapfrog_step(x, q, p, epsilon):
        p -= 0.5 * epsilon * grad_potential(x, q)
        q += epsilon * p
        p -= 0.5 * epsilon * grad_potential(x, q)
        return q, p

    @numba.jit(nopython=True)
    def update_discrete(x0, k0, q, delta_U, n_components):
        x = x0
        k = k0
        distribution = np.ones(n_components)
        distribution[x] = 0
        distribution /= np.sum(distribution)
        proposal_for_ind = np.argmax(np.random.multinomial(1, distribution))
        x = proposal_for_ind
        delta_E = potential(x, q) - potential(x0, q)
        # Decide whether to accept or reject
        if use_k:
            accept = k > delta_E
            if accept:
                delta_U += potential(x, q) - potential(x0, q)
                k -= delta_E
            else:
                x = x0
        else:
            accept = np.random.exponential() > delta_E
            assert k == k0
            if not accept:
                x = x0

        return x, k, delta_U

    x, q = x0, q0
    x_samples, q_samples, accept_list = [], [], []
    for _ in tqdm(range(n_samples)):
        x, q, accept = take_naive_mixed_hmc_step(
            x0=x, q0=q, epsilon=epsilon, L=L, n_components=pi.shape[0]
        )
        x_samples.append(x)
        q_samples.append(q)
        accept_list.append(accept)

    x_samples = np.array(x_samples)
    q_samples = np.array(q_samples)
    accept_list = np.array(accept_list)
    return x_samples, q_samples, accept_list
