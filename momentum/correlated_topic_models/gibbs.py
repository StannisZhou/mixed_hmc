import numpy as np

import numba
from momentum.potential.correlated_topic_models import potential, softmax
from pypolyagamma import PyPolyaGamma
from tqdm import tqdm


def draw_samples_gibbs(n_samples, w, mu, Sigma, beta, use_efficient_proposal=False):
    K, V = beta.shape
    N = w.shape[0]
    Sigma_inv = np.linalg.inv(Sigma)
    pg = PyPolyaGamma()
    z = np.random.randint(0, K, size=w.shape)
    eta = np.random.randn(K - 1)
    z_samples = []
    eta_samples = []
    lbda = np.zeros_like(eta)
    for ii in tqdm(range(n_samples)):
        # Update z
        z = update_z(
            z, eta, w, mu, Sigma, beta, use_efficient_proposal=use_efficient_proposal
        )
        # Update lbda
        eta_full = np.concatenate([np.zeros(1), eta])
        rho = eta_full - np.log(np.sum(np.exp(eta_full)) - np.exp(eta_full))
        pg.pgdrawv(N * np.ones_like(eta), rho[1:], lbda)
        # Update eta
        eta = update_eta(z, eta, lbda, mu, Sigma_inv)
        z_samples.append(z.copy())
        eta_samples.append(eta.copy())

    return np.stack(z_samples), np.stack(eta_samples)


@numba.jit(nopython=True, cache=True)
def update_z(z, eta, w, mu, Sigma, beta, use_efficient_proposal=False):
    for ind in np.random.permutation(z.shape[0]):
        potential_list = []
        for ii in range(beta.shape[0]):
            z_proposal = z.copy()
            z_proposal[ind] = ii
            potential_list.append(potential(z_proposal, eta, w, mu, Sigma, beta))

        potential_array = np.array(potential_list)
        distribution = softmax(-potential_array)
        proposal_dist = distribution.copy()
        if use_efficient_proposal:
            proposal_dist[z[ind]] = 0

        proposal_dist += 1e-12
        proposal_dist /= np.sum(proposal_dist)
        proposal_for_ind = np.argmax(np.random.multinomial(1, proposal_dist))
        if use_efficient_proposal:
            delta_E = np.log(1 - distribution[proposal_for_ind]) - np.log(
                1 - distribution[z[ind]]
            )
            if np.random.exponential() > delta_E:
                z[ind] = proposal_for_ind
        else:
            z[ind] = proposal_for_ind

    return z


@numba.jit(nopython=True, cache=True)
def update_eta(z, eta, lbda, mu, Sigma_inv):
    K = mu.shape[0] + 1
    N = z.shape[0]
    sigma_squared_inv = np.diag(Sigma_inv)
    tau_squared = 1 / (sigma_squared_inv + lbda)
    C = np.zeros(K)
    freq = np.bincount(z)
    C[: len(freq)] = freq
    kai = C[1:] - N / 2
    for ind in np.random.permutation(K - 1):
        all_but_one = np.arange(K - 1) != ind
        mu_k = (
            mu[ind]
            - np.dot(Sigma_inv[ind][all_but_one], eta[all_but_one] - mu[all_but_one])
            / Sigma_inv[ind, ind]
        )
        eta_full = np.concatenate((np.zeros(1), eta))
        gamma_k = tau_squared[ind] * (
            sigma_squared_inv[ind] * mu_k
            + kai[ind]
            + lbda[ind] * np.log(np.sum(np.exp(eta_full)) - np.exp(eta_full[ind + 1]))
        )
        eta[ind] = np.sqrt(tau_squared[ind]) * np.random.randn() + gamma_k

    return eta
