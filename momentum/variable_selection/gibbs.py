import numpy as np

import numba
from pypolyagamma import PyPolyaGamma
from tqdm import tqdm


@numba.jit(nopython=True, cache=True)
def logistic(x):
    return 1 / (1 + np.exp(-x))


@numba.jit(nopython=True, cache=True)
def update_gamma(curr_gamma, curr_beta, N, p, X, y, use_efficient_proposal=False):
    for ind in np.random.permutation(p):
        prob_array = np.zeros((2, N))
        temp_gamma = curr_gamma.copy()
        for value in range(2):
            temp_gamma[ind] = value
            probs = logistic(
                np.dot(np.dot(X, np.diag(temp_gamma).astype(np.float64)), curr_beta)
            )
            prob_array[value][y == 0] = 1 - probs[y == 0]
            prob_array[value][y == 1] = probs[y == 1]

        proposal_dist = np.array([np.prod(prob_array[0]), np.prod(prob_array[1])])
        proposal_dist /= np.sum(proposal_dist)
        if use_efficient_proposal:
            proposal_for_ind = 1 - curr_gamma[ind]
            delta_E = np.log(1 - proposal_dist[proposal_for_ind]) - np.log(
                1 - proposal_dist[curr_gamma[ind]]
            )
            if np.random.exponential() > delta_E:
                curr_gamma[ind] = proposal_for_ind
        else:
            curr_gamma[ind] = np.argmax(np.random.multinomial(1, proposal_dist))

    return curr_gamma


def draw_samples_gibbs(n_samples, X, y, sigma, use_efficient_proposal=False):
    N, p = X.shape
    pg = PyPolyaGamma()
    beta_samples = []
    gamma_samples = []
    curr_w = np.zeros(N)
    curr_beta = np.random.rand(p)
    curr_gamma = np.random.binomial(1, 0.5, size=(p,))
    for ii in tqdm(range(n_samples)):
        curr_X = np.dot(X, np.diag(curr_gamma))
        pg.pgdrawv(np.ones(N), np.dot(curr_X, curr_beta), curr_w)
        Sigma_w = np.linalg.inv(
            np.linalg.multi_dot([curr_X.T, np.diag(curr_w), curr_X])
            + np.eye(p) / sigma ** 2
        )
        m_w = np.dot(Sigma_w, np.dot(curr_X.T, y - 0.5))
        curr_beta = np.random.multivariate_normal(m_w, Sigma_w)
        curr_gamma = update_gamma(
            curr_gamma,
            curr_beta,
            N,
            p,
            X,
            y,
            use_efficient_proposal=use_efficient_proposal,
        )
        beta_samples.append(curr_beta.copy())
        gamma_samples.append(curr_gamma.copy())

    beta_samples = np.stack(beta_samples)
    gamma_samples = np.stack(gamma_samples)
    return gamma_samples, beta_samples
