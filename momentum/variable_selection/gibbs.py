import numpy as np

import numba
from pypolyagamma import PyPolyaGamma
from tqdm import tqdm


@numba.jit(nopython=True, cache=True)
def logistic(x):
    return 1 / (1 + np.exp(-x))


@numba.jit(nopython=True, cache=True)
def update_gamma(curr_gamma, curr_beta, N, p, X, y):
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

        prob = np.prod(prob_array[1] / prob_array[0])
        prob = prob / (1 + prob)
        curr_gamma[ind] = int(np.random.rand() <= prob)

    return curr_gamma


def draw_samples_gibbs(n_samples, X, y, sigma):
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
        curr_gamma = update_gamma(curr_gamma, curr_beta, N, p, X, y)
        beta_samples.append(curr_beta.copy())
        gamma_samples.append(curr_gamma.copy())

    beta_samples = np.stack(beta_samples)
    gamma_samples = np.stack(gamma_samples)
    return gamma_samples, beta_samples
