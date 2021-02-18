"""
Due to numerical issues with numba, scripts/simple_gmm/test_naive_mixed_hmc.py only tests
a simple modified random-walk (mode='RW') proposal.

This script additionally illustrates mixed HMC on the simple 1D GMM with Gibbs (mode='gibbs')
and modified Gibbs (mode='GB') proposals.
"""
import matplotlib.pyplot as plt
import numpy as np

from momentum.simple_gmm.mixed_hmc import draw_samples_mixed_hmc

pi = np.array([0.15, 0.3, 0.3, 0.25])
mu_list = np.array([[-2, 0, 2, 4]]).T
Sigma_list = 0.1 * np.stack([np.eye(1) for _ in range(pi.shape[0])])
n_warm_up_samples = int(1e4)
n_samples = int(2e6)
epsilon = 0.6
L = 20
n_discrete_to_update = 1
mode = 'gibbs'

z_samples, x_samples, accept_array = draw_samples_mixed_hmc(
    n_warm_up_samples + n_samples,
    pi,
    mu_list,
    Sigma_list,
    epsilon,
    L,
    n_discrete_to_update,
    mode=mode,
)
print(np.mean(accept_array))
x = np.linspace(-10, 10, int(1e4))
n_components, _ = mu_list.shape
mixture_density = np.zeros((1, x.shape[0]))
mus = mu_list[:, 0]
sigmas = np.array([Sigma_list[jj, 0, 0] for jj in range(n_components)])
for jj in range(n_components):
    mixture_density[0] += (
        pi[jj]
        * np.exp(-0.5 * (x - mus[jj]) ** 2 / sigmas[jj])
        / (np.sqrt(2 * np.pi * sigmas[jj]))
    )

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.hist(x_samples[n_warm_up_samples:, 0], density=True, bins=500)
ax.plot(x, mixture_density[0])
fig.tight_layout()
plt.show()
