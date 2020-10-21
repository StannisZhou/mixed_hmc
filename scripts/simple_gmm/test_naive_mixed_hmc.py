import matplotlib.pyplot as plt
import numpy as np

from momentum.hmc.naive_mixed_hmc import naive_mixed_hmc
from momentum.potential.simple_gmm import get_mixture_density

pi = np.array([0.15, 0.3, 0.3, 0.25])
mu_list = np.array([-2, 0, 2, 4])
sigma_list = np.sqrt(0.1) * np.ones(pi.shape[0])


x0 = np.random.randint(4)
q0 = np.random.randn()
n_warm_up_samples = int(1e4)
n_samples = int(1e6)
epsilon = 0.4
L = 15
use_k = True


x_samples, q_samples, accept_list = naive_mixed_hmc(
    x0,
    q0,
    n_warm_up_samples + n_samples,
    epsilon,
    L,
    pi,
    mu_list,
    sigma_list,
    use_k=use_k,
)

print(np.mean(accept_list))

x = np.linspace(-10, 10, int(1e4))
mixture_density = get_mixture_density(x, pi, mu_list, sigma_list)
fig, ax = plt.subplots(1, 1)
ax.hist(q_samples[n_warm_up_samples:], density=True, bins=500)
ax.plot(x, mixture_density)
plt.show()
