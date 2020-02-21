import numpy as np

from tqdm import tqdm


def draw_samples_exact(n_samples, pi, mu_list, Sigma_list):
    x_samples = np.zeros((n_samples, mu_list.shape[1]))
    z_samples = np.zeros((n_samples,), dtype=np.int32)
    for ii in tqdm(range(n_samples)):
        z = np.random.choice(np.arange(pi.shape[0]), p=pi)
        z_samples[ii] = z
        x_samples[ii] = np.random.multivariate_normal(mu_list[z], Sigma_list[z])

    return z_samples, x_samples
