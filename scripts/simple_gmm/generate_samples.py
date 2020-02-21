import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np

import joblib
import sacred
from momentum.simple_gmm.dhmc import draw_samples_dhmc
from momentum.simple_gmm.exact import draw_samples_exact
from momentum.simple_gmm.mixed_hmc import draw_samples_mixed_hmc
from momentum.simple_gmm.nuts import draw_samples_nuts
from momentum.simple_gmm.pymc3 import draw_samples_pymc3
from sacred.observers import FileStorageObserver

log_folder = os.path.expanduser('~/logs/simple_gmm_1d_results')
ex = sacred.Experiment('simple_gmm_generate_samples')
ex.observers.append(FileStorageObserver.create(log_folder))


@ex.config
def config():
    pi = np.array([0.15, 0.3, 0.3, 0.25])
    mu_list = np.array([[-2, 0, 2, 4]]).T
    Sigma_list = 0.1 * np.stack([np.eye(1) for _ in range(pi.shape[0])])
    method = 'nuts'
    n_warm_up_samples = int(1e3)
    n_samples = int(2e3)
    epsilon = 0.6
    L = 20
    n_discrete_to_update = 1
    n_chains = 192


@ex.main
def run(
    pi,
    mu_list,
    Sigma_list,
    method,
    n_warm_up_samples,
    n_samples,
    epsilon,
    L,
    n_discrete_to_update,
    n_chains,
):
    # Generate temp folder
    temp_folder = tempfile.TemporaryDirectory()
    temp_folder_name = temp_folder.name
    if method == 'exact':
        z_samples, x_samples = draw_samples_exact(
            n_warm_up_samples + n_samples, pi, mu_list, Sigma_list
        )
        results = {'z': z_samples, 'x': x_samples}
    elif method == 'nuts':
        x_samples = joblib.Parallel(n_jobs=joblib.cpu_count())(
            joblib.delayed(draw_samples_nuts)(
                n_warm_up_samples, n_samples, pi, mu_list, Sigma_list
            )
            for _ in range(n_chains)
        )
        x_samples = np.stack(x_samples)
        results = {'x': x_samples}
    elif method == 'pymc3':
        z_samples, x_samples = draw_samples_pymc3(
            n_warm_up_samples, n_samples, pi, mu_list, Sigma_list, n_chains=n_chains
        )
        results = {
            'z': z_samples.reshape((n_chains, -1)),
            'x': x_samples.reshape((n_chains, -1, mu_list.shape[1])),
        }
    elif method == 'mixed_hmc':
        output = joblib.Parallel(n_jobs=joblib.cpu_count())(
            joblib.delayed(draw_samples_mixed_hmc)(
                n_warm_up_samples + n_samples,
                pi,
                mu_list,
                Sigma_list,
                epsilon,
                L,
                n_discrete_to_update,
            )
            for _ in range(n_chains)
        )
        z_samples, x_samples, accept_array = list(zip(*output))
        z_samples, x_samples, accept_array = (
            np.stack(z_samples),
            np.stack(x_samples),
            np.stack(accept_array),
        )
        print(np.mean(accept_array))
        results = {'z': z_samples, 'x': x_samples, 'accept_array': accept_array}
    elif method == 'dhmc':
        output = joblib.Parallel(n_jobs=joblib.cpu_count())(
            joblib.delayed(draw_samples_dhmc)(
                n_samples=n_warm_up_samples + n_samples,
                pi=pi,
                mu_list=mu_list,
                Sigma_list=Sigma_list,
                epsilon_range=epsilon,
                L_range=L,
            )
            for _ in range(n_chains)
        )
        z_samples, x_samples, accept_array = list(zip(*output))
        z_samples, x_samples, accept_array = (
            np.stack(z_samples),
            np.stack(x_samples),
            np.stack(accept_array),
        )
        print(np.mean(accept_array))
        results = {'z': z_samples, 'x': x_samples, 'accept_array': accept_array}
    else:
        raise ValueError('Unsupported method {}'.format(method))

    if n_chains == 1:
        x = np.linspace(-10, 10, int(1e4))
        n_components, n_dimensions = mu_list.shape
        mixture_density = np.zeros((n_dimensions, x.shape[0]))
        for ii in range(n_dimensions):
            mus = mu_list[:, ii]
            sigmas = np.array([Sigma_list[jj, ii, ii] for jj in range(n_components)])
            for jj in range(n_components):
                mixture_density[ii] += (
                    pi[jj]
                    * np.exp(-0.5 * (x - mus[jj]) ** 2 / sigmas[jj])
                    / (np.sqrt(2 * np.pi * sigmas[jj]))
                )

        assert n_dimensions == 1
        fig, ax = plt.subplots(1, n_dimensions, figsize=(10 * n_dimensions, 10))
        ax.hist(x_samples[0, :, 0], density=True, bins=500)
        ax.plot(x, mixture_density[0])
        fig.tight_layout()
        fig_fname = '{}/histogram.png'.format(temp_folder_name)
        fig.savefig(fig_fname, dpi=400)
        ex.add_artifact(fig_fname)
        plt.show()

    results_fname = '{}/results.joblib'.format(temp_folder_name)
    joblib.dump(results, results_fname)
    ex.add_artifact(results_fname)
    temp_folder.cleanup()


# Experiments for 1D GMM
## NUTS experiments
ex.run(config_updates={'method': 'nuts'})
## Mixed HMC experiments
ex.run(
    config_updates={
        'method': 'mixed_hmc',
        'epsilon': 0.6,
        'L': 20,
        'n_discrete_to_update': 1,
    }
)
## PyMC3 experiments
ex.run(config_updates={'method': 'pymc3'})
## DHMC experiments
for epsilon in [[0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8]]:
    for L in [[5, 10], [10, 15], [15, 20], [20, 25]]:
        ex.run(config_updates={'method': 'dhmc', 'epsilon': epsilon, 'L': L})
