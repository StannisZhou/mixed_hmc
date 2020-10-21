import os
import tempfile

import numpy as np

import joblib
import sacred
from momentum.diagnostics.ess import get_min_ess
from momentum.variable_selection.dhmc import draw_samples_dhmc
from momentum.variable_selection.gibbs import draw_samples_gibbs
from momentum.variable_selection.hmc_within_gibbs import draw_samples_hmc_within_gibbs
from momentum.variable_selection.mixed_hmc import draw_samples_mixed_hmc
from momentum.variable_selection.pymc3 import draw_samples_pymc3
from sacred.observers import FileStorageObserver

log_folder = os.path.expanduser('~/logs/variable_selection_results')
ex = sacred.Experiment('variable_selection_generate_samples')
ex.observers.append(FileStorageObserver.create(log_folder))


@ex.config
def config():
    sigma = 5
    method = 'gibbs'
    n_warm_up_samples = int(1e3)
    n_samples = int(2e3)
    epsilon = 0.06
    L = 600
    total_travel_time = 40
    n_discrete_to_update = 1
    n_chains = 192
    data_fname = 'simulated_data.joblib'
    use_efficient_proposal = True


@ex.main
def run(
    sigma,
    method,
    n_warm_up_samples,
    n_samples,
    epsilon,
    total_travel_time,
    L,
    n_discrete_to_update,
    n_chains,
    data_fname,
    use_efficient_proposal,
):
    # Generate temp folder
    temp_folder = tempfile.TemporaryDirectory()
    temp_folder_name = temp_folder.name
    # Generate simulated data
    data = joblib.load(data_fname)
    X, y, beta = data['X'], data['y'], data['beta']
    ex.add_artifact(data_fname)
    if method == 'gibbs':
        output = joblib.Parallel(n_jobs=joblib.cpu_count())(
            joblib.delayed(draw_samples_gibbs)(
                n_warm_up_samples + n_samples,
                X,
                y,
                sigma,
                use_efficient_proposal=use_efficient_proposal,
            )
            for _ in range(n_chains)
        )
        gamma_samples, beta_samples = list(zip(*output))
        gamma_samples, beta_samples = np.stack(gamma_samples), np.stack(beta_samples)
        results = {
            'beta': beta_samples,
            'gamma': gamma_samples,
            'n_warm_up_samples': n_warm_up_samples,
        }
        for method in ['mean', 'bulk', 'tail']:
            print(
                'Min ess {}: '.format(method),
                get_min_ess(beta_samples[:, n_warm_up_samples:], method=method),
            )
    elif method == 'pymc3':
        gamma_samples, beta_samples, accept_array = draw_samples_pymc3(
            n_warm_up_samples, n_samples, X, y, sigma, n_chains=n_chains
        )
        gamma_samples = gamma_samples.reshape((n_chains, -1, beta.shape[0]))
        beta_samples = beta_samples.reshape((n_chains, -1, beta.shape[0]))
        accept_array = accept_array.reshape((n_chains, -1))
        acceptance_rate = np.mean(accept_array, axis=1)
        print(
            'Filtering out {} pathological chains out of {}'.format(
                np.sum(acceptance_rate <= 0.4), len(acceptance_rate)
            )
        )
        gamma_samples, beta_samples, accept_array = (
            gamma_samples[acceptance_rate > 0.4],
            beta_samples[acceptance_rate > 0.4],
            accept_array[acceptance_rate > 0.4],
        )
        print(np.mean(accept_array))
        results = {
            'beta': beta_samples,
            'gamma': gamma_samples,
            'accept_array': accept_array,
            'n_warm_up_samples': 0,
        }
        for method in ['mean', 'bulk', 'tail']:
            print(
                'Min ess {}: '.format(method), get_min_ess(beta_samples, method=method)
            )
    elif method == 'mixed_hmc':
        output = joblib.Parallel(n_jobs=joblib.cpu_count())(
            joblib.delayed(draw_samples_mixed_hmc)(
                n_samples=n_warm_up_samples + n_samples,
                X=X,
                y=y,
                sigma=sigma,
                epsilon=epsilon,
                total_travel_time=total_travel_time,
                L=L,
                n_discrete_to_update=n_discrete_to_update,
                progbar=False,
            )
            for _ in range(n_chains)
        )
        gamma_samples, beta_samples, accept_array = list(zip(*output))
        gamma_samples, beta_samples, accept_array = (
            np.stack(gamma_samples),
            np.stack(beta_samples),
            np.stack(accept_array),
        )
        acceptance_rate = np.mean(accept_array, axis=1)
        print(
            'Filtering out {} pathological chains'.format(
                np.sum(acceptance_rate <= 0.4)
            )
        )
        gamma_samples, beta_samples, accept_array = (
            gamma_samples[acceptance_rate > 0.4],
            beta_samples[acceptance_rate > 0.4],
            accept_array[acceptance_rate > 0.4],
        )
        print(np.mean(accept_array))
        results = {
            'beta': beta_samples,
            'gamma': gamma_samples,
            'accept_array': accept_array,
            'n_warm_up_samples': n_warm_up_samples,
        }
        for method in ['mean', 'bulk', 'tail']:
            print(
                'Min ess {}: '.format(method),
                get_min_ess(beta_samples[:, n_warm_up_samples:], method=method),
            )
    elif method == 'hmc_within_gibbs':
        output = joblib.Parallel(n_jobs=joblib.cpu_count())(
            joblib.delayed(draw_samples_hmc_within_gibbs)(
                n_samples=n_warm_up_samples + n_samples,
                X=X,
                y=y,
                sigma=sigma,
                epsilon=epsilon,
                L=L,
            )
            for _ in range(n_chains)
        )
        gamma_samples, beta_samples, accept_array = list(zip(*output))
        gamma_samples, beta_samples, accept_array = (
            np.stack(gamma_samples),
            np.stack(beta_samples),
            np.stack(accept_array),
        )
        acceptance_rate = np.mean(accept_array, axis=1)
        print(
            'Filtering out {} pathological chains'.format(
                np.sum(acceptance_rate <= 0.4)
            )
        )
        gamma_samples, beta_samples, accept_array = (
            gamma_samples[acceptance_rate > 0.4],
            beta_samples[acceptance_rate > 0.4],
            accept_array[acceptance_rate > 0.4],
        )
        print(np.mean(accept_array))
        results = {
            'beta': beta_samples,
            'gamma': gamma_samples,
            'accept_array': accept_array,
            'n_warm_up_samples': n_warm_up_samples,
        }
        for method in ['mean', 'bulk', 'tail']:
            print(
                'Min ess {}: '.format(method),
                get_min_ess(beta_samples[:, n_warm_up_samples:], method=method),
            )
    elif method == 'dhmc':
        output = joblib.Parallel(n_jobs=joblib.cpu_count())(
            joblib.delayed(draw_samples_dhmc)(
                n_samples=n_warm_up_samples + n_samples,
                X=X,
                y=y,
                sigma=sigma,
                epsilon=epsilon,
                L=L,
                progbar=False,
            )
            for _ in range(n_chains)
        )
        gamma_samples, beta_samples, accept_array = list(zip(*output))
        gamma_samples, beta_samples, accept_array = (
            np.stack(gamma_samples),
            np.stack(beta_samples),
            np.stack(accept_array),
        )
        acceptance_rate = np.mean(accept_array, axis=1)
        print(
            'Filtering out {} pathological chains'.format(
                np.sum(acceptance_rate <= 0.4)
            )
        )
        gamma_samples, beta_samples, accept_array = (
            gamma_samples[acceptance_rate > 0.4],
            beta_samples[acceptance_rate > 0.4],
            accept_array[acceptance_rate > 0.4],
        )
        print(np.mean(accept_array))
        results = {
            'beta': beta_samples,
            'gamma': gamma_samples,
            'accept_array': accept_array,
            'n_warm_up_samples': n_warm_up_samples,
        }
        for method in ['mean', 'bulk', 'tail']:
            print(
                'Min ess {}: '.format(method),
                get_min_ess(beta_samples[:, n_warm_up_samples:], method=method),
            )
    else:
        raise ValueError('Unsupported method {}'.format(method))

    if n_warm_up_samples + n_samples == gamma_samples.shape[1]:
        samples = gamma_samples[:, n_warm_up_samples:]
    else:
        samples = gamma_samples

    samples = samples.reshape((-1, samples.shape[-1]))
    bin_count = np.bincount(
        np.sum(samples == (beta > 0).astype(np.int32).reshape((1, -1)), axis=1),
        minlength=beta.shape[0],
    )
    print('Bin count: {}, total {}'.format(bin_count, np.sum(bin_count)))
    print('Correct percentage: {}'.format(bin_count[-1] / np.sum(bin_count)))
    print(
        'Hamming distance: {}'.format(
            np.mean(
                np.sum(samples != (beta > 0).astype(np.int32).reshape((1, -1)), axis=1)
            )
            / beta.shape[0]
        )
    )
    results_fname = '{}/results.joblib'.format(temp_folder_name)
    joblib.dump(results, results_fname)
    ex.add_artifact(results_fname)
    temp_folder.cleanup()


# Gibbs experiments
ex.run(config_updates={'method': 'gibbs'})
# Mixed HMC experiments
for L in [200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    for total_travel_time in [30, 40, 50, 60, 70]:
        print(
            'Working on mixed HMC, L {}, total_travel_time {}'.format(
                L, total_travel_time
            )
        )
        ex.run(
            config_updates={
                'method': 'mixed_hmc',
                'epsilon': 0.06,
                'L': L,
                'total_travel_time': total_travel_time,
                'n_discrete_to_update': 1,
            }
        )

for total_travel_time in [30, 40, 50, 60, 70]:
    for n_discrete_to_update, L in [
        (2, 300),
        (3, 200),
        (4, 150),
        (5, 120),
        (6, 100),
        (10, 60),
        (20, 30),
    ]:
        print(
            'Working on mixed HMC, L {}, n_discrete_to_update {}'.format(
                L, n_discrete_to_update
            )
        )
        ex.run(
            config_updates={
                'method': 'mixed_hmc',
                'epsilon': 0.06,
                'L': L,
                'total_travel_time': total_travel_time,
                'n_discrete_to_update': n_discrete_to_update,
            }
        )

# PyMC3 experiments
ex.run(config_updates={'method': 'pymc3'})
# DHMC experiments
for epsilon in [
    [0.03, 0.05],
    [0.05, 0.07],
    [0.07, 0.09],
    [0.09, 0.11],
    [0.11, 0.13],
    [0.13, 0.15],
]:
    for L in [
        [30, 50],
        [50, 70],
        [70, 90],
        [90, 110],
        [110, 130],
        [130, 150],
        [150, 170],
        [170, 190],
    ]:
        print('Working on DHMC, epsilon {}, L {}'.format(epsilon, L))
        ex.run(config_updates={'method': 'dhmc', 'epsilon': epsilon, 'L': L})

# HMC-within-Gibbs experiments
for epsilon in [0.04, 0.06, 0.08, 0.10, 0.12, 0.14]:
    for L in [40, 60, 80, 100, 120, 140, 160, 180]:
        print('Working on HMC-within-Gibbs, epsilon {}, L {}'.format(epsilon, L))
        ex.run(
            config_updates={'method': 'hmc_within_gibbs', 'epsilon': epsilon, 'L': L}
        )
