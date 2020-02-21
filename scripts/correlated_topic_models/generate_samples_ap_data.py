import os
import tempfile

import numpy as np

import joblib
import sacred
from momentum.correlated_topic_models.dhmc import draw_samples_dhmc
from momentum.correlated_topic_models.gibbs import draw_samples_gibbs
from momentum.correlated_topic_models.mixed_hmc import draw_samples_mixed_hmc
from momentum.correlated_topic_models.pymc3 import draw_samples_pymc3
from momentum.diagnostics.ess import get_min_ess
from sacred.observers import FileStorageObserver

log_folder = os.path.expanduser('~/logs/correlated_topic_models_results')
ex = sacred.Experiment('correlated_topic_models_generate_samples')
ex.observers.append(FileStorageObserver.create(log_folder))


@ex.config
def config():
    method = 'pymc3'
    n_warm_up_samples = int(1e3)
    n_samples = int(4e3)
    epsilon = 4
    #  epsilon = [3, 4]
    L_multiplier = 80
    L = [40, 50]
    total_travel_time = 600
    n_discrete_to_update = 1
    n_chains = 96
    n_short_to_exclude = 20
    n_long_to_exclude = 400
    n_documents = 20


@ex.main
def run(
    n_short_to_exclude,
    n_long_to_exclude,
    n_documents,
    method,
    n_warm_up_samples,
    n_samples,
    n_chains,
    epsilon,
    L_multiplier,
    L,
    total_travel_time,
    n_discrete_to_update,
):
    # Generate temp folder
    temp_folder = tempfile.TemporaryDirectory()
    temp_folder_name = temp_folder.name
    # Load data
    data_fname = 'ap_data.joblib'
    ap_data = joblib.load(data_fname)
    documents, K, mu, Sigma, beta = (
        ap_data['documents'],
        ap_data['K'],
        ap_data['mu'],
        ap_data['Sigma'],
        ap_data['beta'],
    )
    document_lengths = np.array([len(w) for w in documents])
    shortest_indices = np.argsort(document_lengths)
    indices = shortest_indices[
        np.floor(
            np.linspace(
                n_short_to_exclude, len(documents) - n_long_to_exclude, n_documents
            )
        ).astype(np.int32)
    ]
    ex.add_artifact(data_fname)
    for ind in indices:
        print(
            'Working on document {} of length {}, method {}'.format(
                ind, len(documents[ind]), method
            )
        )
        if method == 'gibbs':
            output = joblib.Parallel(n_jobs=joblib.cpu_count())(
                joblib.delayed(draw_samples_gibbs)(
                    n_warm_up_samples + n_samples, documents[ind], mu, Sigma, beta
                )
                for _ in range(n_chains)
            )
            z_samples, eta_samples = list(zip(*output))
            z_samples, eta_samples = (np.stack(z_samples), np.stack(eta_samples))
            results = {
                'eta': eta_samples,
                'z': z_samples,
                'n_warm_up_samples': n_warm_up_samples,
            }
            print(
                'Mean: {}'.format(
                    np.mean(
                        eta_samples[:, n_warm_up_samples:].reshape((-1, K - 1)), axis=0
                    )
                )
            )
            for ess_method in ['mean', 'bulk', 'tail']:
                print(
                    'Min ess {}: '.format(ess_method),
                    get_min_ess(eta_samples[:, n_warm_up_samples:], method=ess_method),
                )
        elif method == 'pymc3':
            z_samples, eta_samples, accept_array = draw_samples_pymc3(
                n_warm_up_samples,
                n_samples,
                documents[ind],
                mu,
                Sigma,
                beta,
                n_chains=n_chains,
            )
            z_samples = z_samples.reshape((n_chains, -1, len(documents[ind])))
            eta_samples = eta_samples.reshape((n_chains, -1, K - 1))
            accept_array = accept_array.reshape((n_chains, -1))
            acceptance_rate = np.mean(accept_array, axis=1)
            print(
                'Filtering out {} pathological chains out of {}'.format(
                    np.sum(acceptance_rate <= 0.4), len(acceptance_rate)
                )
            )
            z_samples, eta_samples, accept_array = (
                z_samples[acceptance_rate > 0.4],
                eta_samples[acceptance_rate > 0.4],
                accept_array[acceptance_rate > 0.4],
            )
            print(np.mean(accept_array))
            results = {
                'eta': eta_samples,
                'z': z_samples,
                'accept_array': accept_array,
                'n_warm_up_samples': 0,
            }
            print('Mean: {}'.format(np.mean(eta_samples.reshape((-1, K - 1)), axis=0)))
            for ess_method in ['mean', 'bulk', 'tail']:
                print(
                    'Min ess {}: '.format(ess_method),
                    get_min_ess(eta_samples, method=ess_method),
                )
        elif method == 'mixed_hmc':
            L = len(documents[ind]) * L_multiplier
            adaptive_step_size = np.array(np.diag(Sigma))
            adaptive_step_size /= np.sum(adaptive_step_size)
            output = joblib.Parallel(n_jobs=joblib.cpu_count())(
                joblib.delayed(draw_samples_mixed_hmc)(
                    n_samples=n_warm_up_samples + n_samples,
                    w=documents[ind],
                    mu=mu,
                    Sigma=Sigma,
                    beta=beta,
                    epsilon=epsilon,
                    total_travel_time=total_travel_time,
                    L=L,
                    n_discrete_to_update=n_discrete_to_update,
                    progbar=False,
                    adaptive_step_size=adaptive_step_size,
                )
                for _ in range(n_chains)
            )
            z_samples, eta_samples, accept_array = list(zip(*output))
            z_samples, eta_samples, accept_array = (
                np.stack(z_samples),
                np.stack(eta_samples),
                np.stack(accept_array),
            )
            acceptance_rate = np.mean(accept_array, axis=1)
            print(
                'Filtering out {} pathological chains'.format(
                    np.sum(acceptance_rate <= 0.4)
                )
            )
            z_samples, eta_samples, accept_array = (
                z_samples[acceptance_rate > 0.4],
                eta_samples[acceptance_rate > 0.4],
                accept_array[acceptance_rate > 0.4],
            )
            print(np.mean(accept_array))
            results = {
                'eta': eta_samples,
                'z': z_samples,
                'accept_array': accept_array,
                'n_warm_up_samples': n_warm_up_samples,
            }
            print(
                'Mean: {}'.format(
                    np.mean(
                        eta_samples[:, n_warm_up_samples:].reshape((-1, K - 1)), axis=0
                    )
                )
            )
            for ess_method in ['mean', 'bulk', 'tail']:
                print(
                    'Min ess {}: '.format(ess_method),
                    get_min_ess(eta_samples[:, n_warm_up_samples:], method=ess_method),
                )
        elif method == 'dhmc':
            adaptive_step_size = np.array(np.diag(Sigma))
            adaptive_step_size /= np.sum(adaptive_step_size)
            output = joblib.Parallel(n_jobs=joblib.cpu_count())(
                joblib.delayed(draw_samples_dhmc)(
                    n_samples=n_warm_up_samples + n_samples,
                    w=documents[ind],
                    mu=mu,
                    Sigma=Sigma,
                    beta=beta,
                    epsilon=epsilon,
                    L=L,
                    progbar=False,
                    adaptive_step_size=adaptive_step_size,
                )
                for _ in range(n_chains)
            )
            z_samples, eta_samples, accept_array = list(zip(*output))
            z_samples, eta_samples, accept_array = (
                np.stack(z_samples),
                np.stack(eta_samples),
                np.stack(accept_array),
            )
            acceptance_rate = np.mean(accept_array, axis=1)
            print(
                'Filtering out {} pathological chains'.format(
                    np.sum(acceptance_rate <= 0.4)
                )
            )
            z_samples, eta_samples, accept_array = (
                z_samples[acceptance_rate > 0.4],
                eta_samples[acceptance_rate > 0.4],
                accept_array[acceptance_rate > 0.4],
            )
            print(np.mean(accept_array))
            results = {
                'eta': eta_samples,
                'z': z_samples,
                'accept_array': accept_array,
                'n_warm_up_samples': n_warm_up_samples,
            }
            print(
                'Mean: {}'.format(
                    np.mean(
                        eta_samples[:, n_warm_up_samples:].reshape((-1, K - 1)), axis=0
                    )
                )
            )
            for ess_method in ['mean', 'bulk', 'tail']:
                print(
                    'Min ess {}: '.format(ess_method),
                    get_min_ess(eta_samples[:, n_warm_up_samples:], method=ess_method),
                )
        else:
            raise ValueError('Unsupported method {}'.format(method))

        results_fname = '{}/results_document_{}.joblib'.format(temp_folder_name, ind)
        joblib.dump(results, results_fname)
        ex.add_artifact(results_fname)

    temp_folder.cleanup()


# Gibbs experiments
ex.run(config_updates={'method': 'gibbs'})
# PyMC3 experiments
ex.run(config_updates={'method': 'pymc3'})
# mixed HMC experiments
ex.run(config_updates={'method': 'mixed_hmc'})
