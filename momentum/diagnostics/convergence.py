import numpy as np

import joblib
import numba
from tqdm import tqdm


@numba.jit(nopython=True, cache=True)
def get_max_ks_stat(full_data1, full_data2):
    """get_max_ks_stat

    Parameters
    ----------
    full_data1 : np.array
        Of shape (n, d)
    full_data2 : np.array
        Of shape (n, d)
    Returns
    -------
    """
    n1, p1 = full_data1.shape
    n2, p2 = full_data2.shape
    assert p1 == p2
    ks_stats = np.zeros(p1)
    for ii in range(p1):
        data1 = np.sort(full_data1[:, ii])
        data2 = np.sort(full_data2[:, ii])
        data_all = np.concatenate((data1, data2))
        # using searchsorted solves equal data problem
        cdf1 = np.searchsorted(data1, data_all, side='right') / n1
        cdf2 = np.searchsorted(data2, data_all, side='right') / n2
        cddiffs = cdf1 - cdf2
        minS = -np.min(cddiffs)
        maxS = np.max(cddiffs)
        d = max(minS, maxS)
        lcm = np.lcm(n1, n2)
        h = int(np.round(d * lcm))
        ks_stats[ii] = h * 1.0 / lcm

    return np.max(ks_stats)


@numba.jit(nopython=True, cache=True)
def get_ks_evolution(samples, reference_samples, print_freq=100):
    n = samples.shape[0]
    ks_stats = np.zeros(n)
    ks_stats[0] = 1
    for ii in range(1, n):
        if ii % print_freq == 0:
            print(ii / n)

        ks_stats[ii] = get_max_ks_stat(samples[:ii], reference_samples)

    return ks_stats


def get_ks_evolution_multiple_chains(samples, reference_samples, dim=0, print_freq=100):
    assert samples.ndim == 3
    assert reference_samples.ndim == 2
    n_chains, n_samples, n_dim = samples.shape
    ks_stats = joblib.Parallel(n_jobs=joblib.cpu_count())(
        joblib.delayed(get_ks_evolution)(
            samples[ii, :, dim].reshape((-1, 1)),
            reference_samples[:, dim].reshape((-1, 1)),
            print_freq=print_freq,
        )
        for ii in tqdm(range(n_chains))
    )
    return np.array(ks_stats)
