import numpy as np

import arviz
import joblib
from tqdm import tqdm


def get_min_ess(samples, method='bulk'):
    """get_min_ess

    Parameters
    ----------
    samples : np.array
        (n_chains, n_samples, n_dim)

    Returns
    -------
    """
    if samples.ndim == 1:
        samples = samples[np.newaixs, :, np.newaxis]
    elif samples.ndim == 2:
        samples = samples[np.newaxis, ...]

    n_chains, n_samples, n_dim = samples.shape
    ess_list = joblib.Parallel(n_jobs=joblib.cpu_count(), prefer='threads')(
        joblib.delayed(arviz.ess)(samples[..., ii], relative=True, method=method)
        for ii in tqdm(range(n_dim))
    )
    ess_list = np.array(ess_list)
    return np.min(ess_list)
