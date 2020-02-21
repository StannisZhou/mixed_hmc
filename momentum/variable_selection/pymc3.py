import numpy as np

import joblib
import pymc3 as pm


def draw_samples_pymc3(n_warm_up_samples, n_samples, X, y, sigma, n_chains=1):
    # setup model
    model = pm.Model()
    with model:
        beta = pm.Normal('beta', mu=[0], sigma=sigma, shape=X.shape[1])
        gamma = pm.Bernoulli('gamma', p=0.5, shape=X.shape[1])
        prob = pm.invlogit(pm.math.dot(X, beta * gamma))
        data = pm.Bernoulli('data', p=prob, observed=y)

    # fit model
    with model:
        step1 = pm.NUTS(vars=[beta])
        step2 = pm.BinaryGibbsMetropolis(vars=[gamma])
        tr = pm.sample(
            n_samples,
            step=[step1, step2],
            tune=n_warm_up_samples,
            chains=n_chains,
            cores=min(joblib.cpu_count(), n_chains),
        )

    beta_samples = tr.get_values('beta')
    gamma_samples = tr.get_values('gamma')
    accept_array = tr.get_sampler_stats('mean_tree_accept')
    return gamma_samples, beta_samples, accept_array
