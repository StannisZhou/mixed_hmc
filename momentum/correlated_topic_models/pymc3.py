import numpy as np

import joblib
import pymc3 as pm


def draw_samples_pymc3(n_warm_up_samples, n_samples, w, mu, Sigma, beta, n_chains=1):
    K = beta.shape[0]
    model = pm.Model()

    with model:
        eta = pm.MvNormal('eta', mu=mu, cov=Sigma, shape=(K - 1,))
        eta_full = pm.math.concatenate(
            [pm.Normal('eta0', mu=0, sigma=1, shape=(1,), observed=np.zeros(1)), eta]
        )
        theta = pm.math.exp(eta_full) / pm.math.sum(pm.math.exp(eta_full))
        z = pm.Categorical('z', p=theta, shape=w.shape)
        beta_pymc3 = pm.Normal('beta', mu=0, sigma=1, shape=beta.shape, observed=beta)
        doc = pm.Categorical('doc', p=beta_pymc3[z], shape=w.shape, observed=w)

    # fit model
    with model:
        step1 = pm.NUTS(vars=[eta])
        step2 = pm.ElemwiseCategorical(vars=[z])
        tr = pm.sample(
            n_samples,
            step=[step1, step2],
            tune=n_warm_up_samples,
            chains=n_chains,
            cores=min(joblib.cpu_count(), n_chains),
        )

    z_samples = tr.get_values('z')
    eta_samples = tr.get_values('eta')
    accept_array = tr.get_sampler_stats('mean_tree_accept')
    return z_samples, eta_samples, accept_array
