import joblib
import pymc3 as pm


def draw_samples_pymc3(
    n_warm_up_samples, n_samples, pi, mu_list, Sigma_list, n_chains=1
):
    # setup model
    model = pm.Model()
    with model:
        means = pm.Normal('means', mu=0, sigma=1, shape=mu_list.shape, observed=mu_list)
        Sigmas = pm.Normal(
            'Sigmas', mu=0, sigma=1, shape=Sigma_list.shape, observed=Sigma_list
        )
        # latent cluster of each observation
        category = pm.Categorical('category', p=pi)
        # likelihood for each observed value
        points = pm.MvNormal(
            'obs', mu=means[category], cov=Sigmas[category], shape=mu_list.shape[1]
        )

    # fit model
    with model:
        step1 = pm.NUTS(vars=[points])
        step2 = pm.ElemwiseCategorical(vars=[category])
        tr = pm.sample(
            n_samples,
            step=[step1, step2],
            tune=n_warm_up_samples,
            chains=n_chains,
            cores=joblib.cpu_count(),
        )

    membership_samples = tr.get_values('category')
    samples = tr.get_values('obs')
    return membership_samples, samples
