import numpy as np

import joblib
from momentum.variable_selection.gibbs import logistic

N = 100
p = 20
Sigma = 0.3 * np.ones((p, p))
np.fill_diagonal(Sigma, 3)

X = np.random.multivariate_normal(np.zeros(p), Sigma, size=N)
beta_true = np.zeros(p)
beta_true[[1, 3, 5, 7, 9]] = 0.5
y = np.random.binomial(1, logistic(np.dot(X, beta_true)))


data = {'N': N, 'p': p, 'Sigma': Sigma, 'X': X, 'beta': beta_true, 'y': y}
joblib.dump(data, 'simulated_data.joblib')
