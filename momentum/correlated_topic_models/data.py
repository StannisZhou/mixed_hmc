import os

import numpy as np

from gensim.corpora.bleicorpus import BleiCorpus


def load_ldac_data(fname):
    corpus = BleiCorpus(fname)
    with open(fname, 'r') as f:
        lines = f.readlines()

    documents = []
    for line in lines:
        doc = np.array(corpus.line2doc(line), dtype=np.int32)
        w = np.zeros(np.sum(doc[:, 1]), dtype=np.int32)
        start_ind = 0
        for word in doc:
            w[start_ind : start_ind + word[1]] = word[0]
            start_ind += word[1]

        documents.append(w)

    return documents


def load_ctmc_trained_model(folder):
    K, V = np.loadtxt(
        os.path.join(folder, 'final-param.txt'), dtype=np.int32, usecols=1
    )
    mu = np.loadtxt(os.path.join(folder, 'final-mu.dat'))
    Sigma = np.loadtxt(os.path.join(folder, 'final-cov.dat')).reshape((K - 1, K - 1))
    logbeta = np.loadtxt(os.path.join(folder, 'final-log-beta.dat')).reshape((K, V))
    beta = np.exp(logbeta)
    return K, V, mu, Sigma, beta


def generate_synthetic_data(n_words, mu, Sigma, beta):
    K, V = beta.shape
    eta = np.concatenate((np.zeros(1), np.random.multivariate_normal(mu, Sigma)))
    theta = np.exp(eta) / np.sum(np.exp(eta))
    z = np.random.choice(K, size=(n_words,), replace=True, p=theta)

    def sample_words(z):
        return np.random.choice(V, p=beta[z])

    w = np.vectorize(sample_words)(z)
    return eta[1:], w
