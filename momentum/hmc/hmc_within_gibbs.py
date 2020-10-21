import jax
from jax import numpy as np
from jax.nn import softmax
from momentum.utils import categorical
from numpyro.util import fori_collect


def hmc_within_gibbs(
    q0_discrete,
    q0_continuous,
    n_samples,
    epsilon,
    L,
    key,
    labels_for_discrete,
    potential,
    grad_potential=None,
    mode='RW',
    use_random_scan=True,
):
    if grad_potential is None:
        grad_potential = jax.grad(potential, argnums=1)

    def leapfrog_step(q_discrete, q_continuous, p_continuous, step_size):
        p_continuous -= 0.5 * step_size * grad_potential(q_discrete, q_continuous)
        q_continuous += step_size * p_continuous
        p_continuous -= 0.5 * step_size * grad_potential(q_discrete, q_continuous)
        return q_continuous, p_continuous

    def take_hmc_step(q_discrete, q0_continuous, epsilon, L, key):
        # Resample momentum p
        key, subkey = jax.random.split(key)
        p0_continuous = jax.random.normal(subkey, shape=q0_continuous.shape)
        # Initialize q, p
        q_continuous = np.array(q0_continuous)
        p_continuous = np.array(p0_continuous)
        q_continuous, p_continuous = jax.lax.fori_loop(
            0,
            L,
            lambda ii, val: leapfrog_step(
                q_discrete=q_discrete,
                q_continuous=val[0],
                p_continuous=val[1],
                step_size=epsilon,
            ),
            (q_continuous, p_continuous),
        )
        # Accept or reject
        current_U = potential(q_discrete, q0_continuous)
        current_K = 0.5 * np.sum(p0_continuous ** 2)
        proposed_U = potential(q_discrete, q_continuous)
        proposed_K = 0.5 * np.sum(p_continuous ** 2)
        key, subkey = jax.random.split(key)
        accept = jax.random.uniform(subkey) < np.exp(
            current_U - proposed_U + current_K - proposed_K
        )
        q_continuous = jax.lax.cond(
            accept, q_continuous, lambda x: x, q0_continuous, lambda x: x
        )
        return q_continuous, accept, key

    def gibbs_step_for_ind(q_discrete, q_continuous, ind, key, labels_for_discrete):
        # Get potential array and distribution
        if mode == 'RW':
            proposal_dist = np.ones(labels_for_discrete[ind].shape[0])
            proposal_dist = jax.ops.index_update(
                proposal_dist, jax.ops.index[q_discrete[ind]], 0
            )
            proposal_dist /= np.sum(proposal_dist)
            key, subkey = jax.random.split(key)
            proposal_for_ind = categorical(subkey, proposal_dist)
            q_discrete_proposal = jax.ops.index_update(
                q_discrete, jax.ops.index[ind], proposal_for_ind
            )
            delta_E = potential(q_discrete_proposal, q_continuous) - potential(
                q_discrete, q_continuous
            )
        elif mode == 'GB':
            _, potential_array = jax.lax.scan(
                lambda carry, ii: (
                    None,
                    potential(
                        jax.ops.index_update(q_discrete, jax.ops.index[ind], ii),
                        q_continuous,
                    ),
                ),
                None,
                np.arange(labels_for_discrete[ind].shape[0]),
            )
            distribution = softmax(-potential_array)
            # Get proposal and make proposal
            proposal_dist = jax.ops.index_update(
                distribution, jax.ops.index[q_discrete[ind]], 0
            )
            proposal_dist += 1e-12
            proposal_dist /= np.sum(proposal_dist)
            key, subkey = jax.random.split(key)
            proposal_for_ind = categorical(subkey, proposal_dist)
            q_discrete_proposal = jax.ops.index_update(
                q_discrete, jax.ops.index[ind], proposal_for_ind
            )
            delta_E = np.log(
                1 - distribution[q_discrete_proposal[ind]] + 1e-12
            ) - np.log(1 - distribution[q_discrete[ind]] + 1e-12)
        else:
            assert False

        # Decide whether to accept or reject
        key, subkey = jax.random.split(key)
        accept = jax.random.exponential(subkey) > delta_E
        q_discrete = jax.lax.cond(
            accept, q_discrete_proposal, lambda x: x, q_discrete, lambda x: x
        )
        return q_discrete, key

    def take_hmc_gibbs_step(
        q0_discrete,
        q0_continuous,
        key,
        epsilon,
        L,
        labels_for_discrete,
        visitation_order,
    ):
        """take_hmc_gibbs_step
        Parameters
        ----------
        q0_discrete :
        q0_continuous : HMCState
        key :
        potential :
        labels_for_discrete :
        mode :
        Returns
        -------
        """
        q_continuous, accept, key = take_hmc_step(
            q_discrete=q0_discrete,
            q0_continuous=q0_continuous,
            epsilon=epsilon,
            L=L,
            key=key,
        )
        q_discrete, key = jax.lax.fori_loop(
            0,
            visitation_order.shape[0],
            lambda ii, val: gibbs_step_for_ind(
                q_discrete=val[0],
                q_continuous=q_continuous,
                ind=visitation_order[ii],
                key=val[1],
                labels_for_discrete=labels_for_discrete,
            ),
            (q0_discrete, key),
        )
        return q_discrete, q_continuous, accept, key

    key, subkey = jax.random.split(key)
    init_visitation_order = jax.random.shuffle(
        subkey, np.arange(labels_for_discrete.shape[0])
    )
    q_discrete = q0_discrete
    q_continuous = q0_continuous

    def body_fun(val):
        q_discrete, q_continuous, _, key = val
        key, subkey = jax.random.split(key)
        visitation_order = jax.lax.cond(
            use_random_scan,
            jax.random.shuffle(subkey, np.arange(labels_for_discrete.shape[0])),
            lambda x: x,
            init_visitation_order,
            lambda x: x,
        )
        q_discrete, q_continuous, accept, key = take_hmc_gibbs_step(
            q0_discrete=q_discrete,
            q0_continuous=q_continuous,
            key=key,
            epsilon=epsilon,
            L=L,
            labels_for_discrete=labels_for_discrete,
            visitation_order=visitation_order,
        )
        return q_discrete, q_continuous, accept, key

    output = fori_collect(
        0, n_samples, body_fun, (q_discrete, q_continuous, False, key)
    )
    q_discrete_array, q_continuous_array, accept_array, _ = output
    return q_discrete_array, q_continuous_array, accept_array
