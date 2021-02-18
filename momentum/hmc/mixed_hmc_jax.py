import jax
import jax.numpy as np
from momentum.utils import categorical
from numpyro.util import fori_collect


def mixed_hmc_on_joint(
    q0_discrete,
    q0_continuous,
    n_samples,
    epsilon,
    L,
    key,
    labels_for_discrete,
    potential,
    total_travel_time=None,
    adaptive_step_size=None,
    grad_potential=None,
    n_discrete_to_update=1,
    mode='RW',
    progbar=True,
):
    if total_travel_time is None:
        total_travel_time = epsilon * L

    if adaptive_step_size is None:
        adaptive_step_size = np.ones_like(q0_continuous)

    if grad_potential is None:
        grad_potential = jax.grad(potential, argnums=1)

    def get_step_sizes_n_steps(
        epsilon, total_trave_time, L, n_discrete_variables, n_discrete_to_update, key
    ):
        # Generate random step sizes
        key, subkey = jax.random.split(key)
        step_size_list = jax.random.dirichlet(subkey, np.ones(n_discrete_variables + 1))
        step_size_list, last_step_size = step_size_list[:-1], step_size_list[-1]
        step_size_list = jax.ops.index_update(
            step_size_list, jax.ops.index[0], step_size_list[0] + last_step_size
        )
        n_repeats = int(np.ceil(L * n_discrete_to_update / n_discrete_variables))
        step_size_list = np.tile(step_size_list, n_repeats)[: L * n_discrete_to_update]
        step_size_list = jax.ops.index_update(
            step_size_list, jax.ops.index[0], step_size_list[0] - last_step_size
        )
        step_size_list = np.sum(
            step_size_list.reshape((L, n_discrete_to_update)), axis=1
        )
        step_size_list = total_travel_time * step_size_list / np.sum(step_size_list)
        n_steps_list = np.ceil(step_size_list / epsilon).astype(np.int32)
        step_size_list /= n_steps_list
        return step_size_list, n_steps_list, key

    def take_mixed_hmc_step(
        q0_discrete,
        q0_continuous,
        epsilon,
        total_travel_time,
        L,
        labels_for_discrete,
        key,
        n_discrete_to_update=1,
        mode='RW',
    ):
        def take_one_mixed_hmc_step(ii, val):
            q_discrete, k_discrete, q_continuous, p_continuous, delta_U, key = val
            q_continuous, p_continuous = jax.lax.fori_loop(
                0,
                n_steps_list[ii],
                lambda jj, state: leapfrog_step(
                    q_discrete=q_discrete,
                    q_continuous=state[0],
                    p_continuous=state[1],
                    step_size=step_size_list[ii],
                ),
                (q_continuous, p_continuous),
            )
            start_ind = ii * n_discrete_to_update
            indices = np.roll(visitation_order, -start_ind)[:n_discrete_to_update]
            q_discrete, k_discrete, delta_U, key = jax.lax.fori_loop(
                0,
                indices.shape[0],
                lambda jj, state: momentum_step_for_ind(
                    q_discrete=state[0],
                    q_continuous=q_continuous,
                    k_discrete=state[1],
                    delta_U=state[2],
                    key=state[3],
                    labels_for_discrete=labels_for_discrete,
                    ind=indices[jj],
                    mode=mode,
                ),
                (q_discrete, k_discrete, delta_U, key),
            )
            return q_discrete, k_discrete, q_continuous, p_continuous, delta_U, key

        n_discrete_variables = q0_discrete.shape[0]
        # Resample momentum
        key, subkey = jax.random.split(key)
        p0_continuous = jax.random.normal(subkey, shape=q0_continuous.shape)
        # Resample kinetic energy and visitation order for discrete random variables
        key, subkey = jax.random.split(key)
        k0_discrete = jax.random.exponential(subkey, shape=(n_discrete_variables,))
        key, subkey = jax.random.split(key)
        visitation_order = jax.random.shuffle(subkey, np.arange(n_discrete_variables))
        # Initialize q, k
        q_discrete = np.array(q0_discrete)
        q_continuous = np.array(q0_continuous)
        k_discrete = np.array(k0_discrete)
        p_continuous = np.array(p0_continuous)
        # Get step sizes and n_steps
        step_size_list, n_steps_list, key = get_step_sizes_n_steps(
            epsilon,
            total_travel_time,
            L,
            n_discrete_variables,
            n_discrete_to_update,
            key,
        )
        # Take L steps
        (
            q_discrete,
            k_discrete,
            q_continuous,
            p_continuous,
            delta_U,
            key,
        ) = jax.lax.fori_loop(
            0,
            L,
            take_one_mixed_hmc_step,
            (q_discrete, k_discrete, q_continuous, p_continuous, 0.0, key),
        )
        # Accept or reject
        current_E = potential(q0_discrete, q0_continuous) + 0.5 * np.sum(
            p0_continuous ** 2
        )
        proposed_E = potential(q_discrete, q_continuous) + 0.5 * np.sum(
            p_continuous ** 2
        )
        key, subkey = jax.random.split(key)
        accept = jax.random.uniform(subkey) < np.exp(current_E + delta_U - proposed_E)
        q_discrete, q_continuous = jax.lax.cond(
            accept,
            (q_discrete, q_continuous),
            lambda x: x,
            (q0_discrete, q0_continuous),
            lambda x: x,
        )
        return q_discrete, q_continuous, accept, key

    def leapfrog_step(q_discrete, q_continuous, p_continuous, step_size):
        p_continuous -= (
            0.5
            * step_size
            * adaptive_step_size
            * grad_potential(q_discrete, q_continuous)
        )
        q_continuous += step_size * adaptive_step_size * p_continuous
        p_continuous -= (
            0.5
            * step_size
            * adaptive_step_size
            * grad_potential(q_discrete, q_continuous)
        )
        return q_continuous, p_continuous

    def momentum_step_for_ind(
        q_discrete,
        q_continuous,
        k_discrete,
        delta_U,
        key,
        labels_for_discrete,
        ind,
        mode='RW',
    ):
        print(mode)
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
            proposal_dist /= np.sum(proposal_dist)
            key, subkey = jax.random.split(key)
            proposal_for_ind = categorical(subkey, proposal_dist)
            q_discrete_proposal = jax.ops.index_update(
                q_discrete, jax.ops.index[ind], proposal_for_ind
            )
            delta_E = np.log(1 - distribution[q_discrete_proposal[ind]]) - np.log(
                1 - distribution[q_discrete[ind]]
            )
        elif mode == 'gibbs':
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
            proposal_dist = distribution
            key, subkey = jax.random.split(key)
            proposal_for_ind = categorical(subkey, proposal_dist)
            q_discrete_proposal = jax.ops.index_update(
                q_discrete, jax.ops.index[ind], proposal_for_ind
            )
            delta_E = 0
        else:
            assert False

        # Decide whether to accept or reject
        accept = k_discrete[ind] > delta_E
        k_discrete_proposal = jax.ops.index_update(
            k_discrete, jax.ops.index[ind], k_discrete[ind] - delta_E
        )
        delta_U_proposal = (
            delta_U
            + potential(q_discrete_proposal, q_continuous)
            - potential(q_discrete, q_continuous)
        )
        q_discrete, k_discrete, delta_U = jax.lax.cond(
            accept,
            (q_discrete_proposal, k_discrete_proposal, delta_U_proposal),
            lambda x: x,
            (q_discrete, k_discrete, delta_U),
            lambda x: x,
        )
        return q_discrete, k_discrete, delta_U, key

    def body_fun(val):
        q_discrete, q_continuous, accept, key = val
        q_discrete, q_continuous, accept, key = take_mixed_hmc_step(
            q0_discrete=q_discrete,
            q0_continuous=q_continuous,
            epsilon=epsilon,
            total_travel_time=total_travel_time,
            L=L,
            labels_for_discrete=labels_for_discrete,
            key=key,
            mode=mode,
        )
        return q_discrete, q_continuous, accept, key

    q_discrete = q0_discrete
    q_continuous = q0_continuous
    output = fori_collect(
        0, n_samples, body_fun, (q_discrete, q_continuous, False, key), progbar=progbar
    )
    q_discrete_array, q_continuous_array, accept_array, _ = output
    return q_discrete_array, q_continuous_array, accept_array


def softmax(x):
    expx = np.exp(x)
    return expx / np.sum(expx)
