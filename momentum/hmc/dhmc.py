import jax
import jax.numpy as np
from numpyro.util import fori_collect


def dhmc_on_joint(
    q0_embedded,
    q0_continuous,
    n_samples,
    key,
    epsilon_range,
    L_range,
    potential,
    grad_potential=None,
    adaptive_step_size=None,
    progbar=True,
):
    if grad_potential is None:
        grad_potential = jax.grad(potential, argnums=1)

    if adaptive_step_size is None:
        adaptive_step_size = np.ones_like(q0_continuous)

    def take_dhmc_step(q0_embedded, q0_continuous, key, epsilon_range, L_range):
        # Sample epsilon and L
        key, subkey = jax.random.split(key)
        epsilon = (
            jax.random.uniform(subkey) * (epsilon_range[1] - epsilon_range[0])
            + epsilon_range[0]
        )
        key, subkey = jax.random.split(key)
        L = jax.random.randint(subkey, shape=(), minval=L_range[0], maxval=L_range[1])
        # Resample momentum
        key, subkey = jax.random.split(key)
        p0_discrete = jax.random.laplace(subkey, shape=q0_embedded.shape)
        key, subkey = jax.random.split(key)
        p0_continuous = jax.random.normal(subkey, shape=q0_continuous.shape)
        # Initialize q, k
        q_embedded = np.array(q0_embedded)
        q_continuous = np.array(q0_continuous)
        p_discrete = np.array(p0_discrete)
        p_continuous = np.array(p0_continuous)

        # Take L steps
        def body_fun(ii, val):
            q_embedded, q_continuous, p_discrete, p_continuous, key = val
            q_continuous, p_continuous = leapfrog_halfstep1(
                q_embedded=q_embedded,
                q_continuous=q_continuous,
                p_continuous=p_continuous,
                epsilon=epsilon,
            )
            q_embedded, p_discrete, key = coordinatewise(
                q0_embedded=q_embedded,
                q_continuous=q_continuous,
                p0_discrete=p_discrete,
                key=key,
                epsilon=epsilon,
            )
            q_continuous, p_continuous = leapfrog_halfstep2(
                q_embedded=q_embedded,
                q_continuous=q_continuous,
                p_continuous=p_continuous,
                epsilon=epsilon,
            )
            return q_embedded, q_continuous, p_discrete, p_continuous, key

        q_embedded, q_continuous, p_discrete, p_continuous, key = jax.lax.fori_loop(
            0, L, body_fun, (q_embedded, q_continuous, p_discrete, p_continuous, key)
        )
        # Accept or reject
        current_U = potential(q0_embedded, q0_continuous)
        current_K = np.sum(np.abs(p0_discrete)) + 0.5 * np.sum(p0_continuous ** 2)
        proposed_U = potential(q_embedded, q_continuous)
        proposed_K = np.sum(np.abs(p_discrete)) + 0.5 * np.sum(p_continuous ** 2)
        key, subkey = jax.random.split(key)
        accept = jax.random.uniform(subkey) < np.exp(
            current_U - proposed_U + current_K - proposed_K
        )
        q_embedded, q_continuous = jax.lax.cond(
            accept,
            (q_embedded, q_continuous),
            lambda x: x,
            (q0_embedded, q0_continuous),
            lambda x: x,
        )
        return q_embedded, q_continuous, accept, key

    def leapfrog_halfstep1(q_embedded, q_continuous, p_continuous, epsilon):
        p_continuous -= (
            0.5
            * epsilon
            * adaptive_step_size
            * grad_potential(q_embedded, q_continuous)
        )
        q_continuous += 0.5 * epsilon * adaptive_step_size * p_continuous
        return q_continuous, p_continuous

    def leapfrog_halfstep2(q_embedded, q_continuous, p_continuous, epsilon):
        q_continuous += 0.5 * epsilon * adaptive_step_size * p_continuous
        p_continuous -= (
            0.5
            * epsilon
            * adaptive_step_size
            * grad_potential(q_embedded, q_continuous)
        )
        return q_continuous, p_continuous

    def coordinatewise(q0_embedded, q_continuous, p0_discrete, key, epsilon):
        q_embedded = np.array(q0_embedded)
        p_discrete = np.array(p0_discrete)
        key, subkey = jax.random.split(key)
        coord_order = jax.random.shuffle(subkey, np.arange(q0_embedded.shape[0]))

        def body_fun(ii, val):
            q_embedded, p_discrete = val
            ind = coord_order[ii]
            q_embedded_proposal = jax.ops.index_update(
                q_embedded,
                jax.ops.index[ind],
                q_embedded[ind] + epsilon * np.sign(p_discrete[ind]),
            )
            delta_E = potential(q_embedded_proposal, q_continuous) - potential(
                q_embedded, q_continuous
            )
            # Decide whether to accept or reject
            accept = np.abs(p_discrete[ind]) > delta_E
            p_discrete_accept = jax.ops.index_update(
                p_discrete,
                jax.ops.index[ind],
                np.sign(p_discrete[ind]) * (np.abs(p_discrete[ind]) - delta_E),
            )
            p_discrete_reject = jax.ops.index_update(
                p_discrete, jax.ops.index[ind], -p_discrete[ind]
            )
            q_embedded, p_discrete = jax.lax.cond(
                accept,
                (q_embedded_proposal, p_discrete_accept),
                lambda x: x,
                (q_embedded, p_discrete_reject),
                lambda x: x,
            )
            return q_embedded, p_discrete

        q_embedded, p_discrete = jax.lax.fori_loop(
            0, coord_order.shape[0], body_fun, (q_embedded, p_discrete)
        )
        return q_embedded, p_discrete, key

    q_embedded = q0_embedded
    q_continuous = q0_continuous

    def body_fun(val):
        q_embedded, q_continuous, _, key = val
        q_embedded, q_continuous, accept, key = take_dhmc_step(
            q0_embedded=q_embedded,
            q0_continuous=q_continuous,
            key=key,
            epsilon_range=epsilon_range,
            L_range=L_range,
        )
        return q_embedded, q_continuous, accept, key

    output = fori_collect(
        0, n_samples, body_fun, (q_embedded, q_continuous, True, key), progbar=progbar
    )
    q_embedded_array, q_continuous_array, accept_array, _ = output
    return q_embedded_array, q_continuous_array, accept_array
