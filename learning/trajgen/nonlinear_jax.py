import jax.numpy as jnp
import numpy as np
import jax.scipy.linalg as spl
from .trajutils import _cost_matrix
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_affine_set
from learning.trajgen.nonlinear import _coeff_constr_A, _coeff_constr_b
from jax import jit


def modify_reference(wp, ts, numsteps, order, p, regularizer, coeff0):
    """
    Running projected gradient descent on the neural network cost + min snap cost with constraints
    """
    num_seg = len(ts) - 1
    n = coeff0.shape[2]
    num_coeffs = np.prod(coeff0.shape[1:])
    durations = ts[1:] - ts[:-1]
    cost_mat = spl.block_diag(*[_cost_matrix(order, num_seg, d) for d in durations])
    A_coeff = _coeff_constr_A(ts, n, num_coeffs)
    b_coeff = _coeff_constr_b(wp.T, ts, n)

    cost_mat_full = spl.block_diag(*[cost_mat for i in range(p)])
    A_coeff_full = spl.block_diag(*[A_coeff for i in range(p)])
    b_coeff_full = jnp.ravel(b_coeff)

    # @jit
    def nn_cost(coeffs):
        """
        Function to compute trajectories given polynomial coefficients
        :param coeffs: 4-dim polynomial coefficients (x, y, z, yaw)
        :param ts: waypoint time allocation
        :param numsteps: Total number of samples in the reference
        :return: ref
        """
        ref = jnp.zeros((p, numsteps))
        times = jnp.linspace(ts[0], ts[-1], numsteps)
        for j in range(p):
            ref.at[j, :].set(
                jnp.polyval(
                    coeffs[j * num_seg * (order + 1) : (j + 1) * num_seg * (order + 1)],
                    times,
                )
            )
        print(
            "network cost",
            jnp.exp(regularizer(jnp.append(wp[0, :], jnp.vstack(ref)))[0]),
        )
        return coeffs.T @ cost_mat_full @ coeffs + jnp.exp(  # min jerk cost
            regularizer(jnp.append(wp[0, :], jnp.vstack(ref)))[0]
        )  # network cost

    pg = ProjectedGradient(nn_cost, projection=projection_affine_set)
    print("coeff0's shape", coeff0.shape)
    print("A_coeff_full's shape", A_coeff_full.shape)
    print("b_coeff_full's shape", b_coeff_full.shape)
    sol = pg.run(coeff0.ravel(), hyperparams_proj=(A_coeff_full, b_coeff_full))
    coeff = sol.params
    pred = sol.state.error

    return np.reshape(coeff, (p, num_seg, order + 1)), pred
