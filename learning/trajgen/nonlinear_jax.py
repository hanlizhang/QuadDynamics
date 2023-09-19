import jax.numpy as jnp
import numpy as np
import jax.scipy.linalg as spl
from .trajutils import _cost_matrix, _diff_coeff
from jaxopt import ProjectedGradient, GradientDescent
from jaxopt.projection import projection_affine_set
from learning.trajgen.nonlinear import _coeff_constr_A, _coeff_constr_b
from jax import jit


def coeff_traj(order, numsteps, ts, coeffs, p):
    ref = jnp.zeros((p, numsteps))
    times = jnp.linspace(ts[0], ts[-1], numsteps)
    k = 0
    for j in range(p):
        for i, tt in enumerate(times):
            k = jnp.where(tt > ts[k + 1], k + 1, k)
            ref.at[j, i].set(
                jnp.polyval(
                    np.flip(coeffs[j * k * (order + 1) : j * (k + 1) * (order + 1)]),
                    tt - ts[k],
                )
            )
    return ref


def modify_reference(
    wp,
    ts,
    numsteps,
    order,
    p,
    regularizer,
    cost_mat_full,
    A_coeff_full,
    b_coeff_full,
    coeff0,
):
    """
    Running projected gradient descent on the neural network cost + min snap cost with constraints
    """
    num_seg = len(ts) - 1
    # n = coeff0.shape[2]
    # num_coeffs = np.prod(coeff0.shape[1:])
    # durations = ts[1:] - ts[:-1]
    # import ipdb

    # ipdb.set_trace()
    # cost_mat = spl.block_diag(*[_cost_matrix(order, 4, d) for d in durations])
    # print("cost_mat!!!!!!!!!!!!!!!!!", cost_mat)
    # A_coeff = _coeff_constr_A(ts, n, num_coeffs)
    # b_coeff = _coeff_constr_b(wp.T, ts, n)

    # cost_mat_full = spl.block_diag(*[cost_mat for i in range(p)])
    # A_coeff_full = spl.block_diag(*[A_coeff for i in range(p)])
    # b_coeff_full = jnp.ravel(b_coeff)
    # times = np.concatenate(
    #     [
    #         np.linspace(0, ts[i + 1] - ts[i], (numsteps // num_seg) + 1)
    #         for i in range(num_seg)
    #     ]
    # )

    N = 502
    # time = np.linspace(0, 1.1 * ts[-1], N)
    # print("Times", time.shape)
    time = np.linspace(ts[0], ts[-1], N)
    times = np.zeros(time.shape)
    k = 0
    for i, tt in enumerate(time):
        if tt > ts[k + 1]:
            k += 1
        times[i] = tt - ts[k]

    # print("Times", times.shape)
    # print("Times", times)
    # delta_t = np.diff(ts)
    # # time = np.clip(time, ts[0], ts[-1])
    # for i in range(ts.size - 1):
    #     # delta_t = ts[i + 1] - ts[i]

    #     if ts[i] + delta_t[i] >= time:
    #         break
    # time = time - ts[i]

    ## @jit
    def nn_cost(coeffs):
        """
        Function to compute trajectories given polynomial coefficients
        :param coeffs: 4-dim polynomial coefficients (x, y, z, yaw)
        :param ts: waypoint time allocation
        :param numsteps: Total number of samples in the reference
        :return: ref
        """
        ref = jnp.zeros((p, numsteps))

        # print("Coeffs shape", coeffs.shape)
        # coeffs' shape is (p, num_seg, order + 1)
        # let coeff for each segment be c0, c1, c2, c3, c4, c5, c6, c7(flip the order of the coefficients segment-wise)
        for j in range(p):
            ref.at[j, :].set(
                jnp.polyval(
                    # coeffs[j, :, :].ravel(),
                    coeffs[j * num_seg * (order + 1) : (j + 1) * num_seg * (order + 1)],
                    times,
                )
            )

        # #####anusha
        # for j in range(p):
        #     # ref.at[:, i].set(jnp.dot(_diff_coeff(coeffs.shape[2] - 1, tt - ts[k], 0), coeffs[:, k, :].T))
        #     ref = ref.at[j, :].set(
        #         jnp.polyval(
        #             coeffs[j * num_seg * (order + 1) : (j + 1) * num_seg * (order + 1)],
        #             times,
        #         )
        #     )
        # #####anusha

        # for j in range(p):
        #    ref = ref.at[j, :].set(jnp.polyval(coeffs[j * num_seg * (order + 1):(j + 1) * num_seg * (order + 1)], times))
        # return coeffs.T @ cost_mat_full @ coeffs + regularizer(jnp.append(wp[0, :], jnp.vstack(ref)))[0]

        # for j in range(p):
        #     for m in range(num_seg):
        #         ref.at[
        #             j, m * (numsteps // num_seg) : (m + 1) * (numsteps // num_seg)
        #         ].set(
        #             jnp.polyval(
        #                 np.flip(
        #                     coeffs[j * m * (order + 1) : j * (m + 1) * (order + 1)]
        #                 ),
        #                 np.linspace(ts[m], ts[m + 1], numsteps // num_seg),
        #             )
        #         )
        # ref.at[j, :].set(
        #     jnp.polyval(
        #         coeffs[
        #             j * num_seg * (order + 1) : (j + 1) * num_seg * (order + 1)
        #         ],
        #         times,
        #     )
        # )

        # ref.append(jnp.polyval(coeffs[j, :], tt - ts[k]))
        # ref.append(poly[k](tt - ts[k]))
        # print("Cost matrix", cost_mat_full)
        # print("Coeff  cost", coeffs.T @ cost_mat_full @ coeffs)
        # print("Network cost", jnp.exp(regularizer(jnp.append(wp[0, :], jnp.vstack(ref)))[0]))
        # print("Coeff cost", (coeffs.T @ cost_mat_full @ coeffs))

        # return coeffs.T @ cost_mat_full @ coeffs + jnp.exp(
        #     regularizer(jnp.append(wp[0, :], jnp.vstack(ref)))[0]
        # )

        # ONLY MIN SNAP COST
        # return coeffs.T @ cost_mat_full @ coeffs

        # ONLY NN COST
        return jnp.exp(regularizer(jnp.append(wp[0, :], jnp.vstack(ref)))[0])

        # print(
        #     "network cost",
        #     jnp.exp(regularizer(jnp.append(wp[0, :], jnp.vstack(ref)))[0]),
        # )
        # return coeffs.T @ cost_mat_full @ coeffs + 100000 * jnp.exp(
        #     regularizer(jnp.append(wp[0, :], jnp.vstack(ref)))[0]
        # )

        # return coeffs.T @ cost_mat_full @ coeffs
        # return jnp.exp(
        #     regularizer(
        #         jnp.append(
        #             wp[0, :], jnp.vstack(coeff_traj(order, numsteps, ts, coeffs, p))
        #         )
        #     )
        # )[0]
        # return 100000 * jnp.exp(regularizer(jnp.append(wp[0, :], jnp.vstack(ref)))[0])

    # + jnp.exp(
    #        regularizer(jnp.append(wp[0, :], jnp.vstack(ref)))[0]  # min jerk cost
    #    )  # network cost
    # return coeffs.T @ cost_mat_full @ coeffs
    # return jnp.exp(  # min jerk cost
    #     regularizer(jnp.append(wp[0, :], jnp.vstack(ref)))[0]
    # )  # network cost

    pg = ProjectedGradient(
        nn_cost,
        projection=projection_affine_set,
        # jit=False,
        maxiter=50,
        # verbose=True,
    )
    # pg = GradientDescent(nn_cost, jit=False, maxiter=50)
    # print("coeff0's shape", coeff0.shape)
    # print("A_coeff_full's shape", A_coeff_full.shape)
    # print("b_coeff_full's shape", b_coeff_full.shape)
    """
    # flip the order of the coefficients for coeff0
    print("coeff0 first 10", coeff0[0, 0, :])
    print("coeff0 second segment first 10", coeff0[0, 1, :])
    # coeff0_copy = coeff0.copy()
    for i in range(p):
        # coeff0_copy[i, :, :] = coeff0[i, :, ::-1]
        coeff0[i, :, :] = coeff0[i, :, ::-1]
    print("coeff0 first 10 after flipping: ", coeff0[0, 0, :])
    print("coeff0 second segment first 10 after flipping: ", coeff0[0, 1, :])

    print("coeff0.ravel's first 8", coeff0.ravel()[:8])
    print("coeff0's shape", coeff0.shape)
    print("A_coeff_full's shape", A_coeff_full.shape)
    print("b_coeff_full's shape", b_coeff_full.shape)
    sol = pg.run(coeff0.ravel(), hyperparams_proj=(A_coeff_full, b_coeff_full))
    """
    # sol = pg.run(coeff0)
    # pg = ProjectedGradient(nn_cost, projection=projection_affine_set)
    print("A_coeff_full", A_coeff_full)
    print("b_coeff_full", b_coeff_full)
    print("cost_mat_full", cost_mat_full)
    sol = pg.run(coeff0, hyperparams_proj=(A_coeff_full, b_coeff_full))
    coeff = sol.params
    pred = sol.state.error

    print("Norm difference", np.linalg.norm(coeff0.ravel() - coeff))

    return coeff, pred
    # return np.reshape(coeff, (p, num_seg, order + 1)), pred


def vel(ref):
    """
    Computes the velocity cost by takind first order of the reference trajectory of the form (x, y, z, yaw)
    :param ref: reference trajectory
    :return: velocity cost
    """
    vel_cost = 0
    x_ref = ref[0::4]
    y_ref = ref[1::4]
    z_ref = ref[2::4]
    yaw_ref = ref[3::4]
    # take the first order of the reference trajectory
    x_ref_dot = jnp.diff(x_ref)
    y_ref_dot = jnp.diff(y_ref)
    z_ref_dot = jnp.diff(z_ref)
    yaw_ref_dot = jnp.diff(yaw_ref)
    # compute the velocity cost
    vel_cost += jnp.linalg.norm(x_ref_dot) ** 2
    vel_cost += jnp.linalg.norm(y_ref_dot) ** 2
    vel_cost += jnp.linalg.norm(z_ref_dot) ** 2
    vel_cost += jnp.linalg.norm(yaw_ref_dot) ** 2
    return vel_cost


def modify_ref(aug_state, trained_model_state):
    def calc_cost_GD(ref):
        pred = trained_model_state.apply_fn(trained_model_state.params, ref).ravel()
        # aug_state= np.append(ref[0, :], np.vstack(ref))
        print("pred", jnp.exp(pred[0]))
        return (
            jnp.exp(
                trained_model_state.apply_fn(trained_model_state.params, ref).ravel()[0]
            )
            + jnp.linalg.norm(aug_state - ref) ** 2
            + vel(ref)
        )

    A = np.zeros((6, aug_state.shape[0]))
    A[0, 4] = 1
    A[1, 5] = 1
    A[2, 6] = 1
    # A[3, 7] = 1
    A[-4, -4] = 1
    A[-3, -3] = 1
    A[-2, -2] = 1
    # A[-1, -1] = 1

    b = np.append(aug_state[0:3], aug_state[-4:-1])

    # gd = GradientDescent(calc_cost_GD, verbose=True, maxiter=1)
    pd = ProjectedGradient(calc_cost_GD, projection=projection_affine_set)
    sol = pd.run(aug_state, hyperparams_proj=(A, b))
    # sol = gd.run(aug_state)
    ref = sol.params

    return ref

    # yaw = ref[4::4]
