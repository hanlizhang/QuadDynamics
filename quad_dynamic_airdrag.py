"""
Author: Anusha Srikanthan
Synopsis: Python code to run ADMM iterates on quadrotor dynamics
"""

import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
import jax.scipy as jsp

# import warnings
from trajax import optimizers
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D


# global parameters for trajectory optimization
horizon = 20
dt = 0.05
eq_point = jnp.array([3, 2, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# quadrotor physical constants
g = 9.81
m_q = 1.0
Ix = 8.1 * 1e-3
Iy = 8.1 * 1e-3
Iz = 14.2 * 1e-3

K_d = 0.25  # Adjust this drag coefficient based on your needs


@jax.jit
def quad(x, u, t):
    """
    Dynamics function for defining nonlinear quadrotor dynamics as xdot = f(x, u)
    :param x:
    :param u:
    :param t:
    :return:
    """
    del t

    # Decompose state and control vectors for clarity
    pos = x[0:3]
    vel = x[3:6]  # world frame velocity
    angles = x[6:9]  # phi, theta, psi
    angular_rates = x[9:12]  # roll rate, pitch rate, yaw rate

    # Rotor forces and torques
    Fz = u[0]
    tau = u[1:4]  # tau_x, tau_y, tau_z

    # Calculate rotation matrix (assuming a simple ZYX Euler angle convention)
    phi, theta, psi = angles
    R = jnp.array(
        [
            [
                jnp.cos(psi) * jnp.cos(theta),
                jnp.cos(psi) * jnp.sin(theta) * jnp.sin(phi)
                - jnp.sin(psi) * jnp.cos(phi),
                jnp.cos(psi) * jnp.sin(theta) * jnp.cos(phi)
                + jnp.sin(psi) * jnp.sin(phi),
            ],
            [
                jnp.sin(psi) * jnp.cos(theta),
                jnp.sin(psi) * jnp.sin(theta) * jnp.sin(phi)
                + jnp.cos(psi) * jnp.cos(phi),
                jnp.sin(psi) * jnp.sin(theta) * jnp.cos(phi)
                - jnp.cos(psi) * jnp.sin(phi),
            ],
            [
                -jnp.sin(theta),
                jnp.cos(theta) * jnp.sin(phi),
                jnp.cos(theta) * jnp.cos(phi),
            ],
        ]
    )

    # Dynamics
    pos_dot = vel
    thrust_force = jnp.array([0, 0, Fz])
    drag_force = K_d * vel
    gravitational_force = jnp.array([0, 0, -m_q * g])
    force_body_frame = thrust_force - drag_force + gravitational_force
    vel_dot = jnp.dot(R, force_body_frame) / m_q

    # Assuming simple rigid body dynamics for angular rate derivatives
    angular_rates_dot = jnp.array(
        [
            ((Iy - Iz) / Ix) * angular_rates[1] * angular_rates[2] + tau[0] / Ix,
            ((Iz - Ix) / Iy) * angular_rates[0] * angular_rates[2] + tau[1] / Iy,
            ((Ix - Iy) / Iz) * angular_rates[0] * angular_rates[1] + tau[2] / Iz,
        ]
    )

    return jnp.concatenate([pos_dot, vel_dot, angular_rates, angular_rates_dot])


# Replace with plot drone trajectory sim visualizer
def plot_car(x, param, col="black", col_alpha=1):
    """
    Visualizer for the car trajectories in 2D with animation
    :param x:
    :param param:
    :param col:
    :param col_alpha:
    :return:
    """
    w = param.l / 2
    x_rl = x[:2] + 0.5 * w * np.array([-np.sin(x[2]), np.cos(x[2])])
    x_rr = x[:2] - 0.5 * w * np.array([-np.sin(x[2]), np.cos(x[2])])
    x_fl = x_rl + param.l * np.array([np.cos(x[2]), np.sin(x[2])])
    x_fr = x_rr + param.l * np.array([np.cos(x[2]), np.sin(x[2])])
    x_plot = np.concatenate((x_rl, x_rr, x_fr, x_fl, x_rl))
    plt.plot(x_plot[0::2], x_plot[1::2], linewidth=2, c=col, alpha=col_alpha)
    plt.scatter(x[0], x[1], marker=".", s=200, c=col, alpha=col_alpha)


def angle_wrap(theta):
    """
    Function to wrap angles greater than pi
    :param theta: heading angle of system
    :return: wrapped angle
    """
    return (theta) % (2 * np.pi)


def dynamics(x, u, t):
    return x + dt * quad(x, u, t)


def rollout(x0, U, dynamics, T):
    n = len(x0)
    x = np.zeros((n, T))
    x[:, 0] = x0
    for t in range(T - 1):
        x[:, t + 1] = dynamics(x[:, t], U[:, t], t)
    return x


def admm(x0):
    """
    Runs the ADMM iterations for a given initial condition and plots the 3D trajectories
    :param x0: initial state
    :return: number of iterations to converge, residual error
    """
    rho = 25
    T = horizon + 1
    n = 12
    m = 4

    # Let's setup each suproblem as parameteric problems so we can call them in a loop
    # r-subproblem
    r = cp.Variable((n, T))
    xk = cp.Parameter((n, T))
    uk = cp.Parameter((m, T - 1))
    vk = cp.Parameter((n, T))
    rhok = cp.Parameter(nonneg=True)

    rhok.value = rho
    xk.value = np.zeros((n, T))
    uk.value = np.zeros((m, T - 1))
    vk.value = np.zeros((n, T))

    # this is probably a source of error because I can't implement angle
    # wrapping using cvxpy
    # stage_err = cp.hstack([(r[:, t] - eq_point) for t in range(T-1)])
    stage_err = cp.hstack([(r[:, t]) for t in range(T - 1)])
    final_err = r[:, -1] - eq_point

    stage_cost = 0.1 * cp.sum_squares(stage_err)
    final_cost = 1000 * cp.sum_squares(final_err)
    utility_cost = stage_cost + final_cost
    admm_cost = rhok * cp.sum_squares(r - xk + vk)

    r_suprob = cp.Problem(cp.Minimize(utility_cost + admm_cost))

    @jax.jit
    def solve_xu_subproblem(rk, vk, uk, rho):
        def cost(x, u, t):
            err = rk[:, t] - x + vk[:, t]
            # err = state_wrap(err)
            # trying without state_wrap to be consistent with planning problem
            stage_cost = rho / 2 * jnp.dot(err, err) + 0.01 * jnp.dot(u, u)
            final_cost = rho / 2 * jnp.dot(err, err)
            return jnp.where(t == horizon, final_cost, stage_cost)

        X, U, _, _, _, _, _ = optimizers.ilqr(cost, dynamics, x0, uk.T, maxiter=10)
        return X, U

    # run ADMM algorithms
    # K = 50
    rk = cp.Parameter((n, T))
    rk.value = np.zeros((n, T))
    xk.value = np.zeros((n, T))
    uk.value = np.zeros((m, T - 1))
    vk.value = np.zeros((n, T))
    rhok.value = 25

    # for k in np.arange(K):
    k = 0
    err = 100
    residual = []
    while err >= 1e-2:
        if k > 2000:
            print("ADMM did not converge")
            break
        else:
            k += 1
            print("Iteration k: ", k)
            # update r
            r_suprob.solve()

            # update x u
            rk.value = r.value
            # print(rk.value)

            x, u = solve_xu_subproblem(
                jnp.array(rk.value),
                jnp.array(vk.value),
                jnp.array(uk.value),
                jnp.array(rhok.value),
            )

            # compute residuals
            sxk = rhok.value * (xk.value - x.T).flatten()
            suk = rhok.value * (uk.value - u.T).flatten()
            dual_res_norm = np.linalg.norm(np.hstack([sxk, suk]))
            pr_res_norm = np.linalg.norm(rk.value - xk.value)

            # update rhok and rescale vk
            if pr_res_norm > 10 * dual_res_norm:
                rhok.value = 2 * rhok.value
                vk.value = vk.value / 2
            elif dual_res_norm > 10 * pr_res_norm:
                rhok.value = rhok.value / 2
                vk.value = vk.value * 2

            # update v
            xk.value = np.array(x.T)
            uk.value = np.array(u.T)
            vk.value = vk.value + rk.value - xk.value

            err = np.trace((rk.value - xk.value).T @ (rk.value - xk.value))
            residual.append(err)

    param = lambda: None  # Lazy way to define an empty class in python
    param.nbData = T
    param.l = 0.25  # Length of the car
    param.Mu = np.array(
        [eq_point[0], eq_point[1], eq_point[2]]
    )  # Viapoint (x1,x2,theta,phi)

    # Plotting
    # ===============================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # plt.axis("off")
    # plt.gca().set_aspect('equal', adjustable='box')

    ax.plot3D(xk.value[0, :], xk.value[1, :], xk.value[2, :], c="black")
    ax.scatter(x0[0], x0[1], x0[2], color="b", marker=".", s=200, label="Initial pose")
    ax.scatter(
        param.Mu[0],
        param.Mu[1],
        param.Mu[2],
        color="r",
        marker=".",
        s=200,
        label="Desired pose",
    )

    # nb_plots = 15
    # for i in range(nb_plots):
    #     plot_car(xk.value[:, int(i * param.nbData / nb_plots)], param, 'black', 0.1 + 0.9 * i / nb_plots)
    # plot_car(xk.value[:, -1], param, 'black')

    # plot_car(param.Mu, param, 'red')

    # plt.scatter(param.Mu[0], param.Mu[1], color='r', marker='.', s=200, label="Desired pose")
    # plt.plot(np.array([1, 1]), np.array([1, 2]), color='b', linewidth=4)
    # plt.plot(np.array([2, 1]), np.array([1, 2]), color='b', linewidth=4)
    # plt.plot(np.array([2, 1]), np.array([1, 1]), color='b', linewidth=4)
    # plt.plot(np.array([1, 1]), np.array([0, 1.5]), color='b', linewidth=4)
    # plt.plot(np.array([1, 4]), np.array([1.5, 1.5]), color='b', linewidth=4)
    # plt.plot(np.array([0, 0]), np.array([0, 2.5]), color='b', linewidth=4)
    # plt.plot(np.array([0, 4]), np.array([2.5, 2.5]), color='b', linewidth=4)
    plt.ylim(-0.1, 3)
    plt.xlim(-0.1, 3.5)
    plt.legend()

    plt.show()

    return k * (1 + 10), err, residual


def save_object(obj, filename):
    """
    Save object as a pickle file
    :param obj: Data to be saved
    :param filename: file path
    :return: None
    """
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def main():
    # x0 = jnp.array([0.0, 0.2, 0])

    num_iter = 20
    n = 12
    admm_iter = []
    admm_res = []

    np.random.seed(0)
    rng = np.random.default_rng()

    for i in range(num_iter):
        x0 = rng.standard_normal(n)
        x0[0] = rng.uniform(0, 1)
        # ilqr_iter.append(ilqr(x0))
        a, b, c = admm(x0)
        admm_iter.append([a, b])
        admm_res.append(c)
        print(admm_iter[i])

    admm_iter = np.vstack(admm_iter)

    print(admm_iter)
    print("Mean admm", np.mean(admm_iter[:, 0]))
    print("Std admm", np.std(admm_iter[:, 0]))
    print("ADMM convergence rate", sum(np.where(admm_iter[:, 1] <= 1, 1, 0)) / num_iter)

    save_object(admm_iter, "./uni_admm.pkl")
    save_object(admm_res, "./uni_admm_res.pkl")


if __name__ == "__main__":
    main()
