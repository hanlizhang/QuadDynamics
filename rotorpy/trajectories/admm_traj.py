import time
import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from trajax import optimizers
import matplotlib.pyplot as plt


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


def dynamics(x, u, t):
    return x + dt * quad(x, u, t)


# global parameters for trajectory optimization
# horizon = 500
dt = 0.01
eq_point = jnp.array([3, 2, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# quadrotor physical constants
g = 9.81
m_q = 1.0
Ix = 8.1 * 1e-3
Iy = 8.1 * 1e-3
Iz = 14.2 * 1e-3

K_d = 0.25  # Adjust this drag coefficient based on your needs


class ADMMTraj(object):
    def __init__(
        self,
        x0,
        horizon=500,
        dt=0.01,
        eq_point=jnp.array([3, 2, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ):
        # global parameters for trajectory optimization
        self.horizon = horizon
        self.dt = dt
        self.eq_point = eq_point

        # quadrotor physical constants
        self.g = 9.81
        self.m_q = 1.0
        self.Ix = 8.1 * 1e-3
        self.Iy = 8.1 * 1e-3
        self.Iz = 14.2 * 1e-3

        self.K_d = 0.25  # Adjust this drag coefficient based on your needs
        self.x0 = x0

        self.x_k, self.u_k, self.param = self.admm()

    def admm(self):
        """
        Runs the ADMM iterations for a given initial condition and plots the 3D trajectories
        :param x0: initial state
        :return: number of iterations to converge, residual error
        """
        rho = 25
        T = self.horizon + 1
        n = 12
        m = 4

        # Let's setup each suproblem as parameteric problems so we can call them in a loop
        # r-subproblem
        t_start_r = time.time()
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
        final_err = r[:, -1] - self.eq_point

        stage_cost = 0.1 * cp.sum_squares(stage_err)
        final_cost = 1000 * cp.sum_squares(final_err)
        utility_cost = stage_cost + final_cost
        admm_cost = rhok * cp.sum_squares(r - xk + vk)

        r_suprob = cp.Problem(cp.Minimize(utility_cost + admm_cost))

        t_end_r = time.time()
        t_elapsed_r = t_end_r - t_start_r
        print("time elapsed for r_sub: ", t_elapsed_r)

        @jax.jit
        def solve_xu_subproblem(rk, vk, uk, rho):
            # @jax.jit
            # def quad(x, u, t):
            #     """
            #     Dynamics function for defining nonlinear quadrotor dynamics as xdot = f(x, u)
            #     :param x:
            #     :param u:
            #     :param t:
            #     :return:
            #     """
            #     del t

            #     # Decompose state and control vectors for clarity
            #     pos = x[0:3]
            #     vel = x[3:6]  # world frame velocity
            #     angles = x[6:9]  # phi, theta, psi
            #     angular_rates = x[9:12]  # roll rate, pitch rate, yaw rate

            #     # Rotor forces and torques
            #     Fz = u[0]
            #     tau = u[1:4]  # tau_x, tau_y, tau_z

            #     # Calculate rotation matrix (assuming a simple ZYX Euler angle convention)
            #     phi, theta, psi = angles
            #     R = jnp.array(
            #         [
            #             [
            #                 jnp.cos(psi) * jnp.cos(theta),
            #                 jnp.cos(psi) * jnp.sin(theta) * jnp.sin(phi)
            #                 - jnp.sin(psi) * jnp.cos(phi),
            #                 jnp.cos(psi) * jnp.sin(theta) * jnp.cos(phi)
            #                 + jnp.sin(psi) * jnp.sin(phi),
            #             ],
            #             [
            #                 jnp.sin(psi) * jnp.cos(theta),
            #                 jnp.sin(psi) * jnp.sin(theta) * jnp.sin(phi)
            #                 + jnp.cos(psi) * jnp.cos(phi),
            #                 jnp.sin(psi) * jnp.sin(theta) * jnp.cos(phi)
            #                 - jnp.cos(psi) * jnp.sin(phi),
            #             ],
            #             [
            #                 -jnp.sin(theta),
            #                 jnp.cos(theta) * jnp.sin(phi),
            #                 jnp.cos(theta) * jnp.cos(phi),
            #             ],
            #         ]
            #     )

            #     # Dynamics
            #     pos_dot = vel
            #     thrust_force = jnp.array([0, 0, Fz])
            #     drag_force = K_d * vel
            #     gravitational_force = jnp.array([0, 0, -m_q * g])
            #     force_body_frame = thrust_force - drag_force + gravitational_force
            #     vel_dot = jnp.dot(R, force_body_frame) / m_q

            #     # Assuming simple rigid body dynamics for angular rate derivatives
            #     angular_rates_dot = jnp.array(
            #         [
            #             ((Iy - Iz) / Ix) * angular_rates[1] * angular_rates[2]
            #             + tau[0] / Ix,
            #             ((Iz - Ix) / Iy) * angular_rates[0] * angular_rates[2]
            #             + tau[1] / Iy,
            #             ((Ix - Iy) / Iz) * angular_rates[0] * angular_rates[1]
            #             + tau[2] / Iz,
            #         ]
            #     )

            #     return jnp.concatenate(
            #         [pos_dot, vel_dot, angular_rates, angular_rates_dot]
            #     )

            # def dynamics(x, u, t):
            #     return x + dt * quad(x, u, t)

            t_start = time.time()

            def cost(x, u, t):
                err = rk[:, t] - x + vk[:, t]
                # err = state_wrap(err)
                # trying without state_wrap to be consistent with planning problem
                stage_cost = rho / 2 * jnp.dot(err, err) + 0.01 * jnp.dot(u, u)
                final_cost = rho / 2 * jnp.dot(err, err)
                return jnp.where(t == self.horizon, final_cost, stage_cost)

            X, U, _, _, _, _, _ = optimizers.ilqr(
                cost, dynamics, self.x0, uk.T, maxiter=100
            )
            t_end = time.time()
            t_elapsed = t_end - t_start
            print("time elapsed for xu_sub: ", t_elapsed)
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
            if k > 1000:
                print("ADMM did not converge")
                break
            else:
                k += 1
                print("Iteration k: ", k)
                t_start = time.time()
                # print("time:", time.time())
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
                t_end = time.time()
                t_elapsed = t_end - t_start
                print("time elapsed for iteration: ", t_elapsed)
        param = lambda: None  # Lazy way to define an empty class in python
        param.nbData = T
        param.l = 0.25  # Length of the car
        param.Mu = np.array(
            [self.eq_point[0], self.eq_point[1], self.eq_point[2]]
        )  # Viapoint (x1,x2,theta,phi)

        # xk = np.array(xk.value)
        # uk = np.array(uk.value)

        return xk.value, uk.value, param

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.
        Inputs
            t, time, s
        Outputs
            xk,        position, m
            uk[0],     thrust, N
            uk[1:4],   torque, Nm
        """
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0
        yaw_ddot = 0

        # xk, uk, _ = self.admm()
        # time allocation
        # total_time = self.horizon * self.dt
        # round to nearest time step
        t_index = int(np.round(t / self.dt))
        # print(t_index)
        if t_index > self.horizon - 1:
            u_t = self.u_k[:, -1]
            x_t = self.x_k[:, -1]
        else:
            u_t = self.u_k[:, t_index]
            x_t = self.x_k[:, t_index]
        # u_t = self.u_k[:, t_index]
        print("t_index:", t_index)
        flat_output = {
            "x": x_t,
            "x_dot": x_dot,
            "x_ddot": x_ddot,
            "x_dddot": x_dddot,
            "x_ddddot": x_ddddot,
            "yaw": yaw,
            "yaw_dot": yaw_dot,
            "yaw_ddot": yaw_ddot,
            "u": u_t,
        }

        return flat_output


if __name__ == "__main__":
    # initial condition
    x0 = jnp.array([0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    x0 = jnp.float64(x0)
    traj = ADMMTraj(x0)
    xk, uk, param = traj.admm()
    print(xk)
    print(uk)
    # Plotting
    # ===============================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # plt.axis("off")
    # plt.gca().set_aspect('equal', adjustable='box')

    ax.plot3D(xk[0, :], xk[1, :], xk[2, :], c="black")
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
    plt.ylim(-0.1, 3)
    plt.xlim(-0.1, 3.5)
    plt.legend()

    plt.show()
