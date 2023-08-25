#! /usr/bin/env python3

"""
Generate training data
simple replanning with lissajous trajectory with fixed waypoints
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D projection
import rospy
import numpy as np
import random

from trajgen import quadratic, nonlinear_jax, nonlinear, valuefunc


import torch
import pickle
import sys

import ruamel.yaml as yaml
from flax.training import train_state
import optax
import jax
from mlp import MLP, MLP_torch
from model_learning import restore_checkpoint
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_affine_set
import jax.numpy as jnp


PI = np.pi


def generate_lissajous_traj(
    s,
    x_num_periods,
    y_num_periods,
    z_num_periods,
    yaw_num_periods,
    period,
    x_amp,
    y_amp,
    z_amp,
    yaw_amp,
):
    """
    Function to generate Lissajous trajectory
    :return:
    """
    x = lambda a: x_amp * (1 - np.cos(2 * PI * x_num_periods * a / period))
    y = lambda a: y_amp * np.sin(2 * PI * y_num_periods * a / period)
    z = lambda a: z_amp * np.sin(2 * PI * z_num_periods * a / period)
    yaw = lambda a: yaw_amp * np.sin(2 * PI * yaw_num_periods * a / period)
    return np.array([x(s), y(s), z(s), yaw(s)])


def compute_coeff_deriv(coeff, n, segments):
    """
    Function to compute the nth derivative of a polynomial
    :return:
    """
    coeff_new = coeff.copy()
    for i in range(segments):  # piecewise polynomial
        for j in range(n):  # Compute nth derivative of polynomial
            t = np.poly1d(coeff_new[i, :]).deriv()
            coeff_new[i, j] = 0
            coeff_new[i, j + 1 :] = t.coefficients
    return coeff_new


def sampler(poly, T, ts):
    """
    Function to generate samples given polynomials
    :param coeff:
    :return:
    """
    k = 0
    ref = []
    for i, tt in enumerate(np.linspace(ts[0], ts[-1], T)):
        if tt > ts[k + 1]:
            k += 1
        ref.append(poly[k](tt - ts[k]))
    return ref


def compute_pos_vel_acc(Tref, nn_coeffs, segments, ts):
    """
    Function to compute pos, vel, acc from nn coeffs
    :param timesteps:
    :return:
    """
    # Compute full state
    coeff_x = np.vstack(nn_coeffs[0, :, :])
    coeff_y = np.vstack(nn_coeffs[1, :, :])
    coeff_z = np.vstack(nn_coeffs[2, :, :])
    coeff_yaw = np.vstack(nn_coeffs[3, :, :])

    pos = []
    vel = []
    acc = []
    jerk = []

    x_ref = [np.poly1d(coeff_x[i, :]) for i in range(segments)]
    x_ref = np.vstack(sampler(x_ref, Tref, ts)).flatten()

    y_ref = [np.poly1d(coeff_y[i, :]) for i in range(segments)]
    y_ref = np.vstack(sampler(y_ref, Tref, ts)).flatten()

    z_ref = [np.poly1d(coeff_z[i, :]) for i in range(segments)]
    z_ref = np.vstack(sampler(z_ref, Tref, ts)).flatten()
    pos.append([x_ref, y_ref, z_ref])

    dot_x = compute_coeff_deriv(coeff_x, 1, segments)
    xdot_ref = [np.poly1d(dot_x[i, :]) for i in range(segments)]
    xdot_ref = np.vstack(sampler(xdot_ref, Tref, ts)).flatten()

    dot_y = compute_coeff_deriv(coeff_y, 1, segments)
    ydot_ref = [np.poly1d(dot_y[i, :]) for i in range(segments)]
    ydot_ref = np.vstack(sampler(ydot_ref, Tref, ts)).flatten()

    dot_z = compute_coeff_deriv(coeff_z, 1, segments)
    zdot_ref = [np.poly1d(dot_z[i, :]) for i in range(segments)]
    zdot_ref = np.vstack(sampler(zdot_ref, Tref, ts)).flatten()
    vel.append([xdot_ref, ydot_ref, zdot_ref])

    ddot_x = compute_coeff_deriv(coeff_x, 2, segments)
    xddot_ref = [np.poly1d(ddot_x[i, :]) for i in range(segments)]
    xddot_ref = np.vstack(sampler(xddot_ref, Tref, ts)).flatten()

    ddot_y = compute_coeff_deriv(coeff_y, 2, segments)
    yddot_ref = [np.poly1d(ddot_y[i, :]) for i in range(segments)]
    yddot_ref = np.vstack(sampler(yddot_ref, Tref, ts)).flatten()

    ddot_z = compute_coeff_deriv(coeff_z, 2, segments)
    zddot_ref = [np.poly1d(ddot_z[i, :]) for i in range(segments)]
    zddot_ref = np.vstack(sampler(zddot_ref, Tref, ts)).flatten()
    acc.append([xddot_ref, yddot_ref, zddot_ref])

    dddot_x = compute_coeff_deriv(coeff_x, 3, segments)
    xdddot_ref = [np.poly1d(dddot_x[i, :]) for i in range(segments)]
    xdddot_ref = np.vstack(sampler(xdddot_ref, Tref, ts)).flatten()

    dddot_y = compute_coeff_deriv(coeff_y, 3, segments)
    ydddot_ref = [np.poly1d(dddot_y[i, :]) for i in range(segments)]
    ydddot_ref = np.vstack(sampler(ydddot_ref, Tref, ts)).flatten()

    dddot_z = compute_coeff_deriv(coeff_z, 3, segments)
    zdddot_ref = [np.poly1d(dddot_z[i, :]) for i in range(segments)]
    zdddot_ref = np.vstack(sampler(zdddot_ref, Tref, ts)).flatten()
    jerk.append([xdddot_ref, ydddot_ref, zdddot_ref])

    yaw_ref = [np.poly1d(coeff_yaw[i, :]) for i in range(segments)]
    yaw_ref = np.vstack(sampler(yaw_ref, Tref, ts)).flatten()

    dot_yaw = compute_coeff_deriv(coeff_yaw, 1, segments)
    yawdot_ref = [np.poly1d(dot_yaw[i, :]) for i in range(segments)]
    yawdot_ref = np.vstack(sampler(yawdot_ref, Tref, ts)).flatten()

    return (
        np.vstack(pos),
        np.vstack(vel),
        np.vstack(acc),
        np.vstack(jerk),
        yaw_ref,
        yawdot_ref,
    )


def generate_polynomial_trajectory(start, end, T, order):
    """
    Generates a polynomial trajectory from start to end over time T
    start: start state
    end: end state
    T: total time
    order: order of the polynomial
    """
    # Define the time vector
    t = np.linspace(0, 1, T)

    # Solve for the polynomial coefficients
    # coeffs = np.zeros(order + 1)
    coeffs = np.polyfit(t, t * (end - start) + start, order)

    # Evaluate the polynomial at the desired time steps
    # polynomial = np.zeros(T)
    polynomial = np.polyval(coeffs[::-1], t)
    trajectory = polynomial + start

    return coeffs


def load_torch_model(trained_model_state):
    # Load checkpoint
    weights = trained_model_state.params["params"]

    # Store weights of the network
    hidden_wts = [
        [weights["linear_0"]["kernel"], weights["linear_0"]["bias"]],
        [weights["linear_1"]["kernel"], weights["linear_1"]["bias"]],
        [weights["linear_2"]["kernel"], weights["linear_2"]["bias"]],
    ]
    linear2_wts = [weights["linear2"]["kernel"], weights["linear2"]["bias"]]

    def convert_torch(x):
        print(x.shape)
        return torch.from_numpy(np.array(x))

    # Create network
    inp_size = 1204
    num_hidden = [500, 400, 200]
    mlp_t = MLP_torch(inp_size, num_hidden)

    for i in range(3):
        mlp_t.hidden[i].weight.data = convert_torch(hidden_wts[i][0]).T
        mlp_t.hidden[i].bias.data = convert_torch(hidden_wts[i][1])

    mlp_t.linear2.weight.data = convert_torch(linear2_wts[0]).T
    mlp_t.linear2.bias.data = convert_torch(linear2_wts[1])
    return mlp_t


def simple_replan(selected_waypoints, duration, order, p, vf, rho, idx):
    """
    Function to generate a new trajectory using the selected waypoints
    """

    # Generate time samples for the new trajectory
    ts = np.linspace(0, duration, selected_waypoints.shape[0])

    # Generate the new trajectory using the concatenated waypoints
    _, min_jerk_coeffs = quadratic.generate(
        selected_waypoints, ts, order, duration * 100, p, None, 0
    )
    new_traj_coeffs = np.zeros([p, len(selected_waypoints) - 1, order + 1])

    for k in range(p):
        for j in range(len(selected_waypoints) - 1):
            new_traj_coeffs[k, j, :] = generate_polynomial_trajectory(
                selected_waypoints.T[k, j], selected_waypoints.T[k, j + 1], 100, order
            )

    print("new_traj_coeffs' shape: ", new_traj_coeffs.shape)

    # nn_coeffs = load_object(r"/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/nn_coeffs"+str(rho)+".pkl")
    # import ipdb;
    # ipdb.set_trace()
    start = rospy.Time.now()
    # nn_coeff = quadrotor.generate(torch.tensor(selected_waypoints.T), ts, order, duration * 100, p, rho, vf, torch.tensor(new_traj_coeffs),
    #                              num_iter=150, lr=0.0001)

    nn_coeffs = nonlinear.generate(
        selected_waypoints,
        ts,
        order,
        duration * 100,
        p,
        rho,
        vf,
        min_jerk_coeffs,
        num_iter=100,
        lr=0.001,
    )

    # nn_coeff = nonlinear_jax.generate(selected_waypoints, ts, order, duration * 100, p, rho, vf, new_traj_coeffs,
    #                                  num_iter=100, lr=0.001)
    # print("new_traj_coeffs: ", new_traj_coeffs)

    end = rospy.Time.now()

    generation_time = end - start

    print("generation time: ", generation_time.to_sec())

    # return new_traj_coeffs
    return new_traj_coeffs, min_jerk_coeffs, nn_coeffs


def save_object(obj, filename):
    """
    Function to save to a pickle file
    :param obj:
    :param filename:
    :return:
    """
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(str):
    """
    Function to load to a pickle file
    :param str:
    :return:
    """
    with open(str, "rb") as handle:
        return pickle.load(handle)


def test_opt(trained_model_state, aug_test_state, N, num_inf, rho):
    """
    Planner method that includes the learned tracking cost function and the utility cost
    :param trained_model_state: weights of the trained model
    :param aug_test_state: Test trajectories
    :param N: horizon of each trajectory
    :param num_inf: total number of test trajectories
    :param rho: tracking penalty factor
    :return: sim_cost, init_cost
    """

    new_aug_state = []

    def calc_cost_GD(ref):
        pred = trained_model_state.apply_fn(trained_model_state.params, ref).ravel()
        return jnp.exp(pred[0])

    A = np.zeros((6, (N + 1) * 4))
    A[0, 0] = 1
    A[1, 1] = 1
    A[2, 2] = 1
    A[-3, -3] = 1
    A[-2, -2] = 1
    A[-1, -1] = 1

    solution = []
    sim_cost = []
    init_cost = []
    ref = []
    rollout = []
    times = []

    for i in range(num_inf):
        init = aug_test_state[i, 0:3]
        goal = aug_test_state[i, -3:]

        b = np.append(init, goal)

        init_ref = aug_test_state[i, :].copy()

        # wp = init_ref[3*ts[1]:3*(ts[1]+1)]
        # init_val = calc_cost_GD(wp, init_ref)
        init_val = calc_cost_GD(init_ref)

        pg = ProjectedGradient(ref, projection=projection_affine_set, maxiter=1)
        solution.append(pg.run(aug_test_state[i, :], hyperparams_proj=(A, b)))
        prev_val = init_val
        val = solution[i].state.error
        cur_sol = solution[i]
        chosen_val = val

        if rho < 1:
            loop_val = 5
        else:
            loop_val = 20
        for j in range(loop_val):
            next_sol = pg.update(cur_sol.params, cur_sol.state, hyperparams_proj=(A, b))
            val = next_sol.state.error
            if val < prev_val:
                chosen_val = val
                solution[i] = cur_sol
            prev_val = val
            cur_sol = next_sol

        sol = solution[i]
        new_aug_state.append(sol.params)
        if chosen_val > init_val:
            new_aug_state.append(init_ref)

        # x0 = init_ref[0:3]
        # ref.append(new_aug_state[3:].reshape([N, 3]))

    return new_aug_state


def main():
    # Define the lists to keep track of times for the simulations
    times_nn = []
    times_mj = []
    times_poly = []

    # Initialize neural network
    rho = 1

    # with open(
    #     r"/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/params.yaml"
    # ) as f:
    #     yaml_data = yaml.load(f, Loader=yaml.RoundTripLoader)
    with open(r"/home/mrsl_guest/rotorpy/rotorpy/learning/params.yaml") as f:
        yaml_data = yaml.load(f, Loader=yaml.RoundTripLoader)

    num_hidden = yaml_data["num_hidden"]
    batch_size = yaml_data["batch_size"]
    learning_rate = yaml_data["learning_rate"]

    # Load the trained model
    model = MLP(num_hidden=num_hidden, num_outputs=1)
    # Printing the model shows its attributes
    print(model)

    rng = jax.random.PRNGKey(427)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (batch_size, 4))  # Batch size 64, input size p
    # Initialize the model
    params = model.init(init_rng, inp)

    optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)

    model_state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    model_save = yaml_data["save_path"] + str(rho)

    trained_model_state = restore_checkpoint(model_state, model_save)

    # mlp_t = load_torch_model(trained_model_state)

    # print(mlp_t)

    # vf = valuefunc.MLPValueFunc(mlp_t)

    # vf.network = mlp_t

    vf = model.bind(trained_model_state.params)

    # get the waypoints from circular trajectory in csv--0823
    path = "/home/mrsl_guest/rotorpy/rotorpy/rotorpy/data_out/constwind_1_5_noyaw.csv"
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    # 5 trajs in total, each traj has 502 data points
    num_traj = 5
    num_data = 502
    actual_traj = []
    # add yaw to actual_traj as 0
    for i in range(num_traj):
        actual_traj.append(data[i * num_data : (i + 1) * num_data, 1:4])
        actual_traj[i] = np.hstack((actual_traj[i], np.zeros((num_data, 1))))
    print("actual_traj's shape", np.array(actual_traj).shape)

    # extract waypoints from each traj, density is 10, interval is 50, get 10 waypoints from each traj
    density = 10
    interval = num_data // density  # 50
    # print("interval", interval)
    waypoints = []
    for i in range(num_traj):
        # waypoints.append(traj[i][0::interval, :])
        # no need to include last one
        waypoints.append(actual_traj[i][0::interval, :][:-1, :])

    # print("waypoints's shape", np.array(waypoints).shape)
    # print("waypoints", waypoints)

    # visualize the waypoints
    fig = plt.figure()
    axes = fig.add_subplot(111, projection="3d")
    for i in range(num_traj):
        axes.plot3D(waypoints[i][:, 0], waypoints[i][:, 1], waypoints[i][:, 2], "*")
    axes.set_xlim(-6, 6)
    axes.set_zlim(0, 1)
    axes.set_ylim(-6, 6)
    plt.show()

    # get augstate for the first traj
    ref_traj = []
    # add yaw to ref_traj as 0
    for i in range(num_traj):
        ref_traj.append(data[i * num_data : (i + 1) * num_data, 22:24])

    # focus on the first traj
    i = 0
    p = 4
    order = 5
    duration = 5
    ts = np.linspace(0, duration, len(waypoints[i]))
    print("waypoints[i]'s shape", waypoints[i].shape)

    # get min_jerk_coeffs for each traj
    _, min_jerk_coeffs = quadratic.generate(
        waypoints[i], ts, order, duration * 100, p, None, 0
    )
    print("min_jerk_coeffs's shape", np.array(min_jerk_coeffs).shape)
    print("min_jerk_coeffs", min_jerk_coeffs)

    # get nn_coeffs for each traj
    nn_coeffs = nonlinear.generate(
        torch.tensor(waypoints[i]),
        ts,
        order,
        duration * 100,
        p,
        rho,
        vf,
        torch.tensor(min_jerk_coeffs),
        num_iter=100,
        lr=0.001,
    )
    print("nn_coeffs's shape", np.array(nn_coeffs).shape)
    print("nn_coeffs", nn_coeffs)


"""
# for each traj, get min_jerk_coeffs and nn_coeffs
    # get min_jerk_coeffs for each traj
    min_jerk_coeffs = []
    p = 4
    order = 5
    duration = 5
    ts = np.linspace(0, duration, len(waypoints[i]))
    print("waypoints[i]'s shape", waypoints[i].shape)
    # print data type of waypoints[i]
    print("waypoints[i]'s type", type(waypoints[i]))
    for i in range(num_traj):
        print(i)
        _, min_jerk_coeffs = quadratic.generate(
            waypoints[i].T, ts, order, duration * 100, p, None, 0
        )
        min_jerk_coeffs.append(min_jerk_coeffs)
    print("min_jerk_coeffs's shape", np.array(min_jerk_coeffs).shape)

    # get nn_coeffs for each traj
    nn_coeffs = []
    for i in range(num_traj):
        print(i)
        nn_coeffs.append(
            nonlinear.generate(
                waypoints[i],
                ts,
                order,
                duration * 100,
                p,
                rho,
                vf,
                min_jerk_coeffs[i],
                num_iter=100,
                lr=0.001,
            )
        )
    print("nn_coeffs's shape", np.array(nn_coeffs).shape)
"""

"""
    ###########foget about replanning for now
    # parameters for lissajous trajectory

    np.random.seed(3)

    x_amp = 2
    y_amp = 2
    z_amp = 0.8
    yaw_amp = 0.2

    x_num_periods = 2
    y_num_periods = 2
    z_num_periods = 2
    yaw_num_periods = 2

    # total period for all the trajectories
    period = 6
    p = 4
    order = 5
    Tref = period * 100

    movig_widow = 4
    num_waypoints_per_segment = 4
    duration = 3  # Duration of each replanning iteration

    # Generate the waypoints for the entire trajectory
    ref = generate_lissajous_traj(
        np.linspace(0, period, period * 100 + 1),
        x_num_periods,
        y_num_periods,
        z_num_periods,
        yaw_num_periods,
        period,
        x_amp,
        y_amp,
        z_amp,
        yaw_amp,
    )
    waypt = np.array(ref)[:, 0::30]
    # waypt = np.array(ref)[:, 0::50]
    # Get the number of segments and time samples
    segments = len(waypt.T) - 1
    print("Segments:", segments)
    ts = np.linspace(0, period, segments + 1)

    # we don't need to offset the z axis in sim
    offset = min(waypt[2, :])
    print("Negative offset", offset)
    waypt[2, :] = waypt[2, :] - offset + 0.1

    print(len(waypt.T))

    # Initialize the current waypoint index
    current_waypoint_index = 0
    idx = 0
    while current_waypoint_index < len(waypt.T) - num_waypoints_per_segment + 1:
        print("current_waypoint_index", current_waypoint_index)
        # Determine the start and end indices of the next waypoints and trajectory to consider
        start_idx = current_waypoint_index
        end_idx = start_idx + num_waypoints_per_segment
        # print("start_idx", start_idx)
        # print("end_idx", end_idx)

        # Select the waypoints for replanning
        selected_waypoints = waypt.T[start_idx:end_idx]
        print("selected_waypoints's shape", selected_waypoints.shape)
        mav_obj.publish_waypoints(selected_waypoints.T, 1.0, 0.0, 0.0, 0.9)

        # Replan the trajectory based on previous and next waypoints
        new_traj_coeffs, min_jerk_coeffs, nn_coeff = simple_replan(
            selected_waypoints, duration, order, p, vf, rho, idx
        )

        idx += 1

        Tref = duration * 100

        # Update the current waypoint index
        current_waypoint_index += movig_widow

        print("current_waypoint_index after update", current_waypoint_index)

        # Compute position, velocity, acceleration, jerk, yaw, and yaw rate from the new trajectory
        segment_new = len(selected_waypoints.T) - 1
        ts_new = np.linspace(0, duration, segment_new + 1)
        # pos, vel, acc, jerk, yaw, yaw_dot = compute_pos_vel_acc(Tref, new_traj_coeffs, segment_new, ts_new)
        # pos, vel, acc, jerk, yaw, yaw_dot = compute_pos_vel_acc(Tref, min_jerk_coeffs, segment_new, ts_new)
        pos, vel, acc, jerk, yaw, yaw_dot = compute_pos_vel_acc(
            Tref, nn_coeff, segment_new, ts_new
        )
        # pos[:, -3] = waypt[:3, end_idx]
        # pos[:, -2] = waypt[:3, end_idx]
        # pos[:, -1] = waypt[:3, end_idx]
        # vel[:,-1] = np.zeros(3)
        # acc[:,-1] = np.zeros(3)
        jerk[:, -1] = np.zeros(3)
        # yaw[-1] = 0
        # yaw_dot[-1] = 0

        # pos_nn, _, _, _, _, _ = compute_pos_vel_acc(Tref, nn_coeff, segment_new, ts_new)
        pos_mj, _, _, _, _, _ = compute_pos_vel_acc(
            Tref, min_jerk_coeffs, segment_new, ts_new
        )

        """
"""
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        # ttraj = actual_traj.copy()
        # axes = plt.gca(projection='3d')
        # 0510
        mav_id = 1
        axes.plot3D(pos[0, :], pos[1, :], pos[2, :], label='poly')
        axes.plot3D(pos_nn[0, :], pos_nn[1, :], pos_nn[2, :], label='nn')
        axes.plot3D(pos_mj[0, :], pos_mj[1, :], pos_mj[2, :], label='mj')
        axes.set_xlim(-1, 1)
        axes.set_zlim(0, 4)
        axes.set_ylim(-1, 1)
        axes.plot3D(waypt[0, :], waypt[1, :], waypt[2, :], '*')
        axes.legend()

        fig, ax = plt.subplots(1, 4)
        ax[0].plot(range(0, Tref), pos[:3, :].T, label=['x', 'y', 'z'])
        ax[1].plot(range(0, Tref), vel[:3, :].T, label=['vx', 'vy', 'vz'])
        ax[2].plot(range(0, Tref), acc[:3, :].T, label=['ax', 'ay', 'az'])
        ax[3].plot(range(0, Tref), jerk[:3, :].T, label=['jx', 'jy', 'jz'])
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[3].legend()
        # plt.savefig('./layered_ref_control/src/layered_ref_control/plots/traj_infot3'+str(mav_id)+'.png')
        # plt.savefig('./src/layered_ref_control/plots/traj_infot3'+str(mav_id)+'.png')

        # plt.legend(handles=['position', 'velocity', 'acceleration', 'jerk'])
        plt.show()
        # 0510
        """
"""

        start = rospy.Time.now()
        times_nn.append(start)
        # Pass commands to the controller at a certain frequency
        for i in range(len(pos.T)):
            rospy.logwarn("Publishing pos: %s", pos[:, i])
            mav_obj.publish_pos_cmd(
                pos[0, i],
                pos[1, i],
                pos[2, i],
                vel[0, i],
                vel[1, i],
                vel[2, i],
                acc[0, i],
                acc[1, i],
                acc[2, i],
                jerk[0, i],
                jerk[1, i],
                jerk[2, i],
                yaw[i],
                yaw_dot[i],
            )
            rate.sleep()

        end = rospy.Time.now()
        times_nn.append(end)

        rospy.logwarn("Reached initial waypoints")

    mav_obj.send_wp_block(
        pos[0, -1], pos[1, -1], pos[2, -1], 0.0, 0, 0, False
    )  # x, y, z, yaw, vel, acc, relative

    # save_object(duration, '/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/net_duration.pkl')
    save_object(
        times_nn,
        "/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/times_nn"
        + str(rho)
        + ".pkl",
    )
    # save_object(times_mj, '/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/times_mj.pkl')
    # save_object(times_poly,
    #            '/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/times_poly.pkl')
"""

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
