#! /usr/bin/env python3

"""
Generate training data
simple replanning with lissajous trajectory with fixed waypoints
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D projection

# import rospy
import numpy as np
import random

from learning.trajgen import quadratic, nonlinear_jax, nonlinear, valuefunc
from examples.verify_inference_0909 import VerifyInference


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

import transforms3d.euler as euler
from itertools import accumulate

gamma = 1

PI = np.pi


def compute_traj(sim_data, rho=1, horizon=501, full_state=False):
    # TODO: full state

    # get the reference trajectory
    # col W-Y position
    ref_traj_x = sim_data[:, 22]
    ref_traj_y = sim_data[:, 23]
    ref_traj_z = sim_data[:, 24]
    # col AL yaw_des
    ref_traj_yaw = sim_data[:, 37]
    ref_traj = np.vstack((ref_traj_x, ref_traj_y, ref_traj_z, ref_traj_yaw)).T
    # debug: print the first 10 ref_traj
    print("ref_traj: ", ref_traj[:10, :])

    # get the actual trajectory
    # col C-E position
    actual_traj_x = sim_data[:, 2]
    actual_traj_y = sim_data[:, 3]
    actual_traj_z = sim_data[:, 4]
    # col I-L quaternion
    actual_traj_quat = sim_data[:, 8:12]
    # quat2euler takes a 4 element sequence: w, x, y, z of quaternion
    actual_traj_quat = np.hstack((actual_traj_quat[:, 3:4], actual_traj_quat[:, 0:3]))
    # (cur_roll, cur_pitch, cur_yaw) = tf.transformations.euler_from_quaternion(actual_traj_quat)
    # 4 element sequence: w, x, y, z of quaternion
    # print("actual_traj_quat's shape: ", actual_traj_quat.shape)
    actual_yaw = np.zeros(len(actual_traj_quat))
    (cur_roll, cur_pitch, cur_yaw) = (
        np.zeros(len(actual_traj_quat)),
        np.zeros(len(actual_traj_quat)),
        np.zeros(len(actual_traj_quat)),
    )
    for i in range(len(actual_traj_quat)):
        (cur_roll[i], cur_pitch[i], cur_yaw[i]) = euler.quat2euler(actual_traj_quat[i])
        actual_yaw[i] = cur_yaw[i]
    actual_traj = np.vstack((actual_traj_x, actual_traj_y, actual_traj_z, actual_yaw)).T
    # debug: print the first 10 actual_traj
    print("actual_traj: ", actual_traj[:10, :])
    # print("actual_traj's type: ", type(actual_traj))

    # get the cmd input
    # col BN desired thrust from so3 controller
    input_traj_thrust = sim_data[:, 65]
    # print("input_traj_thrust's shape: ", input_traj_thrust.shape)

    # get the angular velocity from odometry: col M-O
    odom_ang_vel = sim_data[:, 12:15]

    input_traj = sim_data[:, 18:22]

    # debug: print the first 10 input_traj
    print("input_traj: ", input_traj[:10, :])

    # get the cost
    cost_traj = compute_cum_tracking_cost(
        ref_traj, actual_traj, input_traj, horizon, horizon, rho
    )
    # debug: print the first 10 cost_traj
    # print("cost_traj: ", cost_traj[:10, :])

    return ref_traj, actual_traj, input_traj, cost_traj, sim_data[:, 0]


def compute_cum_tracking_cost(ref_traj, actual_traj, input_traj, horizon, N, rho):
    # print type of input
    print("ref_traj's type: ", type(ref_traj))
    print("actual_traj's type: ", type(actual_traj))
    print("input_traj's type: ", type(input_traj))

    import ipdb

    # ipdb.set_trace()

    m, n = ref_traj.shape
    num_traj = int(m / horizon)
    xcost = []
    for i in range(num_traj):
        act = actual_traj[i * horizon : (i + 1) * horizon, :]
        act = np.append(act, act[-1, :] * np.ones((N - 1, n)))
        act = np.reshape(act, (horizon + N - 1, n))
        r0 = ref_traj[i * horizon : (i + 1) * horizon, :]
        r0 = np.append(r0, r0[-1, :] * np.ones((N - 1, n)))
        r0 = np.reshape(r0, (horizon + N - 1, n))

        xcost.append(
            np.linalg.norm(act[:, :3] - r0[:, :3], axis=1) ** 2
            + angle_wrap(act[:, 3] - r0[:, 3]) ** 2
        )

    xcost.reverse()
    cost = []
    for i in range(num_traj):
        tot = list(accumulate(xcost[i], lambda x, y: x * gamma + y))
        # cost.append(np.log(tot[-1]))
        cost.append(tot[-1])
    cost.reverse()
    return np.vstack(cost)


# def input_traj_error(input_traj, input_traj_nn):


def angle_wrap(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


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
    snap = []

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

    dddd_x = compute_coeff_deriv(coeff_x, 4, segments)
    xdddd_ref = [np.poly1d(dddd_x[i, :]) for i in range(segments)]
    xdddd_ref = np.vstack(sampler(xdddd_ref, Tref, ts)).flatten()

    dddd_y = compute_coeff_deriv(coeff_y, 4, segments)
    ydddd_ref = [np.poly1d(dddd_y[i, :]) for i in range(segments)]
    ydddd_ref = np.vstack(sampler(ydddd_ref, Tref, ts)).flatten()

    dddd_z = compute_coeff_deriv(coeff_z, 4, segments)
    zdddd_ref = [np.poly1d(dddd_z[i, :]) for i in range(segments)]
    zdddd_ref = np.vstack(sampler(zdddd_ref, Tref, ts)).flatten()
    snap.append([xdddd_ref, ydddd_ref, zdddd_ref])

    yaw_ref = [np.poly1d(coeff_yaw[i, :]) for i in range(segments)]
    yaw_ref = np.vstack(sampler(yaw_ref, Tref, ts)).flatten()

    dot_yaw = compute_coeff_deriv(coeff_yaw, 1, segments)
    yawdot_ref = [np.poly1d(dot_yaw[i, :]) for i in range(segments)]
    yawdot_ref = np.vstack(sampler(yawdot_ref, Tref, ts)).flatten()

    ddot_yaw = compute_coeff_deriv(coeff_yaw, 2, segments)
    yawddot_ref = [np.poly1d(ddot_yaw[i, :]) for i in range(segments)]
    yawddot_ref = np.vstack(sampler(yawddot_ref, Tref, ts)).flatten()

    return (
        np.vstack(pos),
        np.vstack(vel),
        np.vstack(acc),
        np.vstack(jerk),
        np.vstack(snap),
        yaw_ref,
        yawdot_ref,
        yawddot_ref,
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


def main():
    # Define the lists to keep track of times for the simulations
    times_nn = []
    times_mj = []
    times_poly = []

    # Initialize neural network
    rho = 5
    input_size = 2012
    num_data = 502

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
    inp = jax.random.normal(
        inp_rng, (batch_size, 2012)
    )  # Batch size 32, input size 2012
    # Initialize the model
    params = model.init(init_rng, inp)

    optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)

    model_state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    model_save = yaml_data["save_path"] + str(rho)
    print("model_save", model_save)

    trained_model_state = restore_checkpoint(model_state, model_save)

    # mlp_t = load_torch_model(trained_model_state, input_size, num_hidden)

    # print(mlp_t)

    # vf = valuefunc.MLPValueFunc(mlp_t)

    # vf.network = mlp_t

    vf = model.bind(trained_model_state.params)

    desired_radius = 3
    num_waypoints = 10
    desired_freq = 0.2
    v_avg = desired_radius * (desired_freq * 2 * np.pi)

    waypoints = np.array(
        [
            [desired_radius * np.cos(alpha), desired_radius * np.sin(alpha), 0]
            for alpha in np.linspace(0, 2 * np.pi, num_waypoints)
        ]
    )

    yaw_angles = np.array(0 * np.ones(num_waypoints))

    # visualize the waypoints
    fig = plt.figure()
    axes = fig.add_subplot(111, projection="3d")
    axes.plot3D(
        waypoints[:, 0],
        waypoints[:, 1],
        waypoints[:, 2],
        "*",
    )
    axes.set_xlim(-6, 6)
    axes.set_zlim(0, 1)
    axes.set_ylim(-6, 6)
    plt.show()

    # generate min_jerk trajectory and run simulation
    fname_minjerk = f"min_jerk_init_{rho}"

    init_inference_results = VerifyInference(
        waypoints,
        yaw_angles,
        poly_degree=7,
        yaw_poly_degree=7,
        v_max=15,
        v_avg=v_avg,
        v_start=[0, v_avg, 0],
        v_end=[0, v_avg, 0],
        use_neural_network=False,
        regularizer=None,
        fname=fname_minjerk,
    )

    init_inference_results.run_simulation()

    # generate nn trajectory and run simulation
    fname_nn = f"nn_modified_{rho}"

    modified_inference_results = VerifyInference(
        waypoints,
        yaw_angles,
        poly_degree=7,
        yaw_poly_degree=7,
        v_max=15,
        v_avg=v_avg,
        v_start=[0, v_avg, 0],
        v_end=[0, v_avg, 0],
        use_neural_network=True,
        regularizer=vf,
        fname=fname_nn,
    )

    modified_inference_results.run_simulation()

    """
    # add yaw=0 to waypoints
    waypoints = np.hstack((waypoints, np.zeros((num_waypoints, 1))))
    print("waypoints's shape", np.array(waypoints).shape)

    p = 4  # num_dimensions
    order = 5
    duration = 5
    ts = np.linspace(0, duration, len(waypoints))
    # print("waypoints[i]'s shape", waypoints[i].shape)

    # get min_jerk_coeffs for each traj
    _, min_jerk_coeffs = quadratic.generate(waypoints, ts, order, num_data, p, None, 0)
    print("min_jerk_coeffs's shape", np.array(min_jerk_coeffs).shape)
    # print("min_jerk_coeffs", min_jerk_coeffs)
    # set 4th coeff for yaw to 0
    # min_jerk_coeffs[3, :, :] = 0.0

    # get pos_init, vel_init, acc_init, jerk_init, snap_init, yaw_init, yawdot_init, yawddot_init for each traj
    (
        pos_init,
        vel_init,
        acc_init,
        jerk_init,
        snap_init,
        yaw_init,
        yawdot_init,
        yawddot_init,
    ) = compute_pos_vel_acc(num_data, min_jerk_coeffs, num_waypoints - 1, ts)

    fname = f"min_jerk_init_{rho}"

    init_inference_results = VerifyInference(
        pos_init,
        vel_init,
        acc_init,
        jerk_init,
        snap_init,
        yaw_init,
        yawdot_init,
        yawddot_init,
        fname,
    )

    init_inference_results.run_simulation()

    # get nn_coeffs for each traj
    nn_coeffs, pred = nonlinear_jax.modify_reference(
        waypoints,
        ts,
        num_data,
        order,
        p,
        vf,
        min_jerk_coeffs,
    )

    print("nn_coeffs's shape", np.array(nn_coeffs).shape)  # (4, 9, 6)
    # print("nn_coeffs", nn_coeffs)

    # get pos_nn, vel_nn, acc_nn, jerk_nn, snap_nn, yaw_nn, yawdot_nn, yawddot_nn for each traj
    (
        pos_nn,
        vel_nn,
        acc_nn,
        jerk_nn,
        snap_nn,
        yaw_nn,
        yawdot_nn,
        yawddot_nn,
    ) = compute_pos_vel_acc(num_data, nn_coeffs, num_waypoints - 1, ts)

    fname_nn = f"nn_modified_{rho}"

    modified_inference_results = VerifyInference(
        pos_nn,
        vel_nn,
        acc_nn,
        jerk_nn,
        snap_nn,
        yaw_nn,
        yawdot_nn,
        yawddot_nn,
        fname_nn,
    )

    modified_inference_results.run_simulation()
    """
    ## compute init_min_jerk cost
    # Load the csv file
    sim_data_init_min_jerk = np.loadtxt(
        "/home/mrsl_guest/rotorpy/rotorpy/rotorpy/data_out/" + fname_minjerk + ".csv",
        delimiter=",",
        skiprows=1,
    )
    print("sim_data_init_min_jerk's type", type(sim_data_init_min_jerk))
    # visualize the ref_traj_init_min_jerk and actual_traj_init_min_jerk for init_min_jerk
    (
        ref_traj_init_min_jerk,
        actual_traj_init_min_jerk,
        input_traj_init_min_jerk,
        cost_traj_init_min_jerk,
        times_init_min_jerk,
    ) = compute_traj(sim_data=sim_data_init_min_jerk, horizon=num_data)
    fig = plt.figure()
    axes = fig.add_subplot(111, projection="3d")
    # red line is ref_traj_init_min_jerk, blue line is actual_traj_init_min_jerk
    axes.plot3D(
        ref_traj_init_min_jerk[:, 0],
        ref_traj_init_min_jerk[:, 1],
        ref_traj_init_min_jerk[:, 2],
        "r",
    )
    axes.plot3D(
        actual_traj_init_min_jerk[:, 0],
        actual_traj_init_min_jerk[:, 1],
        actual_traj_init_min_jerk[:, 2],
        "b",
    )
    axes.set_xlim(-6, 6)
    axes.set_zlim(-6, 6)
    axes.set_ylim(-6, 6)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    title = "ref=min_jerk"
    axes.set_title(title)
    plt.show()

    print("cost_traj's shape", np.array(cost_traj_init_min_jerk).shape)
    print("init_min_jerk", cost_traj_init_min_jerk)

    ## compute modified_true cost
    # Load the csv file
    sim_data_modified = np.loadtxt(
        "/home/mrsl_guest/rotorpy/rotorpy/rotorpy/data_out/" + fname_nn + ".csv",
        delimiter=",",
        skiprows=1,
    )
    # visualize the ref_traj_modified and actual_traj_modified for modified_true
    (
        ref_traj_modified,
        actual_traj_modified,
        input_traj_modified,
        cost_traj_modified,
        times_modified,
    ) = compute_traj(sim_data=sim_data_modified, horizon=num_data)
    fig = plt.figure()
    axes = fig.add_subplot(111, projection="3d")
    # red line is ref_traj_modified, blue line is actual_traj_modified
    axes.plot3D(
        ref_traj_modified[:, 0],
        ref_traj_modified[:, 1],
        ref_traj_modified[:, 2],
        "r",
    )
    axes.plot3D(
        actual_traj_modified[:, 0],
        actual_traj_modified[:, 1],
        actual_traj_modified[:, 2],
        "b",
    )
    axes.set_xlim(-6, 6)
    axes.set_zlim(-6, 6)
    axes.set_ylim(-6, 6)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    title = "ref=nn_coeff"
    axes.set_title(title)
    plt.show()

    print("cost_traj's shape", np.array(cost_traj_modified).shape)
    print("modified_true", cost_traj_modified)

    # plot the actual yaw angle over time to see if it's actually varying from the network
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(times_init_min_jerk, actual_traj_init_min_jerk[:, 3], "b")
    axes.plot(times_modified, actual_traj_modified[:, 3], "r")
    axes.set_xlabel("time")
    axes.set_ylabel("yaw")
    axes.set_title("actual yaw")
    plt.show()

    # plot the ref yaw angle over time to see if it's actually varying from the network
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(times_init_min_jerk, ref_traj_init_min_jerk[:, 3], "b")
    axes.plot(times_modified, ref_traj_modified[:, 3], "r")
    axes.set_xlabel("time")
    axes.set_ylabel("yaw")
    axes.set_title("ref yaw")
    plt.show()

    """
    # # # save to csv file and visualization for min_jerk# # #
    # compute pos, vel, acc, jerk, yaw, yaw_dot from min_jerk_coeffs
    pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_ddot = compute_pos_vel_acc(
        502, min_jerk_coeffs, min_jerk_coeffs.shape[1], ts
    )

    # visualize the trajectory
    fig = plt.figure()
    axes = fig.add_subplot(111, projection="3d")
    axes.plot3D(pos[0], pos[1], pos[2], "r")
    axes.set_xlim(-6, 6)
    axes.set_zlim(0, 1)
    axes.set_ylim(-6, 6)
    plt.show()

    # save pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_ddot to csv file
    path = "/home/mrsl_guest/rotorpy/rotorpy/rotorpy/data_out/pos_min_jerk.csv"
    header = [
        "pos_x",
        "pos_y",
        "pos_z",
        "vel_x",
        "vel_y",
        "vel_z",
        "acc_x",
        "acc_y",
        "acc_z",
        "jerk_x",
        "jerk_y",
        "jerk_z",
        "snap_x",
        "snap_y",
        "snap_z",
        "yaw",
        "yaw_dot",
        "yaw_ddot",
    ]
    pos = np.array(pos).T
    vel = np.array(vel).T
    acc = np.array(acc).T
    jerk = np.array(jerk).T
    snap = np.array(snap).T
    yaw = np.array(yaw).reshape((-1, 1))
    yaw_dot = np.array(yaw_dot).reshape((-1, 1))
    yaw_ddot = np.array(yaw_ddot).reshape((-1, 1))

    print("pos's shape", np.array(pos).shape)
    print("vel's shape", np.array(vel).shape)
    print("acc's shape", np.array(acc).shape)
    print("jerk's shape", np.array(jerk).shape)
    print("snap's shape", np.array(snap).shape)
    print("yaw's shape", np.array(yaw).shape)
    print("yaw_dot's shape", np.array(yaw_dot).shape)
    print("yaw_ddot's shape", np.array(yaw_ddot).shape)

    data = np.hstack((pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_ddot))
    np.savetxt(path, data, delimiter=",", header=",".join(header))

    # # # save to csv file and visualization for nn# # #
    # compute pos, vel, acc, jerk, yaw, yaw_dot from nn_coeffs
    pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_ddot = compute_pos_vel_acc(
        502, nn_coeffs, nn_coeffs.shape[1], ts
    )

    # visualize the trajectory
    fig = plt.figure()
    axes = fig.add_subplot(111, projection="3d")
    axes.plot3D(pos[0], pos[1], pos[2], "r")
    axes.set_xlim(-6, 6)
    axes.set_zlim(0, 1)
    axes.set_ylim(-6, 6)
    plt.show()

    # save pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_ddot to csv file
    path = "/home/mrsl_guest/rotorpy/rotorpy/rotorpy/data_out/pos_from_nn_coeff.csv"
    header = [
        "pos_x",
        "pos_y",
        "pos_z",
        "vel_x",
        "vel_y",
        "vel_z",
        "acc_x",
        "acc_y",
        "acc_z",
        "jerk_x",
        "jerk_y",
        "jerk_z",
        "snap_x",
        "snap_y",
        "snap_z",
        "yaw",
        "yaw_dot",
        "yaw_ddot",
    ]
    pos = np.array(pos).T
    vel = np.array(vel).T
    acc = np.array(acc).T
    jerk = np.array(jerk).T
    snap = np.array(snap).T
    yaw = np.array(yaw).reshape((-1, 1))
    yaw_dot = np.array(yaw_dot).reshape((-1, 1))
    yaw_ddot = np.array(yaw_ddot).reshape((-1, 1))

    print("pos's shape", np.array(pos).shape)
    print("vel's shape", np.array(vel).shape)
    print("acc's shape", np.array(acc).shape)
    print("jerk's shape", np.array(jerk).shape)
    print("snap's shape", np.array(snap).shape)
    print("yaw's shape", np.array(yaw).shape)
    print("yaw_dot's shape", np.array(yaw_dot).shape)
    print("yaw_ddot's shape", np.array(yaw_ddot).shape)

    data = np.hstack((pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_ddot))
    np.savetxt(path, data, delimiter=",", header=",".join(header))
    """


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
    # try:
    #     main()
    # except rospy.ROSInterruptException:
    #     pass
    main()
