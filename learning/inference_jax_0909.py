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

from scipy.spatial.transform import Rotation as R

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
    # actual_traj_quat = np.hstack((actual_traj_quat[:, 3:4], actual_traj_quat[:, 0:3]))
    # (cur_roll, cur_pitch, cur_yaw) = tf.transformations.euler_from_quaternion(actual_traj_quat)
    # 4 element sequence: w, x, y, z of quaternion
    # print("actual_traj_quat's shape: ", actual_traj_quat.shape)
    # actual_yaw = np.zeros(len(actual_traj_quat))
    # (cur_roll, cur_pitch, cur_yaw) = (
    #     np.zeros(len(actual_traj_quat)),
    #     np.zeros(len(actual_traj_quat)),
    #     np.zeros(len(actual_traj_quat)),
    # )
    # for i in range(len(actual_traj_quat)):
    #     (cur_roll[i], cur_pitch[i], cur_yaw[i]) = euler.quat2euler(actual_traj_quat[i])
    #     actual_yaw[i] = cur_yaw[i]
    euler_actual = R.from_quat(actual_traj_quat).as_euler("zyx", degrees=False)
    actual_yaw = euler_actual[:, 0]
    actual_traj = np.vstack((actual_traj_x, actual_traj_y, actual_traj_z, actual_yaw)).T
    # debug: print the first 10 actual_traj
    print("actual_traj: ", actual_traj[:10, :])

    # get the cmd input
    # col BN desired thrust from so3 controller
    input_traj_thrust = sim_data[:, 65]
    # print("input_traj_thrust's shape: ", input_traj_thrust.shape)

    # get the angular velocity from odometry: col M-O
    odom_ang_vel = sim_data[:, 12:15]

    input_traj = sim_data[:, 18:22]

    # debug: print the first 10 input_traj
    print("input_traj_motorspeed: ", input_traj[:10, :])

    # get the cost
    cost_traj = compute_tracking_error(
        ref_traj, actual_traj, input_traj, horizon, horizon, rho
    )
    # debug: print the first 10 cost_traj
    # print("cost_traj: ", cost_traj[:10, :])

    return ref_traj, actual_traj, input_traj, cost_traj, sim_data[:, 0]


def compute_tracking_error(ref_traj, actual_traj, input_traj, horizon, N, rho):
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
    rho = 0.01
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

    vf = model.bind(trained_model_state.params)

    desired_radius = 2
    num_waypoints = 10
    desired_freq = 0.2
    v_avg = desired_radius * (desired_freq * 2 * np.pi)

    waypoints = np.array(
        [
            [desired_radius * np.cos(alpha), desired_radius * np.sin(alpha), 0]
            for alpha in np.linspace(0, 2 * np.pi, num_waypoints)
        ]
    )

    yaw_angles = np.ones(num_waypoints)
    # yaw_

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

    # generate min_snap trajectory and run simulation
    fname_minsnap = f"min_snap_init_{rho}"

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
        fname=fname_minsnap,
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

    ## compute init_min_snap cost
    # Load the csv file
    sim_data_init_min_snap = np.loadtxt(
        "/home/mrsl_guest/rotorpy/rotorpy/rotorpy/data_out/" + fname_minsnap + ".csv",
        delimiter=",",
        skiprows=1,
    )
    print("sim_data_init_min_snap's type", type(sim_data_init_min_snap))
    # visualize the ref_traj_init_min_snap and actual_traj_init_min_snap for init_min_snap
    (
        ref_traj_init_min_snap,
        actual_traj_init_min_snap,
        input_traj_init_min_snap,
        cost_traj_init_min_snap,
        times_init_min_snap,
    ) = compute_traj(sim_data=sim_data_init_min_snap, horizon=num_data)
    fig = plt.figure()
    axes = fig.add_subplot(111, projection="3d")
    # red line is ref_traj_init_min_snap, blue line is actual_traj_init_min_snap
    axes.plot3D(
        ref_traj_init_min_snap[:, 0],
        ref_traj_init_min_snap[:, 1],
        ref_traj_init_min_snap[:, 2],
        "r",
    )
    axes.plot3D(
        actual_traj_init_min_snap[:, 0],
        actual_traj_init_min_snap[:, 1],
        actual_traj_init_min_snap[:, 2],
        "b",
    )
    # put legend
    axes.legend(["ref_traj_init_min_snap", "actual_traj_init_min_snap"])
    axes.set_xlim(-6, 6)
    axes.set_zlim(-6, 6)
    axes.set_ylim(-6, 6)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    title = "ref=min_snap"
    axes.set_title(title)
    plt.show()

    print("cost_traj's shape", np.array(cost_traj_init_min_snap).shape)
    print("init_min_snap", cost_traj_init_min_snap)

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
    # put legend
    axes.legend(["ref_traj_modified", "actual_traj_modified"])
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
    axes.plot(times_init_min_snap, actual_traj_init_min_snap[:, 3], "b")
    axes.plot(times_modified, actual_traj_modified[:, 3], "r")
    # put legend
    axes.legend(["actual_traj_init_min_snap", "actual_traj_modified"])
    axes.set_xlabel("time")
    axes.set_ylabel("yaw")
    axes.set_title("actual yaw")
    plt.show()

    # plot the ref yaw angle over time to see if it's actually varying from the network
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(times_init_min_snap, ref_traj_init_min_snap[:, 3], "b")
    axes.plot(times_modified, ref_traj_modified[:, 3], "r")
    # print("ref_traj_init_min_snap", ref_traj_init_min_snap[:, 3])
    # axis limits to be between 0 and 2pi
    axes.set_ylim(0, 2 * np.pi)
    # put legend
    axes.legend(["ref_traj_init_min_snap", "ref_traj_modified"])
    axes.set_xlabel("time")
    axes.set_ylabel("yaw")
    axes.set_title("ref yaw")
    plt.show()


if __name__ == "__main__":
    # try:
    #     main()
    # except rospy.ROSInterruptException:
    #     pass
    main()
