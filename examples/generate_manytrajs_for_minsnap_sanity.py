# Required imports
from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor

from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.minsnap import MinSnap
from rotorpy.wind.default_winds import NoWind
import numpy as np
from scipy.spatial.transform import (
    Rotation,
)  # For doing conversions between different rotation descriptions, applying rotations, etc.
import csv
import os
import pandas as pd


def save_to_csv(dataframe, fname):
    """
    Save the simulation data in a DataFrame to a CSV file.
    """
    if not fname.endswith(".csv"):
        fname += ".csv"
    path = "/home/mrsl_guest/rotorpy/rotorpy/rotorpy/data_out"
    path = os.path.join(path, fname)
    dataframe.to_csv(path, index=False)


def unpack_sim_data(result):
    """
    Unpacks the simulated result dictionary and converts it to a Pandas DataFrame

    """

    headers = [
        "time",  # Time
        "x",
        "y",
        "z",
        "xdot",
        "ydot",
        "zdot",
        "qx",
        "qy",
        "qz",
        "qw",
        "wx",
        "wy",
        "wz",
        "windx",
        "windy",
        "windz",
        "r1",
        "r2",
        "r3",
        "r4",  # Vehicle state                                                                                                                                               # GT body velocity
        "xdes",
        "ydes",
        "zdes",
        "xdotdes",
        "ydotdes",
        "zdotdes",
        "xddotdes",
        "yddotdes",
        "zddotdes",
        "xdddotdes",
        "ydddotdes",
        "zdddotdes",  # Flat outputs
        "xddddotdes",
        "yddddotdes",
        "zddddotdes",
        "yawdes",
        "yawdotdes",
        "ax",
        "ay",
        "az",
        "ax_gt",
        "ay_gt",
        "az_gt",
        "gx",
        "gy",
        "gz",  # IMU measurements
        "mocap_x",
        "mocap_y",
        "mocap_z",
        "mocap_xdot",
        "mocap_ydot",
        "mocap_zdot",
        "mocap_qx",
        "mocap_qy",
        "mocap_qz",
        "mocap_qw",
        "mocap_wx",
        "mocap_wy",
        "mocap_wz",  # Mocap measurements
        "r1des",
        "r2des",
        "r3des",
        "r4des",
        "thrustdes",
        "qxdes",
        "qydes",
        "qzdes",
        "qwdes",
        "mxdes",
        "mydes",
        "mzdes",  # Controller
    ]

    # Extract data into numpy arrays
    time = result["time"].reshape(-1, 1)
    state = result["state"]
    x = state["x"]
    v = state["v"]
    q = state["q"]
    w = state["w"]
    wind = state["wind"]
    rotor_speeds = state["rotor_speeds"]
    control = result["control"]
    cmd_rotor = control["cmd_motor_speeds"]
    cmd_thrust = control["cmd_thrust"].reshape(-1, 1)
    cmd_q = control["cmd_q"]
    cmd_moment = control["cmd_moment"]

    flat = result["flat"]
    x_des = flat["x"]
    v_des = flat["x_dot"]
    a_des = flat["x_ddot"]
    j_des = flat["x_dddot"]
    s_des = flat["x_ddddot"]
    yaw_des = flat["yaw"].reshape(-1, 1)
    yawdot_des = flat["yaw_dot"].reshape(-1, 1)

    imu_measurements = result["imu_measurements"]
    a_measured = imu_measurements["accel"]
    w_measured = imu_measurements["gyro"]

    mocap_measurements = result["mocap_measurements"]
    x_mc = mocap_measurements["x"]
    v_mc = mocap_measurements["v"]
    q_mc = mocap_measurements["q"]
    w_mc = mocap_measurements["w"]

    imu_actual = result["imu_gt"]
    a_actual = imu_actual["accel"]

    state_estimate = result["state_estimate"]
    filter_state = state_estimate["filter_state"]
    covariance = state_estimate["covariance"]

    if filter_state.shape[1] > 0:
        # Computes the standard deviation of the filter covariance diagonal elements
        sd = 3 * np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
        headers.extend(["xhat_" + str(i) for i in range(filter_state.shape[1])])
        headers.extend(["sigma_" + str(i) for i in range(filter_state.shape[1])])

        dataset = np.hstack(
            (
                time,
                x,
                v,
                q,
                w,
                wind,
                rotor_speeds,
                x_des,
                v_des,
                a_des,
                j_des,
                s_des,
                yaw_des,
                yawdot_des,
                a_measured,
                a_actual,
                w_measured,
                x_mc,
                v_mc,
                q_mc,
                w_mc,
                cmd_rotor,
                cmd_thrust,
                cmd_q,
                cmd_moment,
                filter_state,
                sd,
            )
        )
    else:
        sd = []

        dataset = np.hstack(
            (
                time,
                x,
                v,
                q,
                w,
                wind,
                rotor_speeds,
                x_des,
                v_des,
                a_des,
                j_des,
                s_des,
                yaw_des,
                yawdot_des,
                a_measured,
                a_actual,
                w_measured,
                x_mc,
                v_mc,
                q_mc,
                w_mc,
                cmd_rotor,
                cmd_thrust,
                cmd_q,
                cmd_moment,
            )
        )
    df = pd.DataFrame(data=dataset, columns=headers)

    return df


def main():
    # Create a list to store all the simulation data
    all_simulation_data = []

    # change the drag coefficient by increasing it on only one axis.
    quad_params["c_Dx"] = quad_params["c_Dx"] * 5
    # quad_params['c_Dy'] = quad_params['c_Dy']/2  # You can also decrease the drag force on an axis too.
    quad = Multirotor(quad_params)

    # Set up the circular trajectory waypoints.
    desired_freq = 0.2  # In Hz (1/s). To convert to rad/s, we multiply by 2*pi

    num_waypoints = 10

    # Create a list of radii ranging from 1 to 8 meters
    desired_radius = 4

    v_avg = desired_radius * (
        desired_freq * 2 * np.pi
    )  # Get the average speed by multiplying radius by frequency (units should be m/s)
    waypoints = np.array(
        [
            [desired_radius * np.cos(alpha), desired_radius * np.sin(alpha), 0]
            for alpha in np.linspace(0, 2 * np.pi, num_waypoints)
        ]
    )  # Equally spaced waypoints.

    # for fixed yaw
    yaw_fixed = np.ones(num_waypoints)

    traj = MinSnap(
        waypoints,
        yaw_fixed,
        v_max=15,
        v_avg=v_avg,
        v_start=[0, v_avg, 0],
        v_end=[0, v_avg, 0],
    )

    sim_instance = Environment(
        vehicle=quad,  # vehicle object, must be specified.
        controller=SE3Control(
            quad_params, drag_compensation=True
        ),  # controller object, must be specified.
        trajectory=traj,
        wind_profile=NoWind(),
        sim_rate=100,  # OPTIONAL: The update frequency of the simulator in Hz. Default is 100 Hz.
        imu=None,  # OPTIONAL: imu sensor object, if none is supplied it will choose a default IMU sensor.
        mocap=None,  # OPTIONAL: mocap sensor object, if none is supplied it will choose a default mocap.
        estimator=None,  # OPTIONAL: estimator object
        world=None,
        # world=world,  # OPTIONAL: the world, same name as the file in rotorpy/worlds/, default (None) is empty world
        safety_margin=0.25,  # OPTIONAL: defines the radius (in meters) of the sphere used for collision checking
    )

    cmd_hover_speeds = np.sqrt(quad.mass * quad.g / (quad.num_rotors * quad.k_eta))
    flat_init = traj.update(0)
    x0 = {
        "x": flat_init["x"],
        "v": flat_init["x_dot"],
        "q": np.array([0, 0, 0, 1]),  # [i,j,k,w]
        "w": np.zeros(
            3,
        ),
        "wind": np.array(
            [0, 0, 0]
        ),  # Since wind is handled elsewhere, this value is overwritten
        "rotor_speeds": np.ones(quad.num_rotors) * cmd_hover_speeds,
    }
    sim_instance.vehicle.initial_state = x0

    # Execute the simulator for the specified duration
    results = sim_instance.run(
        t_final=5,
        use_mocap=False,
        terminate=False,
        plot=False,
        plot_mocap=False,
        plot_estimator=False,
        plot_imu=False,
        animate_bool=True,
        animate_wind=False,
        verbose=True,
        fname="r4_fixed_yaw1_for_sanity_check.mp4",
    )

    # Convert simulation results to a DataFrame
    simulation_data = unpack_sim_data(results)

    # Append simulation data to the all_simulation_data list
    all_simulation_data.append(simulation_data)

    # for travaling yaw

    yaw_travel_angles = np.array(
        [alpha for alpha in np.linspace(0, 2 * np.pi, num_waypoints)]
    )  # This is how you would make sure the quad is always facing towards the direction of travel.

    traj = MinSnap(
        waypoints,
        yaw_travel_angles,
        v_max=15,
        v_avg=v_avg,
        v_start=[0, v_avg, 0],
        v_end=[0, v_avg, 0],
    )

    sim_instance = Environment(
        vehicle=quad,  # vehicle object, must be specified.
        controller=SE3Control(quad_params),  # controller object, must be specified.
        trajectory=traj,
        wind_profile=NoWind(),
        sim_rate=100,  # OPTIONAL: The update frequency of the simulator in Hz. Default is 100 Hz.
        imu=None,  # OPTIONAL: imu sensor object, if none is supplied it will choose a default IMU sensor.
        mocap=None,  # OPTIONAL: mocap sensor object, if none is supplied it will choose a default mocap.
        estimator=None,  # OPTIONAL: estimator object
        world=None,
        # world=world,  # OPTIONAL: the world, same name as the file in rotorpy/worlds/, default (None) is empty world
        safety_margin=0.25,  # OPTIONAL: defines the radius (in meters) of the sphere used for collision checking
    )

    cmd_hover_speeds = np.sqrt(quad.mass * quad.g / (quad.num_rotors * quad.k_eta))
    flat_init = traj.update(0)
    x0 = {
        "x": flat_init["x"],
        "v": flat_init["x_dot"],
        "q": np.array([0, 0, 0, 1]),  # [i,j,k,w]
        "w": np.zeros(
            3,
        ),
        "wind": np.array(
            [0, 0, 0]
        ),  # Since wind is handled elsewhere, this value is overwritten
        "rotor_speeds": np.ones(quad.num_rotors) * cmd_hover_speeds,
    }
    sim_instance.vehicle.initial_state = x0

    # Execute the simulator for the specified duration
    results = sim_instance.run(
        t_final=5,
        use_mocap=False,
        terminate=False,
        plot=False,
        plot_mocap=False,
        plot_estimator=False,
        plot_imu=False,
        animate_bool=True,
        animate_wind=False,
        verbose=True,
        fname="r4_travel_yaw_for_sanity_check.mp4",
    )

    # Convert simulation results to a DataFrame
    simulation_data_travel = unpack_sim_data(results)

    # Append simulation data to the all_simulation_data list
    all_simulation_data.append(simulation_data_travel)

    # Concatenate all simulation data DataFrames into one
    concatenated_data = pd.concat(all_simulation_data, ignore_index=False)

    # Save all the trajectories' simulation data into one CSV file with timestamp of the current date and time as part of the name
    current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    # # add one column for the csv file to indicate the radius of the trajectory
    # concatenated_data["radius"] = np.concatenate(
    #     [np.ones(500) * radius for radius in desired_radius]
    # )
    # concatenated_data["yaw"] = np.concatenate(
    #     [np.ones(100) * yaw for yaw in yaw_values]
    # )
    save_to_csv(concatenated_data, f"sanity_check_{current_time}.csv")

    print("All simulations completed!")


if __name__ == "__main__":
    main()
