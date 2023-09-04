# The simulator is instantiated using the Environment class
from rotorpy.environments import Environment

# Vehicles. Currently there is only one.
# There must also be a corresponding parameter file.
from rotorpy.vehicles.multirotor import Multirotor

# from rotorpy.vehicles.crazyflie_params import quad_params

from rotorpy.vehicles.hummingbird_params import (
    quad_params,
)  # There's also the Hummingbird

# You will also need a controller (currently there is only one) that works for your vehicle.
from rotorpy.controllers.quadrotor_control import SE3Control

# And a trajectory generator
from rotorpy.trajectories.pos_traj import PosTraj

# You can optionally specify a wind generator, although if no wind is specified it will default to NoWind().
from rotorpy.wind.default_winds import NoWind, ConstantWind, SinusoidWind, LadderWind

# You can also optionally customize the IMU and motion capture sensor models. If not specified, the default parameters will be used.
from rotorpy.sensors.imu import Imu
from rotorpy.sensors.external_mocap import MotionCapture

# You can also specify a state estimator. This is optional. If no state estimator is supplied it will default to null.
from rotorpy.estimators.wind_ukf import WindUKF

# Compute the body rate
from learning.compute_body_rate import *

# Other useful imports
import numpy as np  # For array creation/manipulation
import matplotlib.pyplot as plt  # For plotting, although the simulator has a built in plotter
from scipy.spatial.transform import (
    Rotation,
)  # For doing conversions between different rotation descriptions, applying rotations, etc.
import os  # For path generation


class VerifyInference:
    def __init__(self, pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_ddot):
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.jerk = jerk
        self.snap = snap
        self.yaw = yaw
        self.yaw_dot = yaw_dot
        self.yaw_ddot = yaw_ddot

    def run_simulation(self):
        # An instance of the simulator can be generated as follows:
        pos = self.pos
        vel = self.vel
        acc = self.acc
        jerk = self.jerk
        snap = self.snap
        yaw = self.yaw
        yaw_dot = self.yaw_dot
        yaw_ddot = self.yaw_ddot

        traj = PosTraj(pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_ddot)
        quad = Multirotor(quad_params)

        sim_instance = Environment(
            vehicle=quad,
            controller=SE3Control(quad_params),
            trajectory=traj,
            wind_profile=ConstantWind(1, 1, 1),
            sim_rate=100,  # OPTIONAL: The update frequency of the simulator in Hz. Default is 100 Hz.
            imu=None,  # OPTIONAL: imu sensor object, if none is supplied it will choose a default IMU sensor.
            mocap=None,  # OPTIONAL: mocap sensor object, if none is supplied it will choose a default mocap.
            estimator=None,  # OPTIONAL: estimator object
            world=None,
            # world=world,  # OPTIONAL: the world, same name as the file in rotorpy/worlds/, default (None) is empty world
            safety_margin=0.25,  # OPTIONAL: defines the radius (in meters) of the sphere used for collision checking
        )
        # initial guess
        cmd_hover_speeds = np.sqrt(quad.mass * quad.g / (quad.num_rotors * quad.k_eta))
        flat_init = traj.update(0)
        print("flat_init", flat_init)
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

        # Executing the simulator using the "run" method
        results = sim_instance.run(
            t_final=5,
            use_mocap=False,
            terminate=False,
            plot=True,
            plot_mocap=False,
            plot_estimator=False,
            plot_imu=False,
            animate_bool=True,
            animate_wind=True,
            verbose=True,
            fname="const1_circle_5_modified.mp4",
        )

        # The save location is rotorpy/data_out/
        sim_instance.save_to_csv("const1_circle_5_modified.csv")

        # Instead of producing a CSV, you can manually unpack the dictionary into a Pandas DataFrame using the following:
        from rotorpy.utils.postprocessing import unpack_sim_data

        dataframe = unpack_sim_data(results)

        return dataframe


if __name__ == "__main__":
    # load pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_ddot from csv file
    path = "/home/mrsl_guest/rotorpy/rotorpy/rotorpy/data_out/pos.csv"
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    pos = data[:, 0:3]
    vel = data[:, 3:6]
    acc = data[:, 6:9]
    jerk = data[:, 9:12]
    snap = data[:, 12:15]
    yaw = data[:, 15]
    yaw_dot = data[:, 16]
    yaw_ddot = data[:, 17]
    # Create an instance of the VerifyInference class
    inference_results = VerifyInference(
        pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_ddot
    )

    # Run the simulation
    inference_results.run_simulation()
