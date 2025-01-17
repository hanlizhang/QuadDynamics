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
from rotorpy.trajectories.coeff_poly2 import CoeffPoly

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
    def __init__(self, nn_coeff):
        self.nn_coeff = nn_coeff

    def run_simulation(self):
        # An instance of the simulator can be generated as follows:
        nn_coeff = self.nn_coeff
        traj = CoeffPoly(502, nn_coeff, nn_coeffs.shape[1], np.linspace(0, 5, 10))
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
            "x": flat_init["x_0"],
            "v": flat_init["x_dot_0"],
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
            fname="inference_result.mp4",
        )

        # # Saving simulation data as a CSV file
        # sim_instance.save_to_csv("const1_circle2_5_5.csv")

        # # Unpacking simulation data into a DataFrame
        # dataframe = unpack_sim_data(results)

        # return dataframe


if __name__ == "__main__":
    # load nn_coeffs from csv file
    path = "/home/mrsl_guest/rotorpy/rotorpy/rotorpy/data_out/nn_coeffs.csv"
    nn_coeffs = np.loadtxt(path, delimiter=",")
    nn_coeffs = nn_coeffs.reshape((nn_coeffs.shape[0], nn_coeffs.shape[1] // 6, 6))

    # Create an instance of the VerifyInference class
    inference_results = VerifyInference(nn_coeffs)

    # Run the simulation
    inference_results.run_simulation()
