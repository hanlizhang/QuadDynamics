# The simulator is instantiated using the Environment class
from rotorpy.environments import Environment

# Vehicles. Currently there is only one.
# There must also be a corresponding parameter file.
from rotorpy.vehicles.multirotor import Multirotor

from rotorpy.vehicles.crazyflie_params import quad_params

# You will also need a controller (currently there is only one) that works for your vehicle.
from rotorpy.controllers.quadrotor_control import SE3Control

# And a trajectory generator
from rotorpy.trajectories.minsnap import MinSnap

# You can optionally specify a wind generator, although if no wind is specified it will default to NoWind().
from rotorpy.wind.default_winds import ConstantWind

# Other useful imports
import numpy as np  # For array creation/manipulation
import matplotlib.pyplot as plt  # For plotting, although the simulator has a built in plotter

from rotorpy.trajectories.pos_traj import PosTraj
from learning.trajgen import nonlinear_jax
from scipy.signal import convolve


def moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, mode="same")
    return smoothed_data


class VerifyInference:
    def __init__(
        self,
        points,
        yaw_angles=None,
        poly_degree=7,
        yaw_poly_degree=7,
        v_max=3,
        v_avg=1,
        v_start=[0, 0, 0],
        v_end=[0, 0, 0],
        use_neural_network=False,
        regularizer=None,
        fname=None,
        verify_by_pos=False,
        trained_model_state=None,
    ):
        self.points = points
        self.yaw_angles = yaw_angles
        self.poly_degree = poly_degree
        self.yaw_poly_degree = yaw_poly_degree
        self.v_max = v_max
        self.v_avg = v_avg
        self.v_start = v_start
        self.v_end = v_end
        self.use_neural_network = use_neural_network
        self.regularizer = regularizer
        self.fname = fname
        self.verify_by_pos = verify_by_pos
        self.trained_model_state = trained_model_state

    def run_simulation(self):
        # An instance of the simulator can be generated as follows:
        points = self.points
        yaw_angles = self.yaw_angles
        poly_degree = self.poly_degree
        yaw_poly_degree = self.yaw_poly_degree
        v_max = self.v_max
        v_avg = self.v_avg
        v_start = self.v_start
        v_end = self.v_end
        use_neural_network = self.use_neural_network
        regularizer = self.regularizer
        fname = self.fname
        verify_by_pos = self.verify_by_pos
        trained_model_state = self.trained_model_state

        # change the drag coefficient by increasing it on only one axis.
        # quad_params["motor_noise_std"] = 0
        quad_params["c_Dx"] = (0.5e-2) * 5
        # quad_params["k_d"] = 0
        # quad_params["k_z"] = 0
        # # quad_params["tau_m"] = 1e-4
        # quad_params["c_Dx"] = 0
        # quad_params["c_Dy"] = 0
        # quad_params["c_Dz"] = 0
        quad = Multirotor(quad_params)

        traj = MinSnap(
            points,
            yaw_angles,
            poly_degree,
            yaw_poly_degree,
            v_max=15,
            v_avg=v_avg,
            v_start=[0, v_avg, 0],
            v_end=[0, v_avg, 0],
            use_neural_network=use_neural_network,
            regularizer=regularizer,
        )

        if verify_by_pos:
            t_keyframes = traj.t_keyframes
            N = 502
            time = np.linspace(0, 1.1 * t_keyframes[-1], N)
            flat_output = traj.evaluate_trajectory(time)
            pos = flat_output[:, 0:3]
            vel = flat_output[:, 3:6]
            acc = flat_output[:, 6:9]
            jerk = flat_output[:, 9:12]
            snap = flat_output[:, 12:15]
            yaw = flat_output[:, 15]
            yaw_dot = flat_output[:, 16]
            yaw_ddot = flat_output[:, 17]

            # init_ref = np.vstack(pos, yaw)
            for i in range(N):
                if i == 0:
                    init_ref = np.hstack((pos[i, :], yaw[i]))
                else:
                    init_ref = np.vstack((init_ref, np.hstack((pos[i, :], yaw[i]))))

            aug_state = np.append(init_ref[0, :], init_ref)

            new_ref = nonlinear_jax.modify_ref(aug_state, trained_model_state)

            yaw_new = new_ref[7::4]
            offset = 0
            yaw_new = moving_average(yaw_new + offset, 5)
            # get pos_new (x, y, z)
            x_new = new_ref[0::4]
            y_new = new_ref[1::4]
            z_new = new_ref[2::4]
            # pos_new = np.vstack((x_new, y_new, z_new)).T

            # get new_ref from gradient descent
            traj = PosTraj(pos, vel, acc, jerk, snap, yaw_new, yaw_dot, yaw_ddot)

        sim_instance = Environment(
            vehicle=quad,
            controller=SE3Control(quad_params),
            trajectory=traj,
            wind_profile=ConstantWind(0, 0, 0),
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
            animate_bool=False,
            animate_wind=True,
            verbose=True,
            # fname=fname + ".mp4",
            fname=None,
        )

        # The save location is rotorpy/data_out/
        # csv file name
        csv_name = fname + ".csv"
        sim_instance.save_to_csv(csv_name)

        # Instead of producing a CSV, you can manually unpack the dictionary into a Pandas DataFrame using the following:
        from rotorpy.utils.postprocessing import unpack_sim_data

        dataframe = unpack_sim_data(results)

        return dataframe


if __name__ == "__main__":
    # load pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_ddot from csv file
    path = "/home/mrsl_guest/rotorpy/rotorpy/rotorpy/data_out/pos_min_jerk.csv"
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
