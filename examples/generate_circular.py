# Required imports
from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor

# from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import CircularTraj
from rotorpy.wind.default_winds import NoWind, ConstantWind, SinusoidWind, LadderWind
import numpy as np
import os

# Create a list of radii ranging from 2 to 5
radii = np.linspace(2, 5, 10)
# # generate trajectories for one radius 10 times
# radii = np.ones(10) * 5

# Loop through each radius and generate the circular trajectory simulation
for idx, radius in enumerate(radii):
    print(
        f"Running simulation for circular trajectory {idx + 1} with radius {radius}..."
    )

    # Instantiate the circular trajectory with the current radius
    circular_traj = CircularTraj(radius=radius)

    # Instantiate the simulator environment
    sim_instance = Environment(
        vehicle=Multirotor(quad_params),  # vehicle object, must be specified.
        controller=SE3Control(quad_params),  # controller object, must be specified.
        trajectory=circular_traj,
        wind_profile=ConstantWind(1, 1, 1),
        sim_rate=100,  # OPTIONAL: The update frequency of the simulator in Hz. Default is 100 Hz.
        imu=None,  # OPTIONAL: imu sensor object, if none is supplied it will choose a default IMU sensor.
        mocap=None,  # OPTIONAL: mocap sensor object, if none is supplied it will choose a default mocap.
        estimator=None,  # OPTIONAL: estimator object
        world=None,
        # world=world,  # OPTIONAL: the world, same name as the file in rotorpy/worlds/, default (None) is empty world
        safety_margin=0.25,  # OPTIONAL: defines the radius (in meters) of the sphere used for collision checking
    )

    # Set the initial state at the beginning of the trajectory instead of the origin
    x = circular_traj.update(0)
    x0 = {
        "x": x["x"],
        "v": np.zeros(3),
        "q": np.array([0, 0, 0, 1]),  # [i, j, k, w]
        "w": np.zeros(3),
        "wind": np.array([0, 0, 0]),
        "rotor_speeds": np.array([1788.53, 1788.53, 1788.53, 1788.53]),
    }
    sim_instance.vehicle.initial_state = x0

    # Execute the simulator for the specified duration
    results = sim_instance.run(
        t_final=10,
        use_mocap=False,
        terminate=False,
        plot=False,
        plot_mocap=False,
        plot_estimator=False,
        plot_imu=False,
        animate_bool=True,
        animate_wind=True,
        verbose=True,
        fname=None,
    )

    # Save all the trajs' simulation data into one csv file called constwind_1.csv
    sim_instance.save_to_csv(f"circle_traj_{idx + 1}.csv")

print("All simulations completed!")
