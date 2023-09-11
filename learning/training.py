"""
Training using the bags
"""

import numpy as np
import matplotlib.pyplot as plt
from model_learning import (
    TrajDataset,
    train_model,
    eval_model,
    numpy_collate,
    save_checkpoint,
    restore_checkpoint,
)
import ruamel.yaml as yaml
import torch.utils.data as data
from flax.training import train_state
import optax
import jax
from mlp import MLP

# import tf
import transforms3d.euler as euler
from itertools import accumulate
from sklearn.model_selection import train_test_split

from scipy.spatial.transform import Rotation as R

gamma = 1


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
    # actual_traj = np.vstack((actual_traj_x, actual_traj_y, actual_traj_z, actual_yaw)).T
    # debug: print the first 10 actual_traj
    # print("actual_traj: ", actual_traj[:10, :])
    # print("actual_traj's type: ", type(actual_traj))
    euler_actual = R.from_quat(actual_traj_quat).as_euler("zyx", degrees=False)
    actual_yaw = euler_actual[:, 0]
    actual_traj = np.vstack((actual_traj_x, actual_traj_y, actual_traj_z, actual_yaw)).T
    # get the cmd input
    # col BN desired thrust from so3 controller
    input_traj_thrust = sim_data[:, 65]
    # print("input_traj_thrust's shape: ", input_traj_thrust.shape)

    # get the angular velocity from odometry: col M-O
    odom_ang_vel = sim_data[:, 12:15]

    """
    # coverting quaternion to euler angle and get the angular velocity
    # col BO-BR desired orientation from so3 controller(quaternion)
    input_traj_quat = sim_data[:, 66:70]
    # (cmd_roll, cmd_pitch, cmd_yaw) = tf.transformations.euler_from_quaternion(input_traj_quat)
    input_traj_yaw = np.zeros(len(input_traj_quat))
    (cmd_roll, cmd_pitch, cmd_yaw) = (
        np.zeros(len(input_traj_quat)),
        np.zeros(len(input_traj_quat)),
        np.zeros(len(input_traj_quat)),
    )
    for i in range(len(input_traj_quat)):
        (cmd_roll[i], cmd_pitch[i], cmd_yaw[i]) = euler.quat2euler(input_traj_quat[i])

    # devided by time difference to get the angular velocity x, y, z
    input_traj_ang_vel = (
        np.diff(np.vstack((cmd_roll, cmd_pitch, cmd_yaw)).T, axis=0) / 0.01
    )
    # add the first element to the first row, so that the shape is the same as input_traj_thrust
    input_traj_ang_vel = np.vstack((input_traj_ang_vel[0, :], input_traj_ang_vel))
    # print("input_traj_ang_vel's shape: ", input_traj_ang_vel.shape)
    
    input_traj = np.hstack((input_traj_thrust.reshape(-1, 1), input_traj_ang_vel))
    """

    # input_traj = np.hstack((input_traj_thrust.reshape(-1, 1), odom_ang_vel))
    # input_traj is sum of squares of motor speeds for 4 individual motors, col 19-22
    motor_speed = sim_data[:, 18:22]
    input_traj = np.sqrt(np.sum(motor_speed**2, axis=1)).reshape(-1, 1)

    # debug: print the first 10 input_traj
    print("input_traj_motorspeed: ", input_traj)

    # get the cost
    cost_traj = compute_cum_tracking_cost(
        ref_traj, actual_traj, input_traj, horizon, horizon, rho
    )
    # debug: print the first 10 cost_traj
    print("cost_traj: ", cost_traj[:10, :])

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

        # print("delta yaw: ", act[:, 3] - r0[:, 3])
        # print("angle_wrap: ", angle_wrap(act[:, 3] - r0[:, 3]))
        # print("yaw cost: ", angle_wrap(act[:, 3] - r0[:, 3]))
        xcost.append(
            rho
            * (
                np.linalg.norm(act[:, :3] - r0[:, :3], axis=1) ** 2
                + angle_wrap(act[:, 3] - r0[:, 3]) ** 2
                # ignore the yaw error
            )
            # input_traj cost is sum of squares of motor speeds for 4 individual motors
            + np.linalg.norm(input_traj[i]) * (1 / horizon)
            # + np.linalg.norm(input_traj[i]) ** 2  # Removed 0.1 multiplier
        )
        # print("cost for the input_traj: ", np.linalg.norm(input_traj[i]))
    # print("rho: ", rho)
    # print(
    #     "cost for the deviation from reference: ",
    #     np.linalg.norm(act[:, :3] - r0[:, :3], axis=1) ** 2
    #     + angle_wrap(act[:, 3] - r0[:, 3]) ** 2,
    # )

    xcost.reverse()
    cost = []
    for i in range(num_traj):
        tot = list(accumulate(xcost[i], lambda x, y: x * gamma + y))
        cost.append(np.log(tot[-1]))
        # cost.append(tot[-1])
    cost.reverse()
    return np.vstack(cost)


def angle_wrap(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def main():
    horizon = 502
    rho = 0.01

    rhos = [0, 1, 5, 10, 20, 50, 100]
    gamma = 1
    # Load bag
    # sim_data = load_bag('/home/anusha/2022-09-27-11-49-40.bag')
    # sim_data = load_bag("/home/anusha/2023-02-27-13-35-15.bag")
    # sim_data = load_bag("/home/anusha/dragonfly1-2023-04-12-12-18-27.bag")
    ### Load the csv file here with header
    sim_data = np.loadtxt(
        "/home/mrsl_guest/rotorpy/rotorpy/rotorpy/data_out/sim_airdrag_yawmixed_1500.csv",
        delimiter=",",
        skiprows=1,
    )

    # no need times
    ref_traj, actual_traj, input_traj, cost_traj, times = compute_traj(
        sim_data=sim_data, rho=rho, horizon=horizon
    )

    with open(
        # r"/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/params.yaml"
        r"/home/mrsl_guest/rotorpy/rotorpy/learning/params.yaml"
    ) as f:
        yaml_data = yaml.load(f, Loader=yaml.RoundTripLoader)

    num_hidden = yaml_data["num_hidden"]
    batch_size = yaml_data["batch_size"]
    learning_rate = yaml_data["learning_rate"]
    num_epochs = yaml_data["num_epochs"]
    model_save = yaml_data["save_path"] + str(rho)
    # Construct augmented states

    cost_traj = cost_traj.ravel()
    # # exp_log_cost_traj into cost_traj
    # cost_traj = np.exp(cost_traj)

    # print("Costs", cost_traj)
    print("Costs shape", cost_traj.shape)

    # scatter plot for cost_traj vs index for fixed yaw
    plt.figure()
    plt.scatter(range(len(cost_traj)), np.exp(cost_traj), color="b", label="Cost")
    plt.xlabel("Trajectory Index")
    plt.ylabel("Cost")
    plt.legend()
    plt.title("Cost vs Trajectory Index for fixed yaw")
    # plt.savefig("./plots/cost" + str(rho) + ".png")
    plt.show()

    # plot cost of trajectories with different radii ranging from 1 to 5.5 and varying yaw angles from 0 to 2pi in 3d
    # # 3d scatter plot for cost_traj vs radius vs yaw for fixed yaw
    # radius = np.linspace(2, 5.5, 10)
    # yaw = np.linspace(0, 2 * np.pi, 10)
    # radius, yaw = np.meshgrid(radius, yaw)
    # cost_traj = cost_traj.reshape(10, 10)

    # # scatter plot for cost_traj vs radius for flexible yaw
    # radius = np.linspace(2, 5.5, 10)
    # plt.figure()
    # plt.scatter(radius, cost_traj, color="b", label="Cost")
    # plt.xlabel("Radius")
    # plt.ylabel("Cost")
    # plt.legend()
    # plt.title("Cost vs Radius for flexible yaw")
    # # plt.savefig("./plots/cost" + str(rho) + ".png")
    # plt.show()

    num_traj = int(len(ref_traj) / horizon)
    print("len(ref_traj): ", len(ref_traj))
    print("num_traj: ", num_traj)

    # Create augmented state

    aug_state = []
    for i in range(num_traj):
        r0 = ref_traj[i * horizon : (i + 1) * horizon, :]
        act = actual_traj[i * horizon : (i + 1) * horizon, :]
        aug_state.append(np.append(act[0, :], r0))
        # act[0, :] is the first row of actual_traj, which is the initial state, dimension is 4
        # r0 is the reference trajectory, dimension is 502*4

    aug_state = np.array(aug_state)
    print(aug_state.shape)

    Tstart = 0
    Tend = aug_state.shape[0]

    train_dataset = TrajDataset(
        aug_state[Tstart : Tend - 1, :].astype("float64"),
        input_traj[Tstart : Tend - 1, :].astype("float64"),
        cost_traj[Tstart : Tend - 1, None].astype("float64"),
        aug_state[Tstart + 1 : Tend, :].astype("float64"),
    )

    # Split the dataset into training and testing subsets
    train_data, test_data = train_test_split(
        train_dataset,
        test_size=0.2,  # Specify the proportion of the dataset to use for testing (e.g., 0.2 for 20%)
        random_state=42,  # Set a random seed for reproducibility
    )

    print("Training data length: ", len(train_data))
    print("Testing data length: ", len(test_data))

    p = aug_state.shape[1]
    q = 4

    print(aug_state.shape)

    model = MLP(num_hidden=num_hidden, num_outputs=1)
    # Printing the model shows its attributes
    print(model)

    rng = jax.random.PRNGKey(427)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (batch_size, p))  # Batch size 64, input size p
    # Initialize the model
    params = model.init(init_rng, inp)

    optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)
    # optimizer = optax.adam(learning_rate=learning_rate, b1=0.9, b2=0.999, eps=1e-08)

    model_state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    train_data_loader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate
    )
    trained_model_state = train_model(
        model_state, train_data_loader, num_epochs=num_epochs
    )
    """
    # Train on 2nd dataset
    sim_data = load_bag("/home/anusha/dragonfly2-2023-04-12-12-18-27.bag")

    ref_traj, actual_traj, input_traj, cost_traj, times = compute_traj(
        sim_data, "dragonfly2", "/home/anusha/min_jerk_times.pkl", rho
    )
    sim_data.close()

    # Construct augmented states

    cost_traj = cost_traj.ravel()

    print("Costs", cost_traj)

    num_traj = int(len(ref_traj) / horizon)

    # Create augmented state

    aug_state = []
    for i in range(num_traj):
        r0 = ref_traj[i * horizon : (i + 1) * horizon, :]
        act = actual_traj[i * horizon : (i + 1) * horizon, :]
        aug_state.append(np.append(act[0, :], r0))

    aug_state = np.array(aug_state)
    print(aug_state.shape)

    Tstart = 0
    Tend = aug_state.shape[0]

    train_dataset = TrajDataset(
        aug_state[Tstart : Tend - 1, :].astype("float64"),
        input_traj[Tstart : Tend - 1, :].astype("float64"),
        cost_traj[Tstart : Tend - 1, None].astype("float64"),
        aug_state[Tstart + 1 : Tend, :].astype("float64"),
    )

    p = aug_state.shape[1]
    q = 4

    print(aug_state.shape)

    train_data_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate
    )
    trained_model_state = train_model(
        trained_model_state, train_data_loader, num_epochs=num_epochs
    )
    """

    # Evaluation of trained network

    eval_model(trained_model_state, train_data_loader, batch_size)

    trained_model = model.bind(trained_model_state.params)
    # TODO: save checkpoint
    save_checkpoint(trained_model_state, model_save, 7)

    # Save plot on entire test dataset
    out = []
    true = []
    for batch in train_data_loader:
        data_input, _, cost, _ = batch
        out.append(trained_model(data_input))
        true.append(cost)
        # print("Cost", cost)
        # print("Predicted", trained_model(data_input))

    print("out's shape", out[0].shape)
    out = np.vstack(out)
    true = np.vstack(true)

    # scatter plot
    plt.figure()
    plt.scatter(range(len(out)), out.ravel(), color="b", label="Predictions")
    plt.scatter(range(len(true)), true.ravel(), color="r", label="Actual")
    plt.xlabel("Trajectory Index")
    plt.ylabel("Cost")
    plt.legend()
    plt.title("Predicted vs Actual - Training Dataset")
    # plt.savefig("./plots/inference"+str(rho)+".png")
    plt.show()
    # plt.figure()
    # plt.scatter(range(len(out)), out.ravel(), color="b", label="Predictions")
    # plt.scatter(range(len(true)), true.ravel(), color="r", label="Actual")
    # plt.xlabel("Trajectory Index")
    # plt.ylabel("Cost")
    # plt.legend()
    # plt.title("Predicted vs Actual - Training Dataset")
    # # Set the y-axis range from 0 to 12
    # plt.ylim(0, 12)
    # # plt.savefig("./plots/inference"+str(rho)+".png")
    # plt.show()

    # scatter plot
    # plt.figure()
    # plt.scatter(range(len(true)), true.ravel(), color="r", label="Actual")
    # plt.xlabel("Trajectory Index")
    # plt.ylabel("Cost")
    # plt.legend()
    # plt.title("Actual - Training Dataset")
    # # plt.savefig("./plots/inference"+str(rho)+".png")
    # plt.show()
    plt.figure()
    plt.scatter(range(len(true)), true.ravel(), color="r", label="Actual")
    plt.xlabel("Trajectory Index")
    plt.ylabel("Cost")
    plt.legend()
    plt.title("Actual - Training Dataset")
    # Set the y-axis range from 0 to 500
    plt.ylim(0, 800)
    plt.show()

    # line plot
    plt.figure()
    plt.xlabel("Trajectory Index")
    plt.ylabel("Cost")
    plt.plot(out.ravel(), "b-", label="Predictions")
    plt.plot(true.ravel(), "r--", label="Actual")
    plt.legend()
    plt.title("Predicted vs Actual - Training Dataset")
    plt.show()

    # Two boxplots
    plt.figure()
    plt.boxplot([out.ravel(), true.ravel()], labels=["Predictions", "Actual"])
    plt.xlabel("Trajectory Index")
    plt.ylabel("Cost")
    plt.title("Predicted vs Actual - Train Dataset")
    plt.show()
    """
    # test on 2nd dataset
    # inf_data = load_bag("/home/anusha/dragonfly2-2023-04-12-12-18-27.bag")
    # ref_traj, actual_traj, input_traj, cost_traj, times = compute_traj(
    #     inf_data, "dragonfly2", "/home/anusha/min_jerk_times.pkl", rho
    # )
    # inf_data.close()

    ### Load the csv file here with header
    inf_data = np.loadtxt(
        "/home/mrsl_guest/Desktop/dragonfly2.csv", delimiter=",", skiprows=1
    )
    # no need times
    ref_traj, actual_traj, input_traj, cost_traj, times = compute_traj(
        sim_data, 1, horizon, False
    )

    # # Construct augmented states
    # horizon = 300
    # gamma = 1

    # idx = [0, 1, 2, 12]

    cost_traj = cost_traj.ravel()

    num_traj = int(len(ref_traj) / horizon)

    # Create augmented state

    aug_state = []
    for i in range(num_traj):
        r0 = ref_traj[i * horizon : (i + 1) * horizon, :]
        act = actual_traj[i * horizon : (i + 1) * horizon, :]
        aug_state.append(np.append(act[0, :], r0))

    aug_state = np.array(aug_state)
    print(aug_state.shape)

    Tstart = 0
    Tend = aug_state.shape[0]

    test_dataset = TrajDataset(
        aug_state[Tstart : Tend - 1, :].astype("float64"),
        input_traj[Tstart : Tend - 1, :].astype("float64"),
        cost_traj[Tstart : Tend - 1, None].astype("float64"),
        aug_state[Tstart + 1 : Tend, :].astype("float64"),
    )

    test_data_loader = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate
    )
    """
    # Evaluation of test and train dataset

    test_data_loader = data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate
    )

    eval_model(trained_model_state, test_data_loader, batch_size)

    # Save plot on entire test dataset
    out = []
    true = []
    for batch in test_data_loader:
        data_input, _, cost, _ = batch
        out.append(trained_model(data_input))
        true.append(cost)

    out = np.vstack(out)
    true = np.vstack(true)

    print(out.shape)
    print(true.shape)

    # scatter plot
    plt.figure()
    plt.scatter(range(len(out)), out.ravel(), color="b", label="Predictions")
    plt.scatter(range(len(true)), true.ravel(), color="r", label="Actual")
    plt.xlabel("Trajectory Index")
    plt.ylabel("Cost")
    # plt.plot(out.ravel(), "b-", label="Predictions")
    # plt.plot(true.ravel(), "r--", label="Actual")
    plt.legend()
    plt.title("Predicted vs Actual - Test Dataset")
    # plt.savefig("./plots/inference"+str(rho)+".png")
    plt.show()

    # # line plot
    # plt.figure()
    # plt.xlabel("Trajectory Index")
    # plt.ylabel("Cost")
    # plt.plot(out.ravel(), "b-", label="Predictions")
    # plt.plot(true.ravel(), "r--", label="Actual")
    # plt.legend()
    # plt.title("Predicted vs Actual - Test Dataset")
    # # plt.savefig("./plots/inference"+str(rho)+".png")
    # plt.show()

    # Two boxplots
    plt.figure()
    plt.boxplot([out.ravel(), true.ravel()], labels=["Predictions", "Actual"])
    plt.xlabel("Trajectory Index")
    plt.ylabel("Cost")
    plt.title("Predicted vs Actual - Test Dataset")
    plt.show()

    # eval_model(trained_model_state, test_data_loader, batch_size)


if __name__ == "__main__":
    main()
