import numpy as np
import matplotlib.pyplot as plt
from learning.trajgen import quadratic


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


# get waypoints from a circle which is 3 meters in radius
def get_circle_waypoints(radius, height, num_points, yaw):
    waypoints = []
    for i in range(num_points):
        theta = 2 * np.pi * i / num_points
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = height
        yaw = yaw
        waypoints.append([x, y, z, yaw])
    return waypoints


radius = 3
waypoints = get_circle_waypoints(radius, 2, 40, 0)
waypoints = np.array(waypoints)
print("waypoint's", waypoints.shape)
# visualize the waypoints
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], "o")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# get coefficients
num_data = 502
p = 4  # num_dimensions
order = 5
duration = 5
ts = np.linspace(0, duration, len(waypoints))
# print("waypoints[i]'s shape", waypoints[i].shape)
print("ts", ts)
# get min_jerk_coeffs for each traj
_, min_jerk_coeffs = quadratic.generate(waypoints, ts, order, num_data, p, None, 0)
print("min_jerk_coeffs's shape", np.array(min_jerk_coeffs).shape)

# get pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_ddot
pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_ddot = compute_pos_vel_acc(
    num_data, min_jerk_coeffs, len(waypoints) - 1, ts
)
print("pos's shape", pos.shape)

# visualize the trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(pos[0], pos[1], pos[2], "o")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# save pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_ddot
path = "/home/mrsl_guest/rotorpy/rotorpy/rotorpy/data_out/pos_min_jerk_0903.csv"
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
