"""
In order to get the body rates for the initial state
"""
import numpy as np


def rotationMatrixToQuaternion1(m):
    # q0 = qw
    t = np.matrix.trace(m)
    q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if t > 0:
        t = np.sqrt(t + 1)
        q[3] = 0.5 * t
        t = 0.5 / t
        q[0] = (m[2, 1] - m[1, 2]) * t
        q[1] = (m[0, 2] - m[2, 0]) * t
        q[2] = (m[1, 0] - m[0, 1]) * t

    else:
        i = 0
        if m[1, 1] > m[0, 0]:
            i = 1
        if m[2, 2] > m[i, i]:
            i = 2
        j = (i + 1) % 3
        k = (j + 1) % 3

        t = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (m[k, j] - m[j, k]) * t
        q[j] = (m[j, i] + m[i, j]) * t
        q[k] = (m[k, i] + m[i, k]) * t

    return q


def get_T(acc, g=9.81):
    """
    Function to compute the intermediate T vector
    :param acc:
    :param g:
    :return:
    """
    return [np.array([x[0], x[1], x[2] + g]) for x in acc]


def get_yc(yaw):
    """
    Function to compute intermediate yc vector
        :return:
    """
    Tref = len(yaw)
    temp = np.stack([-np.sin(yaw), np.cos(yaw), np.zeros(Tref)]).flatten()
    temp = temp.reshape((3, Tref))
    return temp.T


def get_xb(yc, zb):
    """
    Function to compute intermediate xb vector
        :return:
    """
    x = []
    for y, z in zip(yc, zb):
        x.append(np.cross(y.flatten(), z.flatten()))
    return np.vstack(x) / np.linalg.norm(np.vstack(x))


def get_yb(zb, xb):
    """
    Function to compute intermediate yb vector
        :return:
    """
    r = []
    for z, x in zip(zb, xb):
        r.append(np.cross(z.flatten(), x.flatten()))
    return np.vstack(r)


def get_zb(T):
    """
    Function to compute intermediate zb vector
        :param T:
        :return:
    """
    return T / np.linalg.norm(T, axis=0)


def compute_acc(cur_vel, prev_vel, time):
    """
    Assume velocity to be linear for each segment and compute acc
    :return:
    """
    return (cur_vel - prev_vel) / time


def compute_jerk(cur_acc, prev_acc, time):
    """
    Assume velocity to be linear for each segment and compute acc
    :return:
    """
    return (cur_acc - prev_acc) / time


def compute_yaw_dot(cur_yaw, prev_yaw, time):
    """
    Assume velocity to be linear for each segment and compute acc
    :return:
    """
    return (cur_yaw - prev_yaw) / time


def get_hw(x, x_dot, yaw, time):
    acc = compute_acc(x_dot, np.zeros(3), time)
    jerk = compute_jerk(acc, np.zeros(3), time)
    zb = get_zb(get_T(acc))
    # sigma: x, yaw
    sigma = np.vstack([x, yaw]).T
    return (
        (jerk - np.dot(zb, np.dot(zb.T, jerk)))
        / np.linalg.norm(sigma, axis=1).reshape(-1, 1)
    ).T


def get_wx(yb, hw):
    return -np.dot(yb, hw)


def get_wy(xb, hw):
    return np.dot(xb, hw)


def get_wz(zb, hw, yaw_dot):
    return np.dot(zb, np.dot(yaw_dot, hw))


def get_bodyrate(x, x_dot, yaw, time):
    """
    Function to compute the body rates
    :param x:
    :param x_dot:
    :param yaw:
    :param time:
    :return:
    """
    acc = compute_acc(x_dot, np.zeros(3), time)
    yaw_dot = compute_yaw_dot(yaw, np.zeros(1), time)
    zb = get_zb(get_T(acc))
    yc = get_yc(yaw)
    xb = get_xb(yc, zb)
    yb = get_yb(zb, xb)
    hw = get_hw(x, x_dot, yaw, time)
    wx = get_wx(yb, hw)
    wy = get_wy(xb, hw)
    wz = get_wz(zb, hw, yaw_dot)
    return np.vstack([wx, wy, wz]).T
