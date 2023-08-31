import numpy as np


class PosTraj(object):
    def __init__(self, pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_ddot):
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.jerk = jerk
        self.snap = snap
        self.yaw = yaw
        self.yaw_dot = yaw_dot
        self.yaw_ddot = yaw_ddot

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs:
            t: time in seconds.

        Outputs:
            flat_output: a dictionary describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                ...
        """
        freq = 100
        # round to nearest time step
        ind = int(np.round(t * freq))
        # print("ind", ind)
        # print("self.pos's shape", self.pos.shape)

        x = self.pos[ind]
        x_dot = self.vel[ind]
        x_ddot = self.acc[ind]
        x_dddot = self.jerk[ind]
        x_ddddot = self.snap[ind]
        yaw = self.yaw[ind]
        yaw_dot = self.yaw_dot[ind]
        yaw_ddot = self.yaw_ddot[ind]

        flat_output = {
            "x": x,
            "x_dot": x_dot,
            "x_ddot": x_ddot,
            "x_dddot": x_dddot,
            "x_ddddot": x_ddddot,
            "yaw": yaw,
            "yaw_dot": yaw_dot,
            "yaw_ddot": yaw_ddot,
        }

        return flat_output
