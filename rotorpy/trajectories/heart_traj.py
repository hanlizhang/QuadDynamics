import numpy as np


class HeartTrajectory(object):
    def __init__(self, center=np.array([0, 0, 3]), scale=1.0, freq=0.2, yaw_bool=False):
        """
        Constructor for the HeartTrajectory object.

        Inputs:
            center, the center of the heart shape (m)
            scale, a scaling factor for the size of the heart (default is 1.0)
            freq, the frequency with which the heart shape is completed (Hz)
            yaw_bool, determines if yaw motion is desired
        """

        # Call the constructor of the parent class
        super().__init__()

        self.center = center
        self.cx, self.cy, self.cz = center[0], center[1], center[2]
        self.scale = scale
        self.freq = freq
        self.omega = 2 * np.pi * self.freq
        self.yaw_bool = yaw_bool

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        a = self.scale
        x = np.array(
            [
                self.cx + a * (16 * np.sin(self.omega * t) ** 3),
                self.cy
                + a
                * (
                    13 * np.cos(self.omega * t)
                    - 5 * np.cos(2 * self.omega * t)
                    - 2 * np.cos(3 * self.omega * t)
                    - np.cos(4 * self.omega * t)
                ),
                self.cz,
            ]
        )

        x_dot = np.array(
            [
                a
                * 48
                * np.pi
                * self.freq
                * np.cos(self.omega * t) ** 2
                * np.sin(self.omega * t),
                -a * 13 * np.pi * self.freq * np.sin(self.omega * t)
                - a * 20 * np.pi * self.freq * np.sin(2 * self.omega * t)
                - a * 6 * np.pi * self.freq * np.sin(3 * self.omega * t)
                - a * 4 * np.pi * self.freq * np.sin(4 * self.omega * t),
                0,
            ]
        )

        x_ddot = np.array(
            [
                -a
                * 96
                * np.pi**2
                * self.freq**2
                * np.cos(self.omega * t)
                * np.sin(self.omega * t)
                + a * 48 * np.pi**2 * self.freq**2 * np.cos(self.omega * t) ** 3,
                -a * 13 * np.pi**2 * self.freq**2 * np.cos(self.omega * t)
                - a * 40 * np.pi**2 * self.freq**2 * np.cos(2 * self.omega * t)
                - a * 18 * np.pi**2 * self.freq**2 * np.cos(3 * self.omega * t)
                - a * 16 * np.pi**2 * self.freq**2 * np.cos(4 * self.omega * t),
                0,
            ]
        )

        x_dddot = np.array(
            [
                -a
                * 144
                * np.pi**3
                * self.freq**3
                * np.cos(self.omega * t) ** 2
                * np.sin(self.omega * t)
                + a * 144 * np.pi**3 * self.freq**3 * np.cos(self.omega * t) ** 4,
                a * 13 * np.pi**3 * self.freq**3 * np.sin(self.omega * t)
                - a * 40 * np.pi**3 * self.freq**3 * np.sin(2 * self.omega * t)
                - a * 54 * np.pi**3 * self.freq**3 * np.sin(3 * self.omega * t)
                - a * 64 * np.pi**3 * self.freq**3 * np.sin(4 * self.omega * t),
                0,
            ]
        )

        x_ddddot = np.array(
            [
                a
                * 672
                * np.pi**4
                * self.freq**4
                * np.cos(self.omega * t)
                * np.sin(self.omega * t)
                - a * 288 * np.pi**4 * self.freq**4 * np.cos(self.omega * t) ** 3,
                a * 13 * np.pi**4 * self.freq**4 * np.cos(self.omega * t)
                + a * 80 * np.pi**4 * self.freq**4 * np.cos(2 * self.omega * t)
                + a * 162 * np.pi**4 * self.freq**4 * np.cos(3 * self.omega * t)
                + a * 256 * np.pi**4 * self.freq**4 * np.cos(4 * self.omega * t),
                0,
            ]
        )

        # Calculate yaw angle and its derivative if yaw_bool is True
        if self.yaw_bool:
            yaw = np.pi / 4 * np.sin(np.pi * t)
            yaw_dot = np.pi**2 / 4 * np.cos(np.pi * t)
            yaw_ddot = -np.pi**3 / 4 * np.sin(np.pi * t)
        else:
            yaw = 0
            yaw_dot = 0
            yaw_ddot = 0

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
