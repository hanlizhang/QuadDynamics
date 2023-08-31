import numpy as np


class CoeffPoly(object):
    def __init__(self, coefficients):
        """
        It takes in coefficients of polynomial trajectories from the network and produces polynomials

        Inputs:
            coefficients: numpy array containing the coefficients of the polynomial trajectories.
                         It should have shape (num_dimensions, num_segments, polynomial_order+1).
                         For example, coefficients[i, j, k] represents the coefficient of the k-th order term
                         for the j-th dimension of the i-th segment's polynomial.
        """
        self.coefficients = coefficients
        (
            self.num_dimensions,
            self.num_segments,
            self.polynomial_order_plus_1,
        ) = coefficients.shape  # 4,9,6

        # construct the polynomial
        self.x_poly = np.zeros(
            self.num_dimensions, self.num_segments, self.polynomial_order_plus_1
        )
        self.y_poly = np.zeros(
            self.num_dimensions, self.num_segments, self.polynomial_order_plus_1
        )

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
        x = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0
        yaw_ddot = 0

        order = self.polynomial_order_plus_1 - 1

        for segment in range(self.num_segments):
            segment_coefficients = self.coefficients[segment]

            # from the coefficients, we can get the polynomial for each dimension

            # for dim in range(self.num_dimensions):
            #     dim_coefficients = segment_coefficients[dim]
            #     order = self.polynomial_order_plus_1 - 1

            #     position[dim] = np.polyval(dim_coefficients, t)
            #     velocity[dim] = np.polyval(np.polyder(dim_coefficients), t)
            #     acceleration[dim] = np.polyval(
            #         np.polyder(np.polyder(dim_coefficients)), t
            #     )
            #     jerk[dim] = np.polyval(
            #         np.polyder(np.polyder(np.polyder(dim_coefficients))), t
            #     )
            #     if order >= 4:
            #         snap[dim] = np.polyval(
            #             np.polyder(
            #                 np.polyder(np.polyder(np.polyder(dim_coefficients)))
            #             ),
            #             t,
            #         )

            # # flat_output[f"x_{segment}"] = position
            # # flat_output[f"x_dot_{segment}"] = velocity
            # # flat_output[f"x_ddot_{segment}"] = acceleration
            # # flat_output[f"x_dddot_{segment}"] = jerk
            # # if order >= 4:
            # #     flat_output[f"x_ddddot_{segment}"] = snap
            # # combine all the segments into one
            # position = np.concatenate(position)
            # velocity = np.concatenate(velocity)
            # acceleration = np.concatenate(acceleration)
            # jerk = np.concatenate(jerk)
            # if order >= 4:
            #     snap = np.concatenate(snap)

            # x = position
            # x_dot = velocity
            # x_ddot = acceleration
            # x_dddot = jerk
            # if order >= 4:
            #     x_ddddot = snap

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
