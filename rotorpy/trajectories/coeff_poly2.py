import numpy as np


class CoeffPoly(object):
    def __init__(self, Tref, nn_coeffs, segments, ts):
        self.nn_coeffs = nn_coeffs
        self.Tref = Tref
        self.segments = segments
        self.ts = ts

        (
            self.pos,
            self.vel,
            self.acc,
            self.jerk,
            self.yaw_ref,
            self.yawdot_ref,
        ) = self.compute_pos_vel_acc(self.Tref, self.nn_coeffs, self.segments, self.ts)

    def compute_coeff_deriv(self, coeff, n, segments):
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

    def compute_pos_vel_acc(self, Tref, nn_coeffs, segments, ts):
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

        x_ref = [np.poly1d(coeff_x[i, :]) for i in range(segments)]
        x_ref = np.vstack(self.sampler(x_ref, Tref, ts)).flatten()

        y_ref = [np.poly1d(coeff_y[i, :]) for i in range(segments)]
        y_ref = np.vstack(self.sampler(y_ref, Tref, ts)).flatten()

        z_ref = [np.poly1d(coeff_z[i, :]) for i in range(segments)]
        z_ref = np.vstack(self.sampler(z_ref, Tref, ts)).flatten()
        pos.append([x_ref, y_ref, z_ref])

        dot_x = self.compute_coeff_deriv(coeff_x, 1, segments)
        xdot_ref = [np.poly1d(dot_x[i, :]) for i in range(segments)]
        xdot_ref = np.vstack(self.sampler(xdot_ref, Tref, ts)).flatten()

        dot_y = self.compute_coeff_deriv(coeff_y, 1, segments)
        ydot_ref = [np.poly1d(dot_y[i, :]) for i in range(segments)]
        ydot_ref = np.vstack(self.sampler(ydot_ref, Tref, ts)).flatten()

        dot_z = self.compute_coeff_deriv(coeff_z, 1, segments)
        zdot_ref = [np.poly1d(dot_z[i, :]) for i in range(segments)]
        zdot_ref = np.vstack(self.sampler(zdot_ref, Tref, ts)).flatten()
        vel.append([xdot_ref, ydot_ref, zdot_ref])

        ddot_x = self.compute_coeff_deriv(coeff_x, 2, segments)
        xddot_ref = [np.poly1d(ddot_x[i, :]) for i in range(segments)]
        xddot_ref = np.vstack(self.sampler(xddot_ref, Tref, ts)).flatten()

        ddot_y = self.compute_coeff_deriv(coeff_y, 2, segments)
        yddot_ref = [np.poly1d(ddot_y[i, :]) for i in range(segments)]
        yddot_ref = np.vstack(self.sampler(yddot_ref, Tref, ts)).flatten()

        ddot_z = self.compute_coeff_deriv(coeff_z, 2, segments)
        zddot_ref = [np.poly1d(ddot_z[i, :]) for i in range(segments)]
        zddot_ref = np.vstack(self.sampler(zddot_ref, Tref, ts)).flatten()
        acc.append([xddot_ref, yddot_ref, zddot_ref])

        dddot_x = self.compute_coeff_deriv(coeff_x, 3, segments)
        xdddot_ref = [np.poly1d(dddot_x[i, :]) for i in range(segments)]
        xdddot_ref = np.vstack(self.sampler(xdddot_ref, Tref, ts)).flatten()

        dddot_y = self.compute_coeff_deriv(coeff_y, 3, segments)
        ydddot_ref = [np.poly1d(dddot_y[i, :]) for i in range(segments)]
        ydddot_ref = np.vstack(self.sampler(ydddot_ref, Tref, ts)).flatten()

        dddot_z = self.compute_coeff_deriv(coeff_z, 3, segments)
        zdddot_ref = [np.poly1d(dddot_z[i, :]) for i in range(segments)]
        zdddot_ref = np.vstack(self.sampler(zdddot_ref, Tref, ts)).flatten()
        jerk.append([xdddot_ref, ydddot_ref, zdddot_ref])

        yaw_ref = [np.poly1d(coeff_yaw[i, :]) for i in range(segments)]
        yaw_ref = np.vstack(self.sampler(yaw_ref, Tref, ts)).flatten()

        dot_yaw = self.compute_coeff_deriv(coeff_yaw, 1, segments)
        yawdot_ref = [np.poly1d(dot_yaw[i, :]) for i in range(segments)]
        yawdot_ref = np.vstack(self.sampler(yawdot_ref, Tref, ts)).flatten()

        return (
            np.vstack(pos),
            np.vstack(vel),
            np.vstack(acc),
            np.vstack(jerk),
            yaw_ref,
            yawdot_ref,
        )

    def update(self, t):
        idx = int(np.floor(t / self.Tref))
        t_rel = t - idx * self.Tref

        pos = self.pos[idx, :]
        vel = self.vel[idx, :]
        acc = self.acc[idx, :]
        jerk = self.jerk[idx, :]
        yaw_ref = self.yaw_ref[idx]
        yawdot_ref = self.yawdot_ref[idx]

        flat_output = {
            "x": pos,
            "x_dot": vel,
            "x_ddot": acc,
            "x_dddot": jerk,
            "yaw": yaw_ref,
            "yaw_dot": yawdot_ref,
        }
        return flat_output[]
