import numpy as np
import scipy.linalg
from copy import deepcopy
from threading import Lock


class UKFException(Exception):
    """Raise for errors in the UKF, usually due to bad inputs"""


class UKF:
    def __init__(self, num_states, process_noise, initial_state, initial_covar, alpha, k, beta, iterate_function):
        """
        Initializes the unscented kalman filter
        """
        self.n_dim = int(num_states)
        self.n_sig = 1 + num_states * 2
        self.q = process_noise
        self.x = initial_state
        self.p = initial_covar
        self.beta = beta
        self.alpha = alpha
        self.k = k
        self.iterate = iterate_function

        # ------------------------------------------------------------------
        # TODO: Calculate self.lambd
        # Formula: alpha^2 * (n_dim + k) - n_dim
        # ------------------------------------------------------------------
        self.lambd = self.alpha**2 * (self.n_dim + self.k) - self.n_dim

        self.covar_weights = np.zeros(self.n_sig)
        self.mean_weights = np.zeros(self.n_sig)

        # ------------------------------------------------------------------
        # TODO: Initialize Weights
        # ------------------------------------------------------------------
        self.mean_weights[0] = self.lambd / (self.n_dim + self.lambd)
        self.covar_weights[0] = (
            self.lambd / (self.n_dim + self.lambd)
            + (1 - self.alpha**2 + self.beta)
        )

        for i in range(1, self.n_sig):
            self.mean_weights[i] = 1.0 / (2 * (self.n_dim + self.lambd))
            self.covar_weights[i] = 1.0 / (2 * (self.n_dim + self.lambd))

        # ------------------------------------------------------------------
        # TODO: Generate Initial Sigmas
        # ------------------------------------------------------------------
        self.sigmas = self.__get_sigmas()

        self.lock = Lock()

    def __get_sigmas(self):
        """generates sigma points"""
        ret = np.zeros((self.n_sig, self.n_dim))

        # ------------------------------------------------------------------
        # TODO: Generate Sigma Points
        # ------------------------------------------------------------------
        sqrt_matrix = scipy.linalg.sqrtm(
            (self.n_dim + self.lambd) * self.p
        )

        ret[0] = self.x

        for i in range(self.n_dim):
            ret[i + 1] = self.x + sqrt_matrix[:, i]
            ret[i + 1 + self.n_dim] = self.x - sqrt_matrix[:, i]

        return ret.T

    def update(self, states, data, r_matrix):
        """
        performs a measurement update
        """

        self.lock.acquire()

        num_states = len(states)
        data = np.asarray(data)

        # ------------------------------------------------------------------
        # TODO: Create Measurement Sigmas (y) and Mean (y_mean)
        # ------------------------------------------------------------------
        y = self.sigmas[states, :]
        y_mean = np.zeros(num_states)

        for i in range(self.n_sig):
            y_mean += self.mean_weights[i] * y[:, i]

        # ------------------------------------------------------------------
        # TODO: Calculate Differences
        # ------------------------------------------------------------------
        y_diff = y - y_mean[:, None]
        x_diff = self.sigmas - self.x[:, None]

        # ------------------------------------------------------------------
        # TODO: Calculate Measurement Covariance (p_yy)
        # ------------------------------------------------------------------
        p_yy = np.zeros((num_states, num_states))
        for i in range(self.n_sig):
            p_yy += self.covar_weights[i] * np.outer(y_diff[:, i], y_diff[:, i])

        p_yy += r_matrix

        # ------------------------------------------------------------------
        # TODO: Calculate Cross Covariance (p_xy)
        # ------------------------------------------------------------------
        p_xy = np.zeros((self.n_dim, num_states))
        for i in range(self.n_sig):
            p_xy += self.covar_weights[i] * np.outer(x_diff[:, i], y_diff[:, i])

        # ------------------------------------------------------------------
        # TODO: Kalman Gain and Update
        # ------------------------------------------------------------------
        K = p_xy @ np.linalg.inv(p_yy)
        self.x = self.x + K @ (data - y_mean)
        self.p = self.p - K @ p_yy @ K.T
        self.sigmas = self.__get_sigmas()

        self.lock.release()

    def predict(self, timestep, inputs=[]):
        """
        performs a prediction step
        """

        self.lock.acquire()

        # ------------------------------------------------------------------
        # TODO: Propagate Sigma Points
        # ------------------------------------------------------------------
        sigmas_out = np.zeros_like(self.sigmas)
        for i in range(self.n_sig):
            sigmas_out[:, i] = self.iterate(
                self.sigmas[:, i], timestep, inputs
            )

        # ------------------------------------------------------------------
        # TODO: Calculate Predicted Mean (x_out)
        # ------------------------------------------------------------------
        x_out = np.zeros(self.n_dim)
        for i in range(self.n_sig):
            x_out += self.mean_weights[i] * sigmas_out[:, i]

        # ------------------------------------------------------------------
        # TODO: Calculate Predicted Covariance (p_out)
        # ------------------------------------------------------------------
        p_out = np.zeros((self.n_dim, self.n_dim))
        for i in range(self.n_sig):
            diff = sigmas_out[:, i] - x_out
            p_out += self.covar_weights[i] * np.outer(diff, diff)

        # ------------------------------------------------------------------
        # TODO: Add Process Noise
        # ------------------------------------------------------------------
        p_out += timestep * self.q

        # ------------------------------------------------------------------
        # TODO: Update State
        # ------------------------------------------------------------------
        self.sigmas = sigmas_out
        self.x = x_out
        self.p = p_out

        self.lock.release()

    def get_state(self, index=-1):
        if index >= 0:
            return self.x[index]
        else:
            return self.x

    def get_covar(self):
        return self.p

    def set_state(self, value, index=-1):
        with self.lock:
            if index != -1:
                self.x[index] = value
            else:
                self.x = value

    def reset(self, state, covar):
        with self.lock:
            self.x = state
            self.p = covar
