import numpy as np
import casadi as ca
from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints

from robust_smc.data import peaks


def dem(x, y, num_frequencies=4):
    """
    Synthetic Digital Elevation Model (DEM) map
    :param x: x coordinates Nx1 numpy array
    :param y: y coordinates Nx1 numpy array
    :return: z coordinates Nx1 numpy array
    """
    a = np.array([300, 80, 60, 40, 20, 10])[:num_frequencies]  # 1x6
    omega = np.array([5, 10, 20, 30, 80, 150])[:num_frequencies]  # 1x6
    omega_bar = np.array([4, 10, 20, 40, 90, 150])[:num_frequencies]  # 1x6
    q = 3 / (2.96 * 1e4)
    # q = 0.5
    peak = peaks(q * x, q * y)
    if type(x) is np.float64:
        peak += np.sum(a * np.sin(omega * q * x) * np.cos(omega_bar * q * y))
    else:
        for i in range(num_frequencies):
            peak += a[i] * np.sin(omega[i] * q * x) * np.cos(omega_bar[i] * q * y)

    return peak


class UKF:
    def __init__(self, data, transition_matrix, transition_cov, observation_cov, m_0, P_0):

        self.data = data
        self.transition_matrix = transition_matrix
        self.transition_cov = transition_cov

        self.y_dim = data.shape[1]
        self.observation_cov = observation_cov * np.eye(self.y_dim)
        self.m_0 = m_0
        self.P_0 = P_0
        self.x_dim = transition_cov.shape[0]
        self.time_step = 0.1

    def fx(self, x, dt=0):
        return self.transition_matrix @ x

    def hx(self, X):
        X0 = np.array([-7.5 * 1e3, 5.0 * 1e3, 1.1 * 1e3, 88.15, -60.53, 0.0])
        num_frequencies = 6
        height = X[2] - dem(X[0], X[1], num_frequencies)
        if type(X) is np.ndarray:
            distance = np.sqrt(np.sum((X[:2] - X0[:2]) ** 2))
            return np.array([height, distance])
        else:
            distance = ca.sqrt((X[0] - X0[0]) ** 2 + (X[1] - X0[1]) ** 2)
            return ca.vertcat(height, distance)

    def f_1x(self, x, dt=0):
        k1 = 0.16
        k2 = 0.0064
        x0 = x[0] - self.time_step * 2 * k1 * x[0] ** 2 + self.time_step * 2 * k2 * x[1]
        x1 = x[1] + self.time_step * k1 * x[0] ** 2 - self.time_step * k2 * x[1]
        return np.array([x0, x1])

    def h_1x(self, x):
        return np.array(x[0]+x[1])

    def filter(self):
        """
        Run the Kalman filter
        """
        self.filter_means = [self.m_0]
        self.filter_covs = [self.P_0]
        sigmas = JulierSigmaPoints(n=self.x_dim, kappa=1)
        ukf = UnscentedKalmanFilter(dim_x=self.x_dim, dim_z=self.y_dim, dt=0, hx=self.h_1x,
                                    fx=self.f_1x,
                                    points=sigmas)
        ukf.x = self.m_0
        ukf.P = self.P_0

        for t in range(self.data.shape[0]):
            ukf.predict()
            y = self.data[t]
            if not np.isnan(y).any():
                ukf.update(y)

            self.filter_covs.append(ukf.P)
            m_bar = ukf.x

            self.filter_means.append(m_bar[:, None])

        self.filter_means = self.filter_means[1:]
        self.filter_covs = self.filter_covs[1:]
