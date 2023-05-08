import numpy as np
import casadi as ca
from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints


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
        k1 = 0.16
        k2 = 0.0064
        x0 = x[0] - self.time_step * 2 * k1 * x[0] ** 2 + self.time_step * 2 * k2 * x[1]
        x1 = x[1] + self.time_step * k1 * x[0] ** 2 - self.time_step * k2 * x[1]
        return np.array([x0, x1])

    def hx(self, x):
        return np.atleast_1d(x[0] + x[1])

    def filter(self):
        """
        Run the Kalman filter
        """
        self.filter_means = [self.m_0]
        self.filter_covs = [self.P_0]
        sigmas = JulierSigmaPoints(n=self.x_dim, kappa=1)
        ukf = UnscentedKalmanFilter(dim_x=self.x_dim, dim_z=self.y_dim, dt=0, hx=self.hx,
                                    fx=self.fx,
                                    points=sigmas)
        ukf.Q = self.transition_cov
        ukf.R = self.observation_cov
        ukf.x = self.m_0
        ukf.P = self.P_0

        for t in range(self.data.shape[0]):
            ukf.predict()
            y = np.atleast_1d(self.data[t])
            if not np.isnan(y).any():
                ukf.update(y)

            self.filter_covs.append(ukf.P)
            m_bar = ukf.x

            self.filter_means.append(m_bar[:, None])

        self.filter_means = self.filter_means[1:]
        self.filter_covs = self.filter_covs[1:]
