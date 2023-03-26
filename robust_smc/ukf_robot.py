import numpy as np
import casadi as ca
from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints
import scipy.linalg as la

u = np.zeros(2,)


class UKFRobot:
    def __init__(self, data, transition_matrix, transition_cov, observation_cov, m_0, P_0, U):

        self.data = data
        self.transition_matrix = transition_matrix
        self.transition_cov = transition_cov

        self.y_dim = data.shape[1]
        self.observation_cov = observation_cov
        self.m_0 = m_0
        self.P_0 = P_0
        self.x_dim = transition_cov.shape[0]
        self.time_step = 1.0 / 15
        self.U = U

    def fx(self, x, dt=0):
        x0 = x[0] + self.time_step * self.u[0] * np.cos(x[2])
        x1 = x[1] + self.time_step * self.u[0] * np.sin(x[2])
        x2 = x[2] + self.time_step * self.u[1]

        return np.array([x0, x1, x2])

    def hx(self, x):
        obstacle0 = [1.052, -2.695]
        obstacle1 = [4.072, -1.752]
        obstacle2 = [6.028, - 3.324]
        l = 0.3296

        y0 = la.norm(np.array([obstacle0[0] - x[0] - l * np.cos(x[2]), obstacle0[1] - x[1] - l * np.sin(x[2])]), 2)
        y1 = la.norm(np.array([obstacle1[0] - x[0] - l * np.cos(x[2]), obstacle1[1] - x[1] - l * np.sin(x[2])]), 2)
        y2 = la.norm(np.array([obstacle2[0] - x[0] - l * np.cos(x[2]), obstacle2[1] - x[1] - l * np.sin(x[2])]), 2)
        y3 = np.arctan((obstacle0[1] - x[1] - l * np.sin(x[2])) / (obstacle0[0] - x[0] - l * np.cos(x[2]))) - x[2]
        y4 = np.arctan((obstacle1[1] - x[1] - l * np.sin(x[2])) / (obstacle1[0] - x[0] - l * np.cos(x[2]))) - x[2]
        y5 = np.arctan((obstacle2[1] - x[1] - l * np.sin(x[2])) / (obstacle2[0] - x[0] - l * np.cos(x[2]))) - x[2]

        return np.array([y0, y1, y2, y3, y4, y5])
        # return np.array([y0, y3])

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
        ukf.x = self.m_0
        ukf.P = self.P_0

        for t in range(self.data.shape[0]):
            if t == 0:
                self.u = np.zeros_like(self.U[t])
            else:
                self.u = self.U[t-1]
            ukf.predict()
            y = np.atleast_1d(self.data[t])
            if not np.isnan(y).any():
                ukf.update(y)

            self.filter_covs.append(ukf.P)
            m_bar = ukf.x

            self.filter_means.append(m_bar[:, None])

        self.filter_means = self.filter_means[1:]
        self.filter_covs = self.filter_covs[1:]
