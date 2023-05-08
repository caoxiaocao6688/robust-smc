import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints


class UKFRobot:
    def __init__(self, data, transition_matrix, transition_cov, observation_cov, m_0, P_0, U, X):

        self.data = data
        self.transition_cov = transition_cov

        self.y_dim = data.shape[1]
        self.observation_cov = observation_cov
        self.m_0 = m_0
        self.P_0 = P_0
        self.x_dim = transition_cov.shape[0]
        self.time_step = 1.0 / 15
        self.U = U
        self.X = X

    def fx(self, x, dt=0):
        x0 = x[0] + self.time_step * self.u[0] * np.cos(x[2]) - 0.0001675046729610055
        x1 = x[1] + self.time_step * self.u[0] * np.sin(x[2]) - 0.0001963914687308423
        x2 = x[2] + self.time_step * self.u[1] + 0.0005640178926637775

        return np.array([x0, x1, x2])

    def hx(self, x):
        px, py, theta = x
        obstacle_info = [[1.052, -2.695], [4.072, -1.752], [6.028, -3.324]]
        obstacle = np.array(obstacle_info)
        der_x_robot = 0.329578
        rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        obstacle_obs = (obstacle - x[:2]) @ rot.T - np.array([der_x_robot, 0])
        dist = np.linalg.norm(obstacle_obs, axis=1)
        angle = np.arctan2(obstacle_obs[:, 1], obstacle_obs[:, 0])
        return np.concatenate([dist, angle]) + np.array([- 0.0312, - 0.0581, - 0.0557, 0.0053, 0.0059, 0.0125])

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
            if t == 0:
                self.u = np.zeros_like(self.U[t])
            else:
                self.u = self.U[t - 1]
            ukf.predict()
            y = np.atleast_1d(self.data[t])
            if not np.isnan(y).any():
                # ukf.update(y)
                pass

            self.filter_covs.append(ukf.P)
            m_bar = ukf.x

            self.filter_means.append(m_bar[:, None])

        self.filter_means = self.filter_means[1:]
        self.filter_covs = self.filter_covs[1:]

    def generate_state_sequence(self):
        generated_states = [self.X[0]]
        for i in range(1, self.X.shape[0]):
            self.u = self.U[i]
            next_state = self.fx(generated_states[-1])
            generated_states.append(next_state)

        return np.array(generated_states)

    def compare_state_sequences(self):
        generated_states = self.generate_state_sequence()

        plt.figure(figsize=(12, 6))
        labels = ['X', 'Y', 'Theta']
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(self.X[:, i], label='True')
            plt.plot(generated_states[:, i], label='Generated', linestyle='--')
            plt.xlabel('Time')
            plt.ylabel(labels[i])
            plt.legend()

        plt.tight_layout()
        plt.show()

    def generate_observations(self):
        generated_observations = [self.hx(state) for state in self.X]
        return np.array(generated_observations)

    def compare_observations(self):
        generated_observations = self.generate_observations()

        plt.figure(figsize=(12, 6))
        labels = ['Distance 1', 'Distance 2', 'Distance 3', 'Angle 1', 'Angle 2', 'Angle 3']
        for i in range(6):
            plt.subplot(3, 2, i + 1)
            plt.plot(self.data[:, i], label='True')
            plt.plot(generated_observations[:, i], label='Generated', linestyle='--')
            plt.xlabel('Time')
            plt.ylabel(labels[i])
            plt.legend()

        plt.tight_layout()
        plt.show()