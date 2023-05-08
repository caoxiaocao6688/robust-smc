import numpy as np
from numpy.linalg import cholesky


class ConstantVelocityModel:
    """
    Constant Velosity model with Gaussian Explosion
    """

    def __init__(self, final_time, time_step=0.1, observation_cov=None,
                 explosion_scale=100.0, contamination_probability=0.05, seed=None):
        self.final_time = final_time
        self.time_step = time_step
        self.simulation_steps = int(final_time / time_step)
        self.observation_cov = np.eye(2) if observation_cov is None else observation_cov
        self.explosion_scale = explosion_scale
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.contamination_probability = contamination_probability
        self._build_system()  # 生成系统参数
        self._simulate_system()  # 仿真系统，得到观测数据

    def _build_system(self):
        self.dim_X = 4
        self.dim_Y = 2

        transition_matrix = np.eye(self.dim_X)
        transition_matrix[0, 2] = transition_matrix[1, 3] = self.time_step
        self.transition_matrix = transition_matrix
        self.observation_matrix = np.eye(2, 4)

        off_diagonal = (self.time_step ** 2 / 2) * np.eye(2)
        upper_diagonal = (self.time_step ** 3 / 3) * np.eye(2)
        lower_diagonal = self.time_step * np.eye(2)
        self.process_cov = np.vstack([np.hstack([upper_diagonal, off_diagonal]),
                                      np.hstack([off_diagonal, lower_diagonal])])

        self.initial_cov = np.eye(self.dim_X)
        self.initial_state = np.array([140., 140., 50., 0.])[:, None]

    def _simulate_system(self):
        L = cholesky(self.process_cov)
        R = cholesky(self.observation_cov)

        X = [self.initial_state + cholesky(self.initial_cov) @ self.rng.randn(self.dim_X, 1)]
        Y = [self.observation_matrix @ X[-1] + R @ self.rng.randn(self.dim_Y, 1)]

        for _ in range(self.simulation_steps - 1):
            X_new = self.transition_matrix @ X[-1] + L @ self.rng.randn(self.dim_X, 1)
            Y_new = self.observation_matrix @ X_new + R @ self.rng.randn(self.dim_Y, 1)

            u = self.rng.rand()
            if u < self.contamination_probability:
                Y_new += self.explosion_scale * self.rng.randn(self.dim_Y, 1)

            X.append(X_new)
            Y.append(Y_new)

        self.X = np.stack(X).squeeze(axis=-1)
        self.Y = np.stack(Y).squeeze(axis=-1)

    def renoise(self):
        R = cholesky(self.observation_cov)

        Y = []
        for X in self.X:
            Y_new = self.observation_matrix @ X[:, None] + R @ self.rng.randn(self.dim_Y, 1)

            u = self.rng.rand()
            if u < self.contamination_probability:
                Y_new += self.explosion_scale * self.rng.randn(self.dim_Y, 1)

            Y.append(Y_new)

        return np.stack(Y).squeeze(axis=-1)


class ReversibleReaction:
    """
    Terrain Aided Navigation (TAN) simulator. Taken from Merlinge et. al. 2019
    """

    def __init__(self, final_time, time_step=0.1, observation_std=None, degrees_of_freedom=1,
                 X0=None, process_std=None, contamination_probability=None, seed=None):
        """
        :param final_time: final time period
        :param time_step: time step
        :param observation_std: observation noise standard deviation
        :param X0: initial state
        :param transition_matrix: transition matrix
        :param process_std: process standard deviation
        :param seed: random seed
        """
        self.final_time = final_time
        self.time_step = time_step
        if X0 is None:
            X0 = np.array([3, 1])
        self.X0 = X0
        self.simulation_steps = int(final_time / time_step)
        self.observation_std = observation_std
        if process_std is None:
            process_std = np.array([0.01, 0.01])
        self.process_std = process_std
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.k1 = 0.16
        self.k2 = 0.0064
        self.contamination_probability = contamination_probability
        self.degrees_of_freedom = degrees_of_freedom
        self.transition_matrix = None
        self._simulate_system()

    def observation_model(self, X):
        return X[:, 0][:, None] + X[:, 1][:, None]

    def noise_model(self, Y):
        u = self.rng.rand(Y.shape[0])
        noise = np.zeros_like(Y)
        norm_loc = (u > self.contamination_probability)
        t_loc = (u <= self.contamination_probability)
        self.contamination_locations = np.argwhere(t_loc)
        noise[norm_loc] = self.rng.randn(norm_loc.sum(), Y.shape[1])
        noise[t_loc] = self.rng.standard_t(df=self.degrees_of_freedom, size=(t_loc.sum(), 1))
        return self.observation_std * noise

    def _simulate_system(self):
        """
        Simulates the TAN system
        """
        X = np.zeros((self.simulation_steps + 1, 2))
        X[0, :] = self.X0
        for t in range(self.simulation_steps):
            X0 = X[t, 0] - self.time_step * 2 * self.k1 * X[t, 0] ** 2 + self.time_step * 2 * self.k2 * X[
                t, 1]
            X1 = X[t, 1] + self.time_step * self.k1 * X[t, 0] ** 2 - self.time_step * self.k2 * X[
                t, 1]
            state_evolution = np.array([[X0], [X1]])
            process_noise = self.process_std[:, None] * self.rng.randn(2, 1)
            X[t + 1, :] = (state_evolution + process_noise)[:, 0]

        Y = self.observation_model(X)
        Y += self.noise_model(Y)
        self.X, self.Y = X, Y

    def renoise(self):
        Y = self.observation_model(self.X)
        Y += self.noise_model(Y)
        return Y
