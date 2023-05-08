import casadi as ca
import numpy as np
import scipy.linalg as la
from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints


class NonlinearMheRobot:
    def __init__(self, data, transition_matrix, transition_cov, observation_cov, m_0, P_0, U):

        self.data = data
        self.transition_matrix = transition_matrix
        self.transition_cov = transition_cov

        self.y_dim = data.shape[1]
        self.u_dim = U.shape[1]
        self.observation_cov = observation_cov
        self.m_0 = m_0
        self.P_0 = P_0
        self.slide_window = 1
        self.x_dim = transition_cov.shape[0]
        self.time_step = 1.0 / 15
        self.U = U

    def fx(self, x, dt=0):
        x0 = x[0] + self.time_step * self.u[0] * np.cos(x[2]) - 0.0001675046729610055
        x1 = x[1] + self.time_step * self.u[0] * np.sin(x[2]) - 0.0001963914687308423
        x2 = x[2] + self.time_step * self.u[1] + 0.0005640178926637775
        return np.array([x0, x1, x2])

    def f_ca(self, x, u):
        x0 = x[0] + self.time_step * u[0] * np.cos(x[2]) - 0.0002
        x1 = x[1] + self.time_step * u[0] * np.sin(x[2]) - 0.0002
        x2 = x[2] + self.time_step * u[1] + 0.0006
        return ca.vertcat(x0, x1, x2)

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

    def h_ca(self, x):
        px, py, theta = x[0], x[1], x[2]
        obstacle_info = [[1.052, -2.695], [4.072, -1.752], [6.028, -3.324]]
        obstacle = np.array(obstacle_info)
        der_x_robot = 0.329578
        rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        temp = ca.repmat(x[:2].T, 3, 1)
        obstacle_obs = (obstacle - temp) @ rot.T - np.tile(np.array([der_x_robot, 0]), (3, 1))
        dist = ca.sqrt(ca.diag(ca.mtimes(obstacle_obs, obstacle_obs.T)))
        angle = np.arctan2(obstacle_obs[:, 1], obstacle_obs[:, 0])
        return ca.vertcat(dist[0] - 0.0312, dist[1] - 0.0581, dist[2] - 0.0557, angle[0] + 0.0053,
                          angle[1] + 0.0059, angle[2] + 0.0125)
    # 示例输入

    def filter(self):
        """
        Run the Kalman filter
        """
        self.filter_means = [self.m_0]
        self.filter_means_ukf = [self.m_0]
        self.filter_covs = [self.P_0]
        u_seq = np.zeros((self.slide_window, self.u_dim))
        y_seq = np.zeros((self.slide_window, self.y_dim))
        sigmas = JulierSigmaPoints(n=self.x_dim, kappa=1)
        ukf = UnscentedKalmanFilter(dim_x=self.x_dim, dim_z=self.y_dim, dt=0, hx=self.hx,
                                    fx=self.fx,
                                    points=sigmas)
        ukf.x = self.m_0
        ukf.P = self.P_0
        ukf.Q = self.transition_cov
        ukf.R = self.observation_cov

        for t in range(self.data.shape[0]):
            if t == 0:
                self.u = np.zeros_like(self.U[t])
            else:
                self.u = self.U[t - 1]
            ukf.x = self.filter_means[-1].flatten()
            ukf.predict()
            y = self.data[t]
            if not np.isnan(y).any():
                ukf.update(y)

            self.filter_covs.append(ukf.P)
            m_bar = ukf.x

            if t < self.slide_window:
                y_seq[t] = y
                u_seq[t] = self.u
                self.filter_means.append(m_bar[:, None])

            else:
                y_seq[0:self.slide_window - 1] = y_seq[1:self.slide_window]
                y_seq[self.slide_window - 1] = y
                u_seq[0:self.slide_window - 1] = u_seq[1:self.slide_window]
                u_seq[self.slide_window - 1] = self.u
                # if self.filter_covs[t - self.slide_window + 1] ==
                sol = self.casadi_mhe(self.filter_means[t - self.slide_window + 1],
                                      self.filter_covs[t - self.slide_window + 1],
                                      y_seq,
                                      u_seq,
                                      slide_window=self.slide_window)
                sol = np.array(sol.full())
                m_bar = self.solve_mhe(sol)[:, None]
                self.filter_means.append(m_bar)

            self.filter_means_ukf.append(ukf.x[:, None])

        self.filter_means = self.filter_means[1:]
        self.filter_covs = self.filter_covs[1:]

    def casadi_mhe(self, x_bar0, P_0, y_seq, u_seq, slide_window):
        ca_x = ca.SX.sym('ca_x', self.x_dim, 1)
        ca_u = ca.SX.sym('ca_u', self.u_dim, 1)
        ca_xi = ca.SX.sym('ca_xi', self.x_dim, 1)

        # 自变量
        ca_x_hat0 = ca.SX.sym('ca_x_hat0', self.x_dim, 1)
        ca_Xi = ca.SX.sym('ca_Xi', self.x_dim, slide_window)

        # 动态参数
        ca_x_bar0 = ca.SX.sym('ca_x_bar0', self.x_dim, 1)
        ca_P0_inv = ca.SX.sym('ca_P0_inv', self.x_dim, self.x_dim)
        ca_Y = ca.SX.sym('Y', self.y_dim, slide_window)
        ca_U = ca.SX.sym('U', self.u_dim, slide_window)

        # 静态参数
        ca_Q_inv = ca.DM(np.linalg.inv(self.transition_cov))
        ca_R_inv = ca.DM(np.linalg.inv(self.observation_cov))

        # 模型
        # ca_RHS = self.transition_matrix @ ca_x + ca_xi # TDDO
        ca_RHS = self.f_ca(ca_x, ca_u) + ca_xi
        ca_f = ca.Function('f', [ca_x, ca_u, ca_xi], [ca_RHS])

        ca_RHS = self.h_ca(ca_x)
        ca_h = ca.Function('h', [ca_x], [ca_RHS])

        ca_x_hat = ca_x_hat0
        scale = 1
        ca_cost_fn = (ca_x_hat - ca_x_bar0).T @ ca_P0_inv @ (ca_x_hat - ca_x_bar0) * scale  # cost function

        for k in range(slide_window):
            ca_xi = ca_Xi[:, k]
            ca_y = ca_Y[:, k]
            ca_u = ca_U[:, k]
            ca_x_hat = ca_f(ca_x_hat, ca_u, ca_xi)
            ca_cost_fn = ca_cost_fn \
                         + (ca_y - ca_h(ca_x_hat)).T @ ca_R_inv @ (ca_y - ca_h(ca_x_hat)) * scale \
                         + ca_xi.T @ ca_Q_inv @ ca_xi * scale

        # 自变量设置
        ca_OPT_variables = ca.vertcat(
            ca_x_hat0.reshape((-1, 1)),  # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            ca_Xi.reshape((-1, 1))
        )

        # 动态参数设置
        ca_P = ca.vertcat(
            ca_x_bar0.reshape((-1, 1)),  # (2,1)
            ca_P0_inv.reshape((-1, 1)),  # (2,2)->(2,2)
            ca_Y.reshape((-1, 1)),
            ca_U.reshape((-1, 1))
        )

        # 求解问题设置
        ca_nlp_prob = {
            'f': ca_cost_fn,
            'x': ca_OPT_variables,
            'p': ca_P
        }

        # 优化器设置
        ca_opts = {
            'ipopt': {
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': False,
            'verbose': False,
        }

        ca_solver = ca.nlpsol('solver', 'ipopt', ca_nlp_prob, ca_opts)

        # 自变量上下界
        ca_lbx = ca.DM.zeros((self.x_dim + self.x_dim * slide_window, 1))
        ca_ubx = ca.DM.zeros((self.x_dim + self.x_dim * slide_window, 1))
        ca_lbx[0: self.x_dim] = -ca.inf
        ca_ubx[0: self.x_dim] = ca.inf
        ca_lbx[self.x_dim:] = -ca.inf
        ca_ubx[self.x_dim:] = ca.inf

        # 迭代初值
        x_init = np.zeros([self.x_dim + self.x_dim * slide_window, 1])
        x_init[0:self.x_dim, :] = np.ones([self.x_dim, 1])
        p = np.vstack((x_bar0.reshape(-1, 1), np.linalg.inv(P_0).reshape(-1, 1), y_seq.transpose().reshape(-1, 1),
                       u_seq.transpose().reshape(-1, 1)))

        sol = ca_solver(
            x0=x_init,
            lbx=ca_lbx,
            ubx=ca_ubx,
            p=p
        )

        return sol['x']

    def solve_mhe(self, sol):
        x = sol[0:self.x_dim]
        for i in range(self.slide_window):
            x_next = sol[
                     self.x_dim + self.x_dim * i:self.x_dim + self.x_dim * i + self.x_dim] + self.fx(x)
            x = x_next
        return x.flatten()
