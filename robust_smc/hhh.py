import numpy as np
import casadi as ca


def hx(x):
    px, py, theta = x
    obstacle_info = [[1.052, -2.695], [4.072, -1.752], [6.028, -3.324]]
    obstacle = np.array(obstacle_info)
    der_x_robot = 0.329578
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    obstacle_obs = (obstacle - x[:2]) @ rot.T - np.array([der_x_robot, 0])
    dist = np.linalg.norm(obstacle_obs, axis=1)
    angle = np.arctan2(obstacle_obs[:, 1], obstacle_obs[:, 0])
    return np.concatenate([dist, angle]) + np.array([- 0.0312, - 0.0581, - 0.0557, 0.0053, 0.0059, 0.0125])


def h_ca(x):
    px, py, theta = x[0], x[1], x[2]
    obstacle_info = [[1.052, -2.695], [4.072, -1.752], [6.028, -3.324]]
    obstacle = ca.MX(obstacle_info)
    der_x_robot = 0.329578
    rot = ca.MX([[ca.cos(theta), ca.sin(theta)], [-ca.sin(theta), ca.cos(theta)]])
    obstacle_obs = (obstacle - x[:2].T) @ rot.T - ca.MX([der_x_robot, 0])
    dist = ca.norm_2(obstacle_obs, axis=1)
    angle = ca.atan2(obstacle_obs[:, 1], obstacle_obs[:, 0])
    return ca.vertcat(dist, angle) + ca.MX([- 0.0312, - 0.0581, - 0.0557, 0.0053, 0.0059, 0.0125])



x = np.array([1.0, 2.0, 0.5])
x_ca = ca.MX(x.reshape(3, 1))  #
hx_result = hx(x)
h_ca_result = h_ca(x_ca)

# 将CasADi数组转换为NumPy数组
h_ca_result_np = np.array(h_ca_result).reshape(hx_result.shape)

# 比较两个结果是否相同
comparison = np.isclose(hx_result, h_ca_result_np, atol=1e-6)
are_same = np.all(comparison)

print("hx 函数的结果：", hx_result)
print("h_ca 函数的结果：", h_ca_result_np)
print("两个函数的输出是否相同：", are_same)
