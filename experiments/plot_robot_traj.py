import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot(true_trajectory, ukf_trajectory, mhe_trajectory, robust_mhe_trajectory, save_path):
    # 设置字体为 serif
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 15

    # 计算误差方差
    error_ukf = np.sqrt(np.sum((true_trajectory - ukf_trajectory) ** 2, axis=2))
    error_mhe = np.sqrt(np.sum((true_trajectory - mhe_trajectory) ** 2, axis=2))
    error_robust_mhe = np.sqrt(np.sum((true_trajectory - robust_mhe_trajectory) ** 2, axis=2))

    # 计算每个轨迹的平均值
    mean_true = np.mean(true_trajectory, axis=0)
    mean_ukf = np.mean(ukf_trajectory, axis=0)
    mean_mhe = np.mean(mhe_trajectory, axis=0)
    mean_robust_mhe = np.mean(robust_mhe_trajectory, axis=0)

    # 创建一个新的绘图
    plt.figure()

    # 绘制每个轨迹的平均值，颜色由误差方差确定
    plt.scatter(mean_true[:, 0], mean_true[:, 1], c='blue', marker='o', label='True trajectory')
    plt.scatter(mean_ukf[:, 0], mean_ukf[:, 1], c=np.mean(error_ukf, axis=0), marker='+', cmap='Greens', label='UKF', edgecolors='face', vmin=0, vmax=np.percentile(error_ukf, 75))
    plt.scatter(mean_mhe[:, 0], mean_mhe[:, 1], c=np.mean(error_mhe, axis=0), marker='+', cmap='Purples', label='MHE', edgecolors='face', vmin=0, vmax=np.percentile(error_mhe, 75))
    plt.scatter(mean_robust_mhe[:, 0], mean_robust_mhe[:, 1], c=np.mean(error_robust_mhe, axis=0), marker='+', cmap='Reds', label='βMHE', edgecolors='face', vmin=0, vmax=np.percentile(error_robust_mhe, 75))

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    # plt.title('Robot Trajectories with Estimation Error Variance')

    # 添加一个颜色条
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=max(np.max(error_ukf), np.max(error_mhe), np.max(error_robust_mhe))))
    sm.set_array([])
    cbar = plt.colorbar(sm, label='Estimation Error Variance')

    # 添加图例
    max_ukf_error = np.percentile(error_ukf, 75)
    max_mhe_error = np.percentile(error_mhe, 75)
    max_robust_mhe_error = np.percentile(error_robust_mhe, 75)

    # 创建代表颜色的列表
    ukf_legend_color = plt.cm.Greens(max_ukf_error / np.percentile(error_ukf, 75))
    mhe_legend_color = plt.cm.Purples(max_mhe_error / np.percentile(error_mhe, 75))
    robust_mhe_legend_color = plt.cm.Reds(max_robust_mhe_error / np.percentile(error_robust_mhe, 75))

    # 创建图例条目
    true_patch = mpl.lines.Line2D([], [], color='blue', marker='o', markersize=5, label='True trajectory')
    ukf_patch = mpl.lines.Line2D([], [], color=ukf_legend_color, marker='+', markersize=5, label='UKF')
    mhe_patch = mpl.lines.Line2D([], [], color=mhe_legend_color, marker='+', markersize=5, label='MHE')
    robust_mhe_patch = mpl.lines.Line2D([], [], color=robust_mhe_legend_color, marker='+', markersize=5, label='βMHE')

    # 添加图例到图中
    plt.legend(handles=[true_patch, ukf_patch, mhe_patch, robust_mhe_patch])

    plt.grid()

    # 保存图像为 PDF
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()


true, ukf, mhe, robust_mhe = np.load(
    '../results/robot_estimation/traj_0.01.pk', allow_pickle=True)
# 绘制并保存轨迹图像
save_path = "../figures/robot_estimation/robot_trajectories.pdf"
plot(true, ukf, mhe, robust_mhe, save_path)
