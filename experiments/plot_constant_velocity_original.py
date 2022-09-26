import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager
# BETA = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
BETA = [0.1, 0.2, 0.5, 0.8]

# BETA = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8]

# BETA = [0.0001, 0.0005]
palette = sns.color_palette("bright", len(BETA)+2)

kf_error, mhe_error, robust_mhes_error = np.load(
    '../results/constant-velocity/impulsive_noise/original_data/beta-sweep-contamination-0.2.pk', allow_pickle=True)
time_length = int(kf_error.shape[1]/5)

pd0_list, pd1_list, pd2_list, pd3_list = [], [], [], []
pd_list = [pd0_list, pd1_list, pd2_list, pd3_list]
for i in range(kf_error.shape[2]):
    for j in range(kf_error.shape[0]):
        pd_list[i].append(pd.DataFrame({'Error': kf_error[j, 0:time_length, i],
                                        'Algorithm': 'KF',
                                        'Step': np.arange(time_length)
                                        }))

        pd_list[i].append(pd.DataFrame({'Error': mhe_error[j, 0:time_length, i],
                                        'Algorithm': 'MHE',
                                        'Step': np.arange(time_length)}))

        for k, beta in enumerate(BETA):
            string = 'beta=' + str(beta)
            pd_list[i].append(pd.DataFrame({'Error': robust_mhes_error[j, k, 0:time_length, i],
                                            'Algorithm': string,
                                            'Step': np.arange(time_length)}))

pd_error0 = pd.concat(pd_list[0])
pd_error1 = pd.concat(pd_list[1])
pd_error2 = pd.concat(pd_list[2])
pd_error3 = pd.concat(pd_list[3])
pd_error0 = pd_error0.reset_index(drop=True)    # Apply reset_index function, drop=False时, 保留旧的索引为index列
pd_error1 = pd_error1.reset_index(drop=True)    # Apply reset_index function, drop=False时, 保留旧的索引为index列
pd_error2 = pd_error2.reset_index(drop=True)    # Apply reset_index function, drop=False时, 保留旧的索引为index列
pd_error3 = pd_error3.reset_index(drop=True)    # Apply reset_index function, drop=False时, 保留旧的索引为index列

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DeJavu Serif",
    "font.sans-serif": ["Helvetica"]})

f1 = plt.figure(1)
ax1 = f1.add_axes([0.155, 0.12, 0.82, 0.86])
g1 = sns.lineplot(x='Step', y="Error", hue="Algorithm", style="Algorithm", data=pd_error0,
                  linewidth=2, palette=palette, dashes=False
                  )
ax1.set_ylabel('Error', fontsize=15)
ax1.set_xlabel("Step", fontsize=15)
plt.xlim(0, time_length)
ax1.set_xticks(40 * np.arange(6))
ax1.set_xticklabels(('0', '40', '80', '120', '160', '200'), fontsize=12)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.savefig("../figures/constant-velocity/impulsive_noise/original/1.pdf")

f2 = plt.figure(2)
ax2 = f2.add_axes([0.155, 0.12, 0.82, 0.86])
g2 = sns.lineplot(x='Step', y="Error", hue="Algorithm", style="Algorithm", data=pd_error1,
                  linewidth=2, palette=palette, dashes=False
                  )
ax2.set_ylabel('Error', fontsize=15)
ax2.set_xlabel("Step", fontsize=15)
plt.xlim(0, time_length)
ax2.set_xticks(40 * np.arange(6))
ax2.set_xticklabels(('0', '40', '80', '120', '160', '200'), fontsize=12)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.savefig("../figures/constant-velocity/impulsive_noise/original/2.pdf")

f3 = plt.figure(3)
ax3 = f3.add_axes([0.155, 0.12, 0.82, 0.86])
g3 = sns.lineplot(x='Step', y="Error", hue="Algorithm", style="Algorithm", data=pd_error2,
                  linewidth=2, palette=palette, dashes=False
                  )
ax3.set_ylabel('Error', fontsize=15)
ax3.set_xlabel("Step", fontsize=15)
plt.xlim(0, time_length)
ax3.set_xticks(40 * np.arange(6))
ax3.set_xticklabels(('0', '40', '80', '120', '160', '200'), fontsize=12)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.savefig("../figures/constant-velocity/impulsive_noise/original/3.pdf")

f4 = plt.figure(4)
ax4 = f4.add_axes([0.155, 0.12, 0.82, 0.86])
g4 = sns.lineplot(x='Step', y="Error", hue="Algorithm", style="Algorithm", data=pd_error3,
                  linewidth=2, palette=palette, dashes=False)
ax4.set_ylabel('Error', fontsize=15)
ax4.set_xlabel("Step", fontsize=15)
plt.xlim(0, time_length)
ax4.set_xticks(40 * np.arange(6))
ax4.set_xticklabels(('0', '40', '80', '120', '160', '200'), fontsize=12)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.savefig("../figures/constant-velocity/impulsive_noise/original/4.pdf")