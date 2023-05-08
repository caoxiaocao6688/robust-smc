import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

BETA = [0.00001, 0.00002, 0.00004, 0.00006, 0.00008, 0.0001]
BETA = [0.0001]
font = 25
lfont = 15

matplotlib.rcParams['font.family'] = ['serif']

palette = sns.color_palette("tab10", len(BETA) + 2)

kf_error, mhe_error, robust_mhes_error = np.load(
    '../results/constant_velocity/original_0.2.pk', allow_pickle=True)
time_length = int(kf_error.shape[1])

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
pd_error0 = pd_error0.reset_index(drop=True)  # Apply reset_index function, drop=False时, 保留旧的索引为index列
pd_error1 = pd_error1.reset_index(drop=True)  # Apply reset_index function, drop=False时, 保留旧的索引为index列
pd_error2 = pd_error2.reset_index(drop=True)  # Apply reset_index function, drop=False时, 保留旧的索引为index列
pd_error3 = pd_error3.reset_index(drop=True)  # Apply reset_index function, drop=False时, 保留旧的索引为index列
f = 4
axes_pos = [0.2, 0.2, 0.6, 0.6]
if f == 1 or f == 5:
    f1 = plt.figure(1)
    ax1 = f1.add_axes(axes_pos)
    g1 = sns.lineplot(x='Step', y="Error", hue="Algorithm", style="Algorithm", data=pd_error0,
                      linewidth=1, palette=palette, dashes=False)
    ax1.set_ylabel('Error', fontsize=font)
    ax1.set_xlabel("Step", fontsize=font)
    plt.xlim(0, time_length)
    plt.ylim(-20, 80)
    plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
    ax1.set_xticks(40 * np.arange(6))
    ax1.set_xticklabels(('0', '40', '80', '120', '160', '200'), fontsize=font)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=['KF', 'MHE', r'$\beta$-MHE'], fontsize=lfont)
    plt.yticks(fontsize=font)
    plt.xticks(fontsize=font)

    ax_in1 = inset_axes(ax1, width="50%", height="50%", loc='center', bbox_to_anchor=(-0.17, 0.05, 1, 1),
                        bbox_transform=ax1.transAxes)
    g1_in = sns.lineplot(x='Step', y="Error", hue="Algorithm", style="Algorithm", data=pd_error0,
                         linewidth=1, palette=palette, dashes=False)
    ax_in1.get_legend().remove()
    ax_in1.get_xaxis().set_visible(False)
    ax_in1.get_yaxis().set_visible(False)
    ax_in1.set_xlim(50, 150)
    ax_in1.set_ylim(-4, 4)
    plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
    mark_inset(ax1, ax_in1, loc1=3, loc2=1, fc="none", ec='r', lw=2, ls='dotted', alpha=0.8)
    mark_inset(ax1, ax_in1, loc1=4, loc2=2, fc="none", ec='r', lw=2, ls='dotted', alpha=0.8)

    plt.savefig("../figures/constant_velocity/original_1.pdf")

if f == 2 or f == 5:
    f2 = plt.figure(2)
    ax2 = f2.add_axes(axes_pos)
    g2 = sns.lineplot(x='Step', y="Error", hue="Algorithm", style="Algorithm", data=pd_error1,
                      linewidth=1, palette=palette, dashes=False
                      )
    ax2.set_ylabel('Error', fontsize=font)
    ax2.set_xlabel("Step", fontsize=font)
    plt.xlim(0, time_length)
    plt.ylim(-20, 80)
    plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
    ax2.set_xticks(40 * np.arange(6))
    ax2.set_xticklabels(('0', '40', '80', '120', '160', '200'), fontsize=font)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles, labels=['KF', 'MHE', r'$\beta$-MHE'], fontsize=lfont)
    plt.yticks(fontsize=font)
    plt.xticks(fontsize=font)

    ax_in2 = inset_axes(ax2, width="50%", height="50%", loc='center', bbox_to_anchor=(-0.17, 0.05, 1, 1),
                        bbox_transform=ax2.transAxes)
    g2_in = sns.lineplot(x='Step', y="Error", hue="Algorithm", style="Algorithm", data=pd_error1,
                         linewidth=1, palette=palette, dashes=False)
    ax_in2.get_legend().remove()
    ax_in2.get_xaxis().set_visible(False)
    ax_in2.get_yaxis().set_visible(False)
    ax_in2.set_xlim(50, 150)
    ax_in2.set_ylim(-4, 4)
    plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
    mark_inset(ax2, ax_in2, loc1=3, loc2=1, fc="none", ec='r', lw=2, ls='dotted', alpha=0.8)
    mark_inset(ax2, ax_in2, loc1=4, loc2=2, fc="none", ec='r', lw=2, ls='dotted', alpha=0.8)

    plt.savefig("../figures/constant_velocity/original_2.pdf")

if f == 3 or f == 5:
    f3 = plt.figure(3)
    ax3 = f3.add_axes(axes_pos)
    g3 = sns.lineplot(x='Step', y="Error", hue="Algorithm", style="Algorithm", data=pd_error2,
                      linewidth=1, palette=palette, dashes=False
                      )
    ax3.set_ylabel('Error', fontsize=font)
    ax3.set_xlabel("Step", fontsize=font)
    plt.xlim(0, time_length)
    plt.ylim(-40, 80)
    plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
    ax3.set_xticks(40 * np.arange(6))
    ax3.set_xticklabels(('0', '40', '80', '120', '160', '200'), fontsize=font)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles=handles, labels=['KF', 'MHE', r'$\beta$-MHE'], fontsize=lfont)
    plt.yticks(fontsize=font)
    plt.xticks(fontsize=font)

    ax_in3 = inset_axes(ax3, width="50%", height="50%", loc='center', bbox_to_anchor=(-0.17, 0.2, 1, 1),
                        bbox_transform=ax3.transAxes)
    g3_in = sns.lineplot(x='Step', y="Error", hue="Algorithm", style="Algorithm", data=pd_error2,
                         linewidth=1, palette=palette, dashes=False)
    ax_in3.get_legend().remove()
    ax_in3.get_xaxis().set_visible(False)
    ax_in3.get_yaxis().set_visible(False)
    ax_in3.set_xlim(50, 150)
    ax_in3.set_ylim(-4, 4)
    plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
    mark_inset(ax3, ax_in3, loc1=3, loc2=1, fc="none", ec='r', lw=2, ls='dotted', alpha=0.8)
    mark_inset(ax3, ax_in3, loc1=4, loc2=2, fc="none", ec='r', lw=2, ls='dotted', alpha=0.8)

    plt.savefig("../figures/constant_velocity/original_3.pdf")

if f == 4 or f == 5:
    f4 = plt.figure(4)
    ax4 = f4.add_axes(axes_pos)
    g4 = sns.lineplot(x='Step', y="Error", hue="Algorithm", style="Algorithm", data=pd_error3,
                      linewidth=1, palette=palette, dashes=False)
    ax4.set_ylabel('Error', fontsize=font)
    ax4.set_xlabel("Step", fontsize=font)
    plt.xlim(0, time_length)
    plt.ylim(-50, 10)
    plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
    ax4.set_xticks(40 * np.arange(6))
    ax4.set_xticklabels(('0', '40', '80', '120', '160', '200'), fontsize=font)
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles=handles, labels=['KF', 'MHE', r'$\beta$-MHE'], fontsize=lfont, bbox_to_anchor=(0.05, 0.4))
    plt.yticks(fontsize=font)
    plt.xticks(fontsize=font)

    ax_in4 = inset_axes(ax4, width="50%", height="50%", loc='center', bbox_to_anchor=(0.23, -0.1, 1, 1),
                        bbox_transform=ax4.transAxes)
    g4_in = sns.lineplot(x='Step', y="Error", hue="Algorithm", style="Algorithm", data=pd_error3,
                         linewidth=1, palette=palette, dashes=False)
    ax_in4.get_legend().remove()
    ax_in4.get_xaxis().set_visible(False)
    ax_in4.get_yaxis().set_visible(False)
    ax_in4.set_xlim(50, 150)
    ax_in4.set_ylim(-4, 4)
    plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
    mark_inset(ax4, ax_in4, loc1=3, loc2=1, fc="none", ec='r', lw=2, ls='dotted', alpha=0.8)
    mark_inset(ax4, ax_in4, loc1=4, loc2=2, fc="none", ec='r', lw=2, ls='dotted', alpha=0.8)

    plt.savefig("../figures/constant_velocity/original_4.pdf")
