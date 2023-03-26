from cycler import cycler

import numpy as np
from scipy.stats import mode

import matplotlib.pyplot as plt
from matplotlib import cm, rc, patches, lines
import matplotlib

from robust_smc.data import ConstantVelocityModel
from experiment_utilities import pickle_load

# matplotlib Global Settings
palette = cycler(color=cm.Set1.colors)
rc('axes', prop_cycle=palette)
matplotlib.rcParams['font.family'] = ['serif']

# BETA = [r'$10^{-5}$', r'$2 \times 10^{-5}$', r'$4 \times 10^{-5}$', r'$6 \times 10^{-5}$', r'$8 \times 10^{-5}$',
#         r'$10^{-4}$', r'$2 \times 10^{-4}$']
BETA = ['0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5']
CONTAMINATION = [0.2]
LABELS = ['UKF', 'MHE'] + [r'$\beta$ = {}'.format(b) for b in BETA]
TITLES = [
    'Displacement in $x$ direction',
    'Displacement in $y$ direction',
    'Velocity in $x$ direction',
    "Velocity in $y$ direction"
]

SIMULATOR_SEED = 1400
NUM_LATENT = 3


def aggregate_box_plot(contamination, results_file, figsize, save_path=None):
    fig = plt.figure(figsize=figsize, dpi=300)

    for metric in ['mse']:
        if metric == 'mse':
            metric_idx = 0
            ylabel = 'RMSE'
            scale = 'linear'

        if metric == 'mse':
            normaliser = np.ones((1, NUM_LATENT))
        kalman_data, mhe_data, robust_mhe_data = pickle_load(results_file)

        kalman_data = kalman_data[:, :, metric_idx] / normaliser
        mhe_data = mhe_data[:, :, metric_idx] / normaliser
        robust_mhe_data = robust_mhe_data[:, :, :, metric_idx] / normaliser[None, ...]

        kalman_data = np.sqrt(kalman_data.mean(axis=-1))
        mhe_data = np.sqrt(mhe_data.mean(axis=-1))
        robust_mhe_data = np.sqrt(robust_mhe_data.mean(axis=-1))

        plt.yscale(scale)

        mean_data = np.zeros(2 + len(BETA), )
        mean_data[0] = kalman_data.mean(axis=0)
        mean_data[1] = mhe_data.mean(axis=0)
        mean_data[2:] = robust_mhe_data.mean(axis=0)

        plt.plot(np.arange(1, len(BETA) + 3), mean_data, color='k', lw=2, ls='dashed', marker='s', markersize=10,
                 zorder=2)

        kalman_plot = plt.boxplot(kalman_data, positions=[1], sym='x',
                                  patch_artist=True, widths=0.5, showfliers=False, zorder=1)
        mhe_plot = plt.boxplot(mhe_data, positions=[2], sym='x',
                               patch_artist=True, widths=0.5, showfliers=False, zorder=1)

        robust_mhe_plot = plt.boxplot(robust_mhe_data, positions=range(3, len(BETA) + 3),
                                      sym='x', patch_artist=True, widths=0.5, showfliers=False, zorder=1)

        kalman_plot['boxes'][0].set_facecolor('C1')
        kalman_plot['boxes'][0].set_edgecolor('black')
        kalman_plot['boxes'][0].set_alpha(0.5)

        mhe_plot['boxes'][0].set_facecolor('C2')
        mhe_plot['boxes'][0].set_edgecolor('black')
        mhe_plot['boxes'][0].set_alpha(0.5)

        for pc in robust_mhe_plot['boxes']:
            pc.set_facecolor('C3')
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)

        for element in ['medians']:
            kalman_plot[element][0].set_color('black')
            mhe_plot[element][0].set_color('black')
            [box.set_color('black') for box in robust_mhe_plot[element]]
        # plt.ylim(5, 30)
        plt.ylabel(ylabel, fontsize=30)
        plt.yticks(fontsize=30)
        plt.xticks(ticks=range(1, len(BETA) + 3),
                   labels=['UKF', 'MHE'] + BETA, fontsize=30,
                   rotation=-30)
        plt.grid(axis='y', alpha=0.2, c='k')

        colors = ['C1', 'C2', 'C3']
        labels = ['UKF', 'MHE', r'$\beta$-MHE']
        plot_patches = [patches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
        # plot_patches = plot_patches + [lines.Line2D([0], [0], color='gold', ls='-.', label='Predictive Selection')]

        # plt.legend(handles=plot_patches, loc='lower center',
        #              frameon=False, bbox_to_anchor=(0.5, -0.8), ncol=2)
        leg = plt.legend(handles=plot_patches, loc='lower center',
                         frameon=False, bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=36)
        for lh in leg.legendHandles:
            lh.set_alpha(0.5)
        plt.xlabel(r'$\beta$', fontsize=30)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')


if __name__ == '__main__':
    for contamination in CONTAMINATION:
        title = str(contamination).replace('.', '_')
        aggregate_box_plot(
            contamination=contamination,
            results_file=f'../results/robot_estimation/beta-sweep-contamination-{contamination}.pk',
            figsize=(16, 9),
            save_path=f'C:/Users/15291/beta-mhe/figures/robot_estimation/robot-{contamination}.pdf'
        )
