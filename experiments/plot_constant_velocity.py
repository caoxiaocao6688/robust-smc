import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib import cm, rc, patches

from experiment_utilities import pickle_load
from robust_smc.data import ConstantVelocityModel

# matplotlib Global Settings
palette = cycler(color=cm.Set1.colors)
rc('axes', prop_cycle=palette)
matplotlib.rcParams['font.family'] = ['serif']

BETA = [r'$10^{-5}$', r'$2 \times 10^{-5}$', r'$4 \times 10^{-5}$', r'$6 \times 10^{-5}$', r'$8 \times 10^{-5}$',
        r'$10^{-4}$', r'$2 \times 10^{-4}$']
CONTAMINATION = [0.2]
LABELS = ['KF', 'MHE'] + [r'$\beta$ = {}'.format(b) for b in BETA]
TITLES = [
    'Displacement in $x$ direction',
    'Displacement in $y$ direction',
    'Velocity in $x$ direction',
    "Velocity in $y$ direction"
]

SIMULATOR_SEED = 1400
NOISE_VAR = 1.0
FINAL_TIME = 100
TIME_STEP = 0.1
EXPLOSION_SCALE = 100.0
NUM_LATENT = 4
fontsize = 30

def aggregate_box_plot(contamination, results_file, figsize, save_path=None):
    fig = plt.figure(figsize=figsize, dpi=300)

    for metric in ['mse']:
        if metric == 'mse':
            metric_idx = 0
            ylabel = 'RMSE'
            scale = 'linear'

        observation_cov = NOISE_VAR * np.eye(2)
        simulator = ConstantVelocityModel(
            final_time=FINAL_TIME,
            time_step=TIME_STEP,
            observation_cov=observation_cov,
            explosion_scale=EXPLOSION_SCALE,
            contamination_probability=contamination,
            seed=SIMULATOR_SEED
        )

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

        plt_scale = 2
        plt.plot(np.arange(1*plt_scale, len(BETA)*plt_scale + 3*plt_scale, plt_scale), mean_data, color='k', lw=2, ls='dashed', marker='s', markersize=10,
                 zorder=2)

        kalman_plot = plt.boxplot(kalman_data, positions=[1*plt_scale], sym='x',
                                  patch_artist=True, widths=1, showfliers=False, zorder=1)
        mhe_plot = plt.boxplot(mhe_data, positions=[2*plt_scale], sym='x',
                               patch_artist=True, widths=1, showfliers=False, zorder=1)

        robust_mhe_plot = plt.boxplot(robust_mhe_data, positions=range(3*plt_scale, len(BETA)*plt_scale + 3*plt_scale, plt_scale),
                                      sym='x', patch_artist=True, widths=1, showfliers=False, zorder=1)

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
        plt.xlim(1, (len(BETA)+2)*plt_scale+1)
        plt.ylim(5, 30)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(ticks=range(1*plt_scale, len(BETA)*plt_scale + 3*plt_scale, plt_scale),
                   labels=['KF', 'MHE'] + BETA, fontsize=fontsize,
                   rotation=-30)
        plt.grid(axis='y', alpha=0.2, c='k')

        colors = ['C1', 'C2', 'C3']
        labels = ['KF', 'MHE', r'$\beta$-MHE']
        plot_patches = [patches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
        leg = plt.legend(handles=plot_patches, loc='lower center',
                         frameon=False, bbox_to_anchor=(0.5, -0.47), ncol=3, fontsize=fontsize)
        for lh in leg.legendHandles:
            lh.set_alpha(0.5)
        plt.xlabel(r'$\beta$', fontsize=fontsize)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')


if __name__ == '__main__':
    for contamination in CONTAMINATION:
        title = str(contamination).replace('.', '_')
        aggregate_box_plot(
            contamination=contamination,
            results_file=f'../results/constant_velocity/error_{contamination}.pk',
            figsize=(18, 9),
            save_path=f'../figures/constant_velocity/error_{title}.pdf'
        )
