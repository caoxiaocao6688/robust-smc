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


# def plot(results_file, nrows, ncols, figsize, metric='mse', save_path=None):
#     if metric == 'mse':
#         metric_idx = 0
#         ylabel = 'MSE'
#         scale = 'log'
#     elif metric == 'coverage':
#         metric_idx = 1
#         ylabel = '90% EC'
#         scale = 'linear'
#     else:
#         raise NotImplementedError
#     kalman_data, mhe_data, vanilla_bpf_data, robust_mhe_data = pickle_load(results_file)
#     fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=150, sharex=True)
#     ax = ax.flatten()
#     for var in range(NUM_LATENT):
#         ax[var].set_yscale(scale)
#         boxes = [mhe_data[:, var, metric_idx], kalman_data[:, var, metric_idx], vanilla_bpf_data[:, var, metric_idx]] \
#                 + [robust_mhe_data[:, i, var, metric_idx] for i in range(len(BETA))]
#         ax[var].boxplot(boxes)
#         ax[var].set_title(TITLES[var])
#         ax[var].set_ylabel(ylabel)
#         xtickNames = plt.setp(ax[var], xticklabels=LABELS)
#         plt.setp(xtickNames, fontsize=12)
#
#     if save_path:
#         plt.savefig(save_path)


# def violin_plot(contamination, results_file, nrows, ncols, figsize, metric='mse', save_path=None):
#     if metric == 'mse':
#         metric_idx = 0
#         ylabel = 'NMSE'
#         scale = 'log'
#     elif metric == 'coverage':
#         metric_idx = 1
#         ylabel = '90% Empirical Coverage'
#         scale = 'linear'
#     else:
#         raise NotImplementedError
#
#     observation_cov = NOISE_VAR * np.eye(2)
#     simulator = ConstantVelocityModel(
#         final_time=FINAL_TIME,
#         time_step=TIME_STEP,
#         observation_cov=observation_cov,
#         explosion_scale=EXPLOSION_SCALE,
#         contamination_probability=contamination,
#         seed=SIMULATOR_SEED
#     )
#
#     if metric == 'mse':
#         normaliser = (np.sum(simulator.X ** 2, axis=0) / simulator.X.shape[0])[None, :]
#     else:
#         normaliser = np.ones((1, NUM_LATENT))
#
#     kalman_data, mhe_data, vanilla_bpf_data, robust_mhe_data = pickle_load(results_file)
#
#     kalman_data = kalman_data[:, :, metric_idx] / normaliser
#     mhe_data = mhe_data[:, :, metric_idx] / normaliser
#     vanilla_bpf_data = vanilla_bpf_data[:, :, metric_idx] / normaliser
#     robust_mhe_data = robust_mhe_data[:, :, :, metric_idx] / normaliser[None, ...]
#
#     fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=150, sharex=True)
#     ax = ax.flatten()
#     points = 25
#     for var in range(NUM_LATENT):
#         ax[var].set_yscale(scale)
#         mhe_plot = ax[var].violinplot(dataset=mhe_data[:, var], points=points,
#                                          showmedians=True, positions=[1])
#         kalman_plot = ax[var].violinplot(dataset=kalman_data[:, var], points=points,
#                                          showmedians=True, positions=[2])
#         bpf_plot = ax[var].violinplot(dataset=vanilla_bpf_data[:, var], points=points,
#                                       showmedians=True, positions=[3])
#         robust_mhe_plot = ax[var].violinplot(dataset=robust_mhe_data[:, :, var], points=points,
#                                              showmedians=True, positions=range(4, len(BETA) + 4))
#
#         kalman_plot['bodies'][0].set_facecolor('C2')
#         kalman_plot['bodies'][0].set_edgecolor('black')
#         kalman_plot['bodies'][0].set_alpha(1)
#
#         mhe_plot['bodies'][0].set_facecolor('C3')
#         mhe_plot['bodies'][0].set_edgecolor('black')
#         mhe_plot['bodies'][0].set_alpha(1)
#
#         bpf_plot['bodies'][0].set_facecolor('C1')
#         bpf_plot['bodies'][0].set_edgecolor('black')
#         bpf_plot['bodies'][0].set_alpha(1)
#
#         for pc in robust_mhe_plot['bodies']:
#             pc.set_facecolor('C0')
#             pc.set_edgecolor('black')
#             pc.set_alpha(1)
#
#         for element in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
#             kalman_plot[element].set_color('black')
#             mhe_plot[element].set_color('black')
#             bpf_plot[element].set_color('black')
#             robust_mhe_plot[element].set_color('black')
#
#         ax[var].set_title(TITLES[var])
#         ax[var].set_ylabel(ylabel)
#         ax[var].set_xticks(range(1, len(BETA) + 3))
#         xtickNames = plt.setp(ax[var], xticklabels=LABELS)
#         plt.setp(xtickNames, fontsize=12)
#
#         colors = ['C3', 'C2', 'C1', 'C0']
#         labels = ['MHE', 'Kalman Filter', 'BPF', r'$\beta$-BPF']
#         plot_patches = [patches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
#
#         ax[var].legend(handles=plot_patches, loc='center right',
#                        bbox_to_anchor=(1.15, 0.5), frameon=False)
#         ax[var].grid(axis='y')
#
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')
#
#
# def individual_violin_plot(contamination, state, results_file, figsize, save_path=None):
#
#     fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=150, sharex=True)
#     plt.subplots_adjust(hspace=0.05)
#     ax = ax.flatten()
#     points = 25
#
#     for metric in ['mse', 'coverage']:
#         if metric == 'mse':
#             metric_idx = 0
#             ylabel = 'NMSE'
#             scale = 'log'
#         elif metric == 'coverage':
#             metric_idx = 1
#             ylabel = '90% EC'
#             scale = 'linear'
#         else:
#             raise NotImplementedError
#
#         observation_cov = NOISE_VAR * np.eye(2)
#         simulator = ConstantVelocityModel(
#             final_time=FINAL_TIME,
#             time_step=TIME_STEP,
#             observation_cov=observation_cov,
#             explosion_scale=EXPLOSION_SCALE,
#             contamination_probability=contamination,
#             seed=SIMULATOR_SEED
#         )
#
#         if metric == 'mse':
#             normaliser = (np.sum(simulator.X ** 2, axis=0) / simulator.X.shape[0])[None, :]
#         else:
#             normaliser = np.ones((1, NUM_LATENT))
#
#         kalman_data, mhe_data, vanilla_bpf_data, robust_mhe_data = pickle_load(results_file)
#
#         kalman_data = kalman_data[:, :, metric_idx] / normaliser
#         mhe_data = mhe_data[:, :, metric_idx] / normaliser
#         vanilla_bpf_data = vanilla_bpf_data[:, :, metric_idx] / normaliser
#         robust_mhe_data = robust_mhe_data[:, :, :, metric_idx] / normaliser[None, ...]
#
#         ax[metric_idx].set_yscale(scale)
#         mhe_plot = ax[metric_idx].violinplot(dataset=mhe_data[:, state], points=points,
#                                              showmedians=True, positions=[1], showextrema=False)
#         kalman_plot = ax[metric_idx].violinplot(dataset=kalman_data[:, state], points=points,
#                                          showmedians=True, positions=[2], showextrema=False)
#         bpf_plot = ax[metric_idx].violinplot(dataset=vanilla_bpf_data[:, state], points=points,
#                                       showmedians=True, positions=[3], showextrema=False)
#         robust_mhe_plot = ax[metric_idx].violinplot(dataset=robust_mhe_data[:, :, state], points=points,
#                                              showmedians=True, positions=range(4, len(BETA) + 4), showextrema=False)
#
#         if metric == 'coverage':
#             ax[metric_idx].axhline(0.9, c='k', ls='--', alpha=0.6, lw=1.2, zorder=-1)
#
#         kalman_plot['bodies'][0].set_facecolor('C2')
#         kalman_plot['bodies'][0].set_edgecolor('black')
#         kalman_plot['bodies'][0].set_alpha(1)
#
#         mhe_plot['bodies'][0].set_facecolor('C3')
#         mhe_plot['bodies'][0].set_edgecolor('black')
#         mhe_plot['bodies'][0].set_alpha(1)
#
#         bpf_plot['bodies'][0].set_facecolor('C1')
#         bpf_plot['bodies'][0].set_edgecolor('black')
#         bpf_plot['bodies'][0].set_alpha(1)
#
#         for pc in robust_mhe_plot['bodies']:
#             pc.set_facecolor('C0')
#             pc.set_edgecolor('black')
#             pc.set_alpha(1)
#
#         # for element in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
#         for element in ['cmedians']:
#             kalman_plot[element].set_color('black')
#             mhe_plot[element].set_color('black')
#             bpf_plot[element].set_color('black')
#             robust_mhe_plot[element].set_color('black')
#
#         ax[metric_idx].set_ylabel(ylabel)
#         ax[metric_idx].set_xticks(range(1, len(BETA) + 3))
#         xtickNames = plt.setp(ax[metric_idx], xticklabels=['', ''] + BETA)
#         plt.setp(xtickNames, fontsize=12, rotation=-45)
#
#         ax[metric_idx].grid(axis='y')
#
#     colors = ['C3', 'C2', 'C1', 'C0']
#     labels = ['mhe_filter', 'Kalman Filter', 'BPF', r'$\beta$-BPF']
#     plot_patches = [patches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
#
#     ax[0].legend(handles=plot_patches, loc='upper center', frameon=False)
#     ax[1].set_xlabel(r'$\beta$')
#     ax[0].set_title(TITLES[state], fontsize=18)
#
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')
#
#
# def individual_box_plot(contamination, state, results_file, figsize, save_path=None):
#
#     fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=150, sharex=True)
#     plt.subplots_adjust(hspace=0.05)
#     ax = ax.flatten()
#
#     for metric in ['mse', 'coverage']:
#         if metric == 'mse':
#             metric_idx = 0
#             ylabel = 'NMSE'
#             scale = 'log'
#         elif metric == 'coverage':
#             metric_idx = 1
#             ylabel = '90% EC'
#             scale = 'linear'
#         else:
#             raise NotImplementedError
#
#         observation_cov = NOISE_VAR * np.eye(2)
#         simulator = ConstantVelocityModel(
#             final_time=FINAL_TIME,
#             time_step=TIME_STEP,
#             observation_cov=observation_cov,
#             explosion_scale=EXPLOSION_SCALE,
#             contamination_probability=contamination,
#             seed=SIMULATOR_SEED
#         )
#
#         if metric == 'mse':
#             normaliser = (np.sum(simulator.X ** 2, axis=0) / simulator.X.shape[0])[None, :]
#         else:
#             normaliser = np.ones((1, NUM_LATENT))
#
#         kalman_data, mhe_data, vanilla_bpf_data, robust_mhe_data = pickle_load(results_file)
#
#         kalman_data = kalman_data[:, :, metric_idx] / normaliser
#         mhe_data = mhe_data[:, :, metric_idx] / normaliser
#         vanilla_bpf_data = vanilla_bpf_data[:, :, metric_idx] / normaliser
#         robust_mhe_data = robust_mhe_data[:, :, :, metric_idx] / normaliser[None, ...]
#
#         ax[metric_idx].set_yscale(scale)
#         mhe_plot = ax[metric_idx].boxplot(mhe_data[:, state], positions=[1], sym='x',
#                                              patch_artist=True, widths=0.5)
#         kalman_plot = ax[metric_idx].boxplot(kalman_data[:, state], positions=[2], sym='x',
#                                              patch_artist=True, widths=0.5)
#         bpf_plot = ax[metric_idx].boxplot(vanilla_bpf_data[:, state], positions=[3], sym='x',
#                                           patch_artist=True, widths=0.5)
#         robust_mhe_plot = ax[metric_idx].boxplot(robust_mhe_data[:, :, state], positions=range(4, len(BETA) + 4),
#                                                  sym='x', patch_artist=True, widths=0.5)
#
#         if metric == 'coverage':
#             ax[metric_idx].axhline(0.9, c='k', ls='--', alpha=0.6, lw=1.2, zorder=-1)
#
#         kalman_plot['boxes'][0].set_facecolor('C2')
#         kalman_plot['boxes'][0].set_edgecolor('black')
#         kalman_plot['boxes'][0].set_alpha(1)
#
#         mhe_plot['boxes'][0].set_facecolor('C3')
#         mhe_plot['boxes'][0].set_edgecolor('black')
#         mhe_plot['boxes'][0].set_alpha(1)
#
#         bpf_plot['boxes'][0].set_facecolor('C1')
#         bpf_plot['boxes'][0].set_edgecolor('black')
#         bpf_plot['boxes'][0].set_alpha(1)
#
#         for pc in robust_mhe_plot['boxes']:
#             pc.set_facecolor('C0')
#             pc.set_edgecolor('black')
#             pc.set_alpha(1)
#
#         for element in ['medians']:
#             kalman_plot[element][0].set_color('black')
#             mhe_plot[element][0].set_color('black')
#             bpf_plot[element][0].set_color('black')
#             [box.set_color('black') for box in robust_mhe_plot[element]]
#
#         ax[metric_idx].set_ylabel(ylabel)
#         ax[metric_idx].set_xticks(range(1, len(BETA) + 3))
#         xtickNames = plt.setp(ax[metric_idx], xticklabels=['', ''] + BETA)
#         plt.setp(xtickNames, fontsize=12, rotation=-45)
#
#         ax[metric_idx].grid(axis='y')
#
#     colors = ['C3', 'C2', 'C1', 'C0']
#     labels = ['MHE', 'Kalman Filter', 'BPF', r'$\beta$-BPF']
#     plot_patches = [patches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
#
#     ax[0].legend(handles=plot_patches, loc='upper center', frameon=False)
#     ax[1].set_xlabel(r'$\beta$')
#     ax[0].set_title(TITLES[state], fontsize=18)
#
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')


def aggregate_box_plot1(contamination, results_file, figsize, save_path=None):
    # predictive_scores = pickle_load(
    #     f'../results/constant-velocity/impulsive_noise_predictive/beta-sweep-contamination-{contamination}.pk'
    # )

    # best_beta = np.argmin(predictive_scores, axis=1)

    # majority_vote = mode(best_beta)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=150, sharex=True)
    plt.subplots_adjust(hspace=0.05)
    ax = ax.flatten()

    for metric in ['mse', 'coverage']:
        if metric == 'mse':
            metric_idx = 0
            ylabel = 'RMSE'
            scale = 'log'
        elif metric == 'coverage':
            metric_idx = 1
            ylabel = '90% EC'
            scale = 'linear'
        else:
            raise NotImplementedError

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
            # normaliser = (np.sum(simulator.X ** 2, axis=0) / simulator.X.shape[0])[None, :]
            normaliser = np.ones((1, NUM_LATENT))
        else:
            normaliser = np.ones((1, NUM_LATENT))

        kalman_data, mhe_data, robust_mhe_data = pickle_load(results_file)

        kalman_data = kalman_data[:, :, metric_idx] / normaliser
        mhe_data = mhe_data[:, :, metric_idx] / normaliser
        robust_mhe_data = robust_mhe_data[:, :, :, metric_idx] / normaliser[None, ...]

        kalman_data = np.sqrt(kalman_data.mean(axis=-1))
        mhe_data = np.sqrt(mhe_data.mean(axis=-1))
        robust_mhe_data = np.sqrt(robust_mhe_data.mean(axis=-1))

        ax[metric_idx].set_yscale(scale)
        mhe_plot = ax[metric_idx].boxplot(mhe_data, positions=[1], sym='x',
                                          patch_artist=True, widths=0.5)
        kalman_plot = ax[metric_idx].boxplot(kalman_data, positions=[2], sym='x',
                                             patch_artist=True, widths=0.5)
        robust_mhe_plot = ax[metric_idx].boxplot(robust_mhe_data, positions=range(3, len(BETA) + 3),
                                                 sym='x', patch_artist=True, widths=0.5)

        # ax[metric_idx].axvline(2 + majority_vote.mode, color='gold',
        #                        ls='-.', zorder=0, label='Predictive Selection')

        if metric == 'coverage':
            ax[metric_idx].axhline(0.9, c='k', ls='--', alpha=0.6, lw=1.2, zorder=-1)

        kalman_plot['boxes'][0].set_facecolor('C2')
        kalman_plot['boxes'][0].set_edgecolor('black')
        kalman_plot['boxes'][0].set_alpha(1)

        mhe_plot['boxes'][0].set_facecolor('C3')
        mhe_plot['boxes'][0].set_edgecolor('black')
        mhe_plot['boxes'][0].set_alpha(1)

        for pc in robust_mhe_plot['boxes']:
            pc.set_facecolor('C1')
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        for element in ['medians']:
            kalman_plot[element][0].set_color('black')
            mhe_plot[element][0].set_color('black')
            [box.set_color('black') for box in robust_mhe_plot[element]]

        ax[metric_idx].set_ylabel(ylabel)
        ax[metric_idx].set_xticks(range(1, len(BETA) + 3))
        xtickNames = plt.setp(ax[metric_idx], xticklabels=['', ''] + BETA)
        plt.setp(xtickNames, fontsize=12, rotation=-30)

        ax[metric_idx].grid(axis='y')

    colors = ['C3', 'C2', 'C1']
    labels = ['MHE', 'Kalman Filter', r'$\beta$-MHE']
    plot_patches = [patches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    # plot_patches = plot_patches + [lines.Line2D([0], [0], color='gold', ls='-.', label='Predictive Selection')]

    ax[1].legend(handles=plot_patches, loc='lower center',
                 frameon=False, bbox_to_anchor=(0.5, -0.8), ncol=2)
    ax[1].set_xlabel(r'$\beta$')
    ax[0].set_title(
        r'Wiener velocity: aggregate metrics for $p_c = {}$'.format(contamination),
        fontsize=14
    )

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def aggregate_box_plot(contamination, results_file, figsize, save_path=None):
    fig = plt.figure(figsize=figsize, dpi=300)

    for metric in ['mse']:
        if metric == 'mse':
            metric_idx = 0
            ylabel = 'RMSE'
            scale = 'Linear'

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
        plt.ylim(5, 30)
        plt.ylabel(ylabel, fontsize=30)
        plt.yticks(fontsize=30)
        plt.xticks(ticks=range(1, len(BETA) + 3),
                   labels=['KF', 'MHE'] + BETA, fontsize=30,
                   rotation=-30)
        plt.grid(axis='y', alpha=0.2, c='k')

        colors = ['C1', 'C2', 'C3']
        labels = ['KF', 'MHE', r'$\beta$-MHE']
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
    # for metric in ['mse', 'coverage']:
    #     for contamination in CONTAMINATION:
    #         # plot(
    #         #     f'./results/constant-velocity/impulsive_noise/beta-sweep-contamination-{contamination}.pk',
    #         #     nrows=4,
    #         #     ncols=1,
    #         #     figsize=(20, 14),
    #         #     metric=metric,
    #         #     save_path=f'./figures/constant-velocity/impulsive_noise/{metric}/beta-sweep-contamination-{contamination}.pdf'
    #         # )
    #
    #         violin_plot(
    #             contamination=contamination,
    #             results_file=f'./results/constant-velocity/impulsive_noise/beta-sweep-contamination-{contamination}.pk',
    #             nrows=4,
    #             ncols=1,
    #             figsize=(20, 14),
    #             metric=metric,
    #             save_path=f'./figures/constant-velocity/impulsive_noise/{metric}/violin_beta-sweep-contamination-{contamination}.pdf'
    #         )

    # for state in range(NUM_LATENT):
    #     # individual_violin_plot(
    #     #     contamination=0.1,
    #     #     state=state,
    #     #     results_file=f'./results/constant-velocity/impulsive_noise/beta-sweep-contamination-{contamination}.pk',
    #     #     figsize=(8, 5),
    #     #     save_path=f'./figures/constant-velocity/impulsive_noise/latents/violin_latent_{state}.pdf'
    #     # )
    #     individual_box_plot(
    #         contamination=0.1,
    #         state=state,
    #         results_file=f'./results/constant-velocity/impulsive_noise/beta-sweep-contamination-{contamination}.pk',
    #         figsize=(8, 5),
    #         save_path=f'./figures/constant-velocity/impulsive_noise/latents/boxplot_latent_{state}.pdf'
    #     )

    for contamination in CONTAMINATION:
        title = str(contamination).replace('.', '_')
        aggregate_box_plot(
            contamination=contamination,
            results_file=f'../results/constant-velocity/impulsive_noise/beta-sweep-contamination-{contamination}.pk',
            figsize=(16, 9),
            save_path=f'../figures/constant-velocity/impulsive_noise/latents/boxplot_aggregate_{title}.pdf'
        )
