import os
from cycler import cycler

import numpy as np
from scipy.stats import mode

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import rc, cm

from robust_smc.data import ReversibleReaction

from experiment_utilities import pickle_load

# matplotlib Global Settings
palette = cycler(color=cm.Set1.colors)
rc('axes', prop_cycle=palette)
rc('lines', lw=2)
rc('axes', lw=1.2, titlesize='large', labelsize='x-large')
rc('legend', fontsize='x-large')
rc('font', family='serif')

SIMULATOR_SEED = 1992
NOISE_STD = 0.1
FINAL_TIME = 100
TIME_STEP = 0.1
BETA = [r'$10^{-4}$', r'$2 \times 10^{-4}$']
CONTAMINATION = [0, 0.05, 0.1, 0.15, 0.2, 0.25]

LABELS = np.array(['UKF'] + ['MHE'] + [r'$\beta$ = {}'.format(b) for b in BETA])
TITLES = [
    'Displacement in $x$ direction',
    'Displacement in $y$ direction',
    'Displacement in $z$ direction',
    'Velocity in $x$ direction',
    'Velocity in $y$ direction',
    'Velocity in $z$ direction'
]

NUM_LATENT = 2


def plot_metrics(results_path, figsize, save_path=None):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=150, sharex=True)
    ax = ax.flatten()

    plt.subplots_adjust(hspace=0.05)

    for metric in ['mse', 'coverage']:
        if metric == 'mse':
            metric_idx = 0
            label = 'RMSE'
            scale = 'log'
        elif metric == 'coverage':
            metric_idx = 1
            label = '90% Empirical Coverage'
            scale = 'linear'
        else:
            raise NotImplementedError

        plot_data = []
        for contamination in CONTAMINATION:
            # predictive_scores = pickle_load(
            #     os.path.join(results_path, f'beta-predictive-sweep-contamination-{contamination}.pk')
            # )
            # best_beta = np.argmin(predictive_scores, axis=1)
            # majority_vote = mode(best_beta)
            #
            # print(majority_vote)

            simulator = ReversibleReaction(
                final_time=FINAL_TIME,
                time_step=TIME_STEP,
                observation_std=NOISE_STD,
                process_std=None,
                contamination_probability=contamination,
                seed=SIMULATOR_SEED
            )

            if metric == 'mse':
                # normaliser = (np.sum(simulator.X ** 2, axis=0) / simulator.X.shape[0])[None, :]
                normaliser = 1
            else:
                normaliser = 1

            results_file = os.path.join(results_path, f'beta-sweep-contamination-{contamination}.pk')
            ukf_data, mhe_data, robust_mhe_data = pickle_load(results_file)
            concatenated_data = np.concatenate([
                ukf_data[:, None, :, metric_idx],
                mhe_data[:, None, :, metric_idx],
                robust_mhe_data[:, :, :, metric_idx],
            ], axis=1)
            concatenated_data = concatenated_data / normaliser  # contamination x N x models x num_latent
            plot_data.append(concatenated_data)

        plot_data = np.stack(plot_data).mean(axis=-1)[:, :, :]

        # print(plot_data.shape)

        # colors = [f'C{i}' for i in range(len(BETA) + 1)] * 12 #, 'C3', 'C4', 'C0']
        colors = cm.tab20.colors
        labels = LABELS
        positions = np.arange(1, len(BETA) + 3)

        ax[metric_idx].set_yscale(scale)
        ax[metric_idx].set_ylim([10, 200.])
        for i in range(len(CONTAMINATION)):
            bplot = ax[metric_idx].boxplot(plot_data[i, :, :], positions=(i * 15) + positions,
                                           sym='x', patch_artist=True, manage_ticks=False)
            for box, m, color, l in zip(bplot['boxes'], bplot['medians'], colors, labels):
                box.set_facecolor(color)
                box.set_label(l)
                m.set_color('k')

        if metric == 'coverage':
            ax[metric_idx].set_ylim([0, 1.])
            ax[metric_idx].axhline(0.9, c='k', ls='--', alpha=0.6, lw=1.2, zorder=-1)

        ax[metric_idx].set_ylabel(label)
        ax[metric_idx].set_xticks(np.arange(5.5, 5.5 + 15 * len(CONTAMINATION), 15))
        ax[metric_idx].grid(axis='y')
        xtickNames = plt.setp(ax[metric_idx], xticklabels=CONTAMINATION)
        plt.setp(xtickNames, fontsize=12)

    ax[0].legend(handles=bplot['boxes'], loc='center right', bbox_to_anchor=(1.15, 0.0), frameon=False)

    ax[0].set_title('Reversible Reaction: aggregate metrics', fontsize=14)
    ax[0].set_ylim([0.1, 100])

    ax[-1].set_xlabel(r'Contamination probability $p_c$')
    if save_path:
        save_file = os.path.join(save_path, f'full_plots.pdf')
        plt.savefig(save_file, bbox_inches='tight')


def plot_aggregate_latent1(results_path, figsize, save_path=None):
    selected_models = [0, 1, 2, 3, 4, 5]
    colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

    labels = LABELS[selected_models]
    positions = np.arange(1, len(selected_models) + 1)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=150, sharex='all')
    axes = axes.flatten()

    plt.subplots_adjust(hspace=0.05)

    for ax, metric in zip(axes, ['mse', 'coverage']):
        if metric == 'mse':
            metric_idx = 0
            label = 'RMSE'
            scale = 'log'
        elif metric == 'coverage':
            metric_idx = 1
            label = '90% EC'
            scale = 'linear'
        else:
            raise NotImplementedError

        plot_data = []
        for contamination in CONTAMINATION:
            # predictive_scores = pickle_load(
            #     os.path.join(results_path, f'beta-predictive-sweep-contamination-{contamination}.pk')
            # )
            # best_beta = np.argmin(predictive_scores, axis=1)
            # majority_vote = mode(best_beta)

            simulator = ReversibleReaction(
                final_time=FINAL_TIME,
                time_step=TIME_STEP,
                observation_std=NOISE_STD,
                process_std=None,
                contamination_probability=contamination,
                seed=SIMULATOR_SEED
            )

            if metric == 'mse':
                # normaliser = (np.sum(simulator.X ** 2, axis=0) / simulator.X.shape[0])[None, :]
                normaliser = 1
            else:
                normaliser = 1

            results_file = os.path.join(results_path, f'beta-sweep-contamination-{contamination}.pk')
            ukf_data, mhe_data, robust_mhe_data = pickle_load(results_file)
            concatenated_data = np.concatenate([
                ukf_data[:, None, :, metric_idx],
                mhe_data[:, None, :, metric_idx],
                robust_mhe_data[:, :, :, metric_idx],
            ], axis=1)
            concatenated_data = concatenated_data / normaliser  # contamination x N x models x num_latent
            plot_data.append(concatenated_data)

        # plot_data = np.stack(plot_data)
        # plot_data = np.median(plot_data, axis=-1)
        plot_data = np.stack(plot_data).mean(axis=-1)[:, :, :]

        if metric == 'coverage':
            ax.axhline(0.9, c='k', ls='--', alpha=0.6, lw=1.2, zorder=-1)

        ax.set_yscale(scale)
        for i in range(len(CONTAMINATION)):
            bplot = ax.boxplot(plot_data[i, :, selected_models].T, positions=(i * 7) + positions,
                               sym='x', patch_artist=True, manage_ticks=False,
                               widths=0.6, flierprops={'markersize': 4})
            for box, m, color, l in zip(bplot['boxes'], bplot['medians'], colors, labels):
                box.set_facecolor(color)
                box.set_label(l)
                m.set_color('k')
        ax.set_xticks(np.arange(2.5, 2.5 + 7 * len(CONTAMINATION), 7))
        xtickNames = plt.setp(ax, xticklabels=CONTAMINATION)
        plt.setp(xtickNames, fontsize=12)
        ax.set_ylabel(label)
        ax.grid(axis='y')

    axes[0].set_ylim([0.1, 100])
    # axes[1].set_ylim([0, 0.04])
    axes[0].set_title('Reversible Reaction: aggregate metrics', fontsize=14)
    axes[-1].set_xlabel(r'Contamination probability $p_c$')
    axes[-1].legend(handles=bplot['boxes'], loc='center', bbox_to_anchor=(0.5, -0.4), frameon=False, ncol=6)
    if save_path:
        save_file = os.path.join(save_path, f'aggregate_plot.pdf')
        plt.savefig(save_file, bbox_inches='tight')


def plot_aggregate_latent_old(results_path, figsize, save_path=None):
    # selected_models = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    selected_models = [0, 1, 2, 3]
    colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

    labels = LABELS[selected_models]
    positions = np.arange(1, len(selected_models) + 1)

    fig = plt.figure(figsize=figsize, dpi=300)

    for metric in (['mse']):
        if metric == 'mse':
            metric_idx = 0
            label = 'RMSE'
            scale = 'Linear'

        plot_data = []
        for contamination in CONTAMINATION:

            simulator = ReversibleReaction(
                final_time=FINAL_TIME,
                time_step=TIME_STEP,
                observation_std=NOISE_STD,
                process_std=None,
                contamination_probability=contamination,
                seed=SIMULATOR_SEED
            )

            if metric == 'mse':
                # normaliser = (np.sum(simulator.X ** 2, axis=0) / simulator.X.shape[0])[None, :]
                normaliser = np.ones((1, NUM_LATENT))

            results_file = os.path.join(results_path, f'beta-sweep-contamination-{contamination}.pk')
            ukf_data, mhe_data, robust_mhe_data = pickle_load(results_file)
            concatenated_data = np.concatenate([
                ukf_data[:, None, :, metric_idx],
                mhe_data[:, None, :, metric_idx],
                robust_mhe_data[:, :, :, metric_idx],
            ], axis=1)
            concatenated_data = concatenated_data / normaliser  # contamination x N x models x num_latent
            plot_data.append(concatenated_data)

        plot_data = np.stack(plot_data).mean(axis=-1)[:, :, :]

        plt.yscale(scale)
        for i in range(len(CONTAMINATION)):
            # bplot = plt.boxplot(plot_data[i, :, selected_models].T, positions=(i * 7) + positions,
            #                    sym='x', patch_artist=True, manage_ticks=False,
            #                    widths=0.6, flierprops={'markersize': 4}, showfliers=False, zorder=1)
            bplot = plt.boxplot(plot_data[i, :, selected_models].T, positions=(i * 7) + positions,
                                sym='x', patch_artist=True, manage_ticks=False,
                                widths=0.6, flierprops={'markersize': 4}, showfliers=False, zorder=1)
            for box, m, color, l in zip(bplot['boxes'], bplot['medians'], colors, labels):
                box.set_facecolor(color)
                box.set_label(l)
                m.set_color('k')

            plt.plot((i * 7) + positions, plot_data[i, :, selected_models].mean(axis=1),
                     color='k', lw=1, ls='dashed', marker='s', markersize=5, zorder=2)

        plt.ylim(0, 100)
        plt.yticks(fontsize=20)
        plt.xticks(ticks=np.arange(2.5, 2.5 + 7 * len(CONTAMINATION), 7), labels=CONTAMINATION, fontsize=20)
        plt.ylabel(label, fontsize=20)
        plt.grid(axis='y')

    plt.xlabel(r'Contamination probability $p_c$', fontsize=20)
    # plt.legend(handles=bplot['boxes'], loc='center', bbox_to_anchor=(0.5, -0.4), frameon=False, ncol=2, fontsize=20)
    if save_path:
        save_file = os.path.join(save_path, f'Reactor Model.pdf')
        plt.savefig(save_file, bbox_inches='tight')


def plot_aggregate_latent(results_path, figsize, save_path=None):
    selected_models = [0, 1, 2, 3]
    colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

    labels = LABELS[selected_models]
    positions = 40 * np.arange(1, len(selected_models) + 1)

    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.2, 0.2, 0.6, 0.6])
    for metric in (['mse']):
        if metric == 'mse':
            metric_idx = 0
            label = 'RMSE'
            scale = 'Log'

        plot_data = []
        for contamination in CONTAMINATION:

            simulator = ReversibleReaction(
                final_time=FINAL_TIME,
                time_step=TIME_STEP,
                observation_std=NOISE_STD,
                process_std=None,
                contamination_probability=contamination,
                seed=SIMULATOR_SEED
            )

            if metric == 'mse':
                normaliser = np.ones((1, NUM_LATENT))

            results_file = os.path.join(results_path, f'beta-sweep-contamination-{contamination}.pk')
            ukf_data, mhe_data, robust_mhe_data = pickle_load(results_file)
            concatenated_data = np.concatenate([
                ukf_data[:, None, :, metric_idx],
                mhe_data[:, None, :, metric_idx],
                robust_mhe_data[:, :, :, metric_idx],
            ], axis=1)
            concatenated_data = concatenated_data / normaliser  # contamination x N x models x num_latent
            plot_data.append(concatenated_data)

        plot_data = np.stack(plot_data).mean(axis=-1)[:, :, :]

        plt.yscale(scale)
        for i in range(len(CONTAMINATION)):
            bplot = ax1.violinplot(plot_data[i, :, selected_models].T, positions=(i * 260) + positions, showmeans=False,
                                   showmedians=False, widths=40, vert=True, showextrema=False)

            for box, color, l in zip(bplot['bodies'], colors, labels):
                box.set_facecolor(color)
                box.set_label(l)
                box.set_edgecolor('black')
            ax1.plot((i * 260) + positions, plot_data[i, :, selected_models].mean(axis=1),
                     color='k', lw=1, ls='dashed', marker='s', markersize=3, zorder=2)

        plt.yticks(fontsize=20)
        plt.xticks(ticks=np.arange(100, 100 + 260 * len(CONTAMINATION), 260), labels=CONTAMINATION, fontsize=20)
        plt.ylabel(label, fontsize=20)
        plt.grid(axis='y')

        plt.xlabel(r'Contamination probability $p_c$', fontsize=20)
        plt.legend(handles=bplot['bodies'], loc='center', bbox_to_anchor=(1.5, 0.8), frameon=False, ncol=2, fontsize=15)
        ax_in1 = inset_axes(ax1, width="70%", height="70%", loc='center', bbox_to_anchor=(1, -0.3, 1, 1),
                            bbox_transform=ax1.transAxes)
        for i in range(len(CONTAMINATION)):
            g1_in = ax_in1.violinplot(plot_data[i, :, selected_models].T, positions=(i * 260) + positions, showmeans=False,
                                   showmedians=False, widths=40, vert=True, showextrema=False)
            for box, color, l in zip(g1_in['bodies'], colors, labels):
                box.set_facecolor(color)
                box.set_label(l)
                box.set_edgecolor('black')
            # ax_in1.get_legend().remove()
            ax_in1.plot((i * 260) + positions, plot_data[i, :, selected_models].mean(axis=1),
                 color='k', lw=1, ls='dashed', marker='s', markersize=3, zorder=2)
        # ax_in1.get_xaxis().set_visible(False)
        # ax_in1.get_yaxis().set_visible(False)
        plt.yticks(fontsize=10)
        plt.xticks(ticks=np.arange(100, 100 + 260 * len(CONTAMINATION), 260), labels=CONTAMINATION, fontsize=10)
        ax_in1.set_ylim(1, 10)
        ax_in1.set_xlim(-40.0, 1500.0)
        mark_inset(ax1, ax_in1, loc1=1, loc2=4, fc="none", ec='r', lw=1, ls='dotted', alpha=1)
        mark_inset(ax1, ax_in1, loc1=3, loc2=2, fc="none", ec='r', lw=1, ls='dotted', alpha=1)


    if save_path:
        save_file = os.path.join(save_path, f'Reactor Model.pdf')
        plt.savefig(save_file, bbox_inches='tight')


if __name__ == '__main__':
    # plot_metrics(
    #     f'../results/reversible_reaction/impulsive_noise_with_student_t/',
    #     figsize=(20, 8),
    #     save_path='../figures/reversible_reaction/impulsive_noise_with_student_t/variation_with_contamination/'
    # )
    #
    # for latent in range(6):
    #     plot_single_latent(
    #         f'./results/tan/impulsive_noise_with_student_t/',
    #         latent=latent,
    #         figsize=(8, 5),
    #         save_path='./figures/tan/impulsive_noise_with_student_t/variation_with_contamination/'
    #     )

    plot_aggregate_latent(
        f'../results/reversible_reaction/impulsive_noise_with_student_t/',
        figsize=(8, 5),
        save_path='../figures/reversible_reaction/impulsive_noise_with_student_t/variation_with_contamination/'
    )
