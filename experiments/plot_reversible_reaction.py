import os
from cycler import cycler

import numpy as np
from scipy.stats import mode

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import rc, cm
import matplotlib
from robust_smc.data import ReversibleReaction

from experiment_utilities import pickle_load

# matplotlib Global Settings
palette = cycler(color=cm.Set1.colors)
rc('axes', prop_cycle=palette)
matplotlib.rcParams['font.family'] = ['serif']

SIMULATOR_SEED = 1992
NOISE_STD = 0.1
FINAL_TIME = 100
TIME_STEP = 0.1
CONTAMINATION = [0, 0.05, 0.1, 0.15, 0.2]
BETA = [1e-4, 2e-4]


def format_beta(beta_values):
    formatted_betas = []
    for beta in beta_values:
        if beta < 1e-3:
            exponent = int(np.floor(np.log10(beta)))
            coefficient = beta / (10 ** exponent)
            if coefficient == 1:
                formatted_beta = r"$10^{{{}}}$".format(exponent)
            else:
                formatted_beta = f"{coefficient:g}" + r"$\times10^{{{}}}$".format(exponent)
        else:
            formatted_beta = str(beta)
        formatted_betas.append(formatted_beta)
    return formatted_betas


BETA = format_beta(BETA)

LABELS = np.array(['UKF'] + ['MHE'] + [r'$\beta$ = {}'.format(b) for b in BETA])
NUM_LATENT = 2
fontsize = 22


# def plot_aggregate_latent(results_path, figsize, save_path=None):
#     selected_models = [0, 1, 2, 3]
#     colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
#
#     labels = LABELS[selected_models]
#     width = 2
#     spacing = 0.6
#     positions = width * np.arange(1, len(selected_models) + 1) + spacing * np.arange(len(selected_models))
#     bias = width * (len(positions) + 1)
#     shift = positions[-1] + bias / 4 + spacing
#
#     f1 = plt.figure(1, figsize=figsize)
#     ax1 = f1.add_axes([0.2, 0.2, 0.6, 0.6])
#     for metric in (['mse']):
#         if metric == 'mse':
#             metric_idx = 0
#             label = 'RMSE'
#             scale = 'log'
#
#         plot_data = []
#         for contamination in CONTAMINATION:
#             if metric == 'mse':
#                 normaliser = np.ones((1, NUM_LATENT))
#
#             results_file = os.path.join(results_path, f'error_{contamination}.pk')
#             ukf_data, mhe_data, robust_mhe_data = pickle_load(results_file)
#             concatenated_data = np.concatenate([
#                 ukf_data[:, None, :, metric_idx],
#                 mhe_data[:, None, :, metric_idx],
#                 robust_mhe_data[:, :, :, metric_idx],
#             ], axis=1)
#             concatenated_data = concatenated_data / normaliser  # contamination x N x models x num_latent
#             plot_data.append(concatenated_data)
#
#         plot_data = np.stack(plot_data).mean(axis=-1)[:, :, :]
#         plt.yscale(scale)
#         for i in range(len(CONTAMINATION)):
#             bplot = ax1.violinplot(plot_data[i, :, selected_models].T, positions=(i * shift) + positions,
#                                    showmeans=False,
#                                    showmedians=False, widths=width, vert=True, showextrema=False)
#             plt.show(block=False)
#             for box, color, l in zip(bplot['bodies'], colors, labels):
#                 box.set_facecolor(color)
#                 box.set_label(l)
#                 box.set_edgecolor('black')
#             ax1.plot((i * shift) + positions, plot_data[i, :, selected_models].mean(axis=1),
#                      color='k', lw=1, ls='dashed', marker='s', markersize=3, zorder=2)
#             plt.show(block=False)
#         plt.yticks(fontsize=fontsize)
#         plt.xticks(ticks=np.arange(positions[-1] / 2 + width / 2, shift * len(CONTAMINATION) + width / 2, shift),
#                    labels=CONTAMINATION,
#                    fontsize=fontsize)
#         plt.ylabel(label, fontsize=fontsize)
#         plt.grid(axis='y')
#
#         plt.xlabel(r'$p_c$', fontsize=fontsize)
#         plt.legend(handles=bplot['bodies'], loc='center', bbox_to_anchor=(0.45, -0.53), frameon=False, ncol=2,
#                    fontsize=fontsize, columnspacing=1)
#
#     if save_path:
#         save_file = os.path.join(save_path, f'Reactor_Model.pdf')
#         plt.savefig(save_file, bbox_inches='tight')
def plot_aggregate_latent(results_path, figsize, save_path=None):
    selected_models = [0, 1, 2, 3]
    colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    labels = LABELS[selected_models]
    width = 0.2
    positions = np.arange(1, len(selected_models) + 1) * width
    shift = positions[-1] + width * 2

    f1 = plt.figure(1, figsize=figsize)
    ax1 = f1.add_axes([0.2, 0.2, 0.6, 0.6])
    for metric in (['mse']):
        if metric == 'mse':
            metric_idx = 0
            label = 'RMSE'
            scale = 'linear'

        plot_data = []
        for contamination in CONTAMINATION:
            if metric == 'mse':
                normaliser = np.ones((1, NUM_LATENT))

            results_file = os.path.join(results_path, f'error_{contamination}.pk')
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

        for i, contamination_data in enumerate(plot_data):
            for j, model_data in enumerate(contamination_data[:, selected_models].T):
                ax1.bar(i * shift + j * width, model_data.mean(), yerr=model_data.std(), width=width, color=colors[j],
                        label=labels[j] if i == 0 else None, alpha=0.5)

        plt.yticks(ticks=[0, 4, 8, 12, 16, 20], fontsize=fontsize)
        plt.xticks(ticks=np.arange(len(CONTAMINATION)) * shift + positions[-1] / 2, labels=CONTAMINATION,
                   fontsize=fontsize)
        plt.ylabel(label, fontsize=fontsize)
        plt.grid(axis='y')
        plt.xlabel(r'$p_c$', fontsize=fontsize)
        ax1.legend(loc='center', bbox_to_anchor=(0.45, -0.46), frameon=False, ncol=2, fontsize=fontsize,
                   columnspacing=1)
        plt.ylim(0, 20)
    if save_path:
        save_file = os.path.join(save_path, f'Reactor_Model.pdf')
        plt.savefig(save_file, bbox_inches='tight')


if __name__ == '__main__':
    plot_aggregate_latent(
        f'../results/reversible_reaction/',
        figsize=(13, 5),
        save_path='../figures/reversible_reaction/'
    )
