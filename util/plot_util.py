import pickle
import numpy as np
import matplotlib.pyplot as plt

from os import makedirs, listdir
from os.path import join, splitext
from matplotlib.pyplot import figure


linestyles = {
    'our': ",",
    'join_exp': "X",
    'appind_exp': "*",
    'smooth': "+",
    'agg_tree': 5,
    'our-cdp': "p",
    'line2': "2"
}

colors = {
    'our': "lightseagreen",
    'join_exp': "mediumpurple",
    'appind_exp': "darkorange",
    'smooth': "cornflowerblue",
    'agg_tree': "violet",
    'our-cdp': "firebrick",
    'color2': "peru"
}

algo_name = {
    'our': "AQ",
    'join_exp': "JointExp",
    'appind_exp': "AppindExp",
    'smooth': "CSmooth",
    'agg_tree': "AggTree",
    'our-cdp': "AQ-zCDP"
}


def plot_times_results(result_path, save_folder):
    pth_files = [i for i in listdir(result_path) if '.pickle' in i]
    makedirs(save_folder, exist_ok=True)
    figure(figsize=(4, 3), dpi=300)
    exp_times = {}
    for k, file_name in enumerate(pth_files):
        with open(join(result_path, file_name), 'rb') as handle:
            x, results, times = pickle.load(handle)

        for label, t in times:
            exp_times[label] = np.mean(t, axis=0) + exp_times.get(label, 0)

    for label, t in exp_times.items():
        plt.title('Average run time across all datasets', fontsize=14)
        plt.xlabel('# quantiles', fontsize=12)
        plt.ylabel('time (s)', fontsize=12)
        plt.yscale('log')
        t /= len(pth_files)
        plt.plot(x, t,  label=algo_name[label], color=colors[label],
                 marker=linestyles[label], linewidth=2, markevery=29, markersize=9, alpha=0.9)

    plt.tight_layout()
    plt.figlegend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=False,
        fontsize=14)

    plt.savefig(join(save_folder, 'times.png'), dpi=300, bbox_inches='tight')


def plot_accuracy_results(result_path, save_folder, pth_files, zoom_in=None, cdp=''):
    makedirs(save_folder, exist_ok=True)
    is_label = True
    figure(figsize=(12, 6), dpi=300)
    for k, file_name in enumerate(pth_files):
        name, tile = splitext(file_name)
        with open(join(result_path, file_name), 'rb') as handle:
            x, results, times = pickle.load(handle)
        exs = plt.subplot(1, 2, k + 1)
        plt.subplot(exs)

        for label, res in results:
                m = np.mean(res, axis=0)
                plt.title(name, fontsize=14)
                plt.xlabel('# quantiles', fontsize=12)
                plt.ylabel('error per quantile', fontsize=12)
                plt.yscale('log')

                if is_label:
                    plt.plot(x[:zoom_in], m[:zoom_in], label=algo_name[label], color=colors[label],
                             marker=linestyles[label], linewidth=2, markevery=29, markersize=9, alpha=0.9)
                else:
                    plt.plot(x[:zoom_in], m[:zoom_in], color=colors[label],
                             marker=linestyles[label], linewidth=2, markevery=29, markersize=9, alpha=0.9)
                plt.yticks([1, 10, 100])
        is_label = False

    plt.figlegend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=6,
        frameon=False,
        fontsize=14)

    plt.tight_layout()
    if zoom_in:
        plt.savefig(join(save_folder, 'accuracy' + cdp +'_zoom_in.png'), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(join(save_folder, 'accuracy' + cdp + '.png'), dpi=300, bbox_inches='tight')