import multiprocessing as mp

from algorithms.approximate_quantiles_algo import approximate_quantiles_algo
from algorithms.baseline_algos import run_agg_tree, run_appind_exp, run_joint_exp
from util.data_reader import uniform_data_reader, gaussian_data_reader
from util.experiments_util import run_and_save
from util.plot_util import plot_accuracy_results, plot_times_results


def main():
    algos = [('agg_tree', run_agg_tree),
             ('appind_exp', run_appind_exp),
             ('join_exp', run_joint_exp),
             ('our', approximate_quantiles_algo)]

    data_size = 1000
    output_folder = 'results\dp'

    datasets = [uniform_data_reader, gaussian_data_reader]

    process = []
    for data in datasets:
        p = mp.Process(target=run_and_save, args=(data, algos, output_folder, data_size))
        process.append(p)
        p.start()

    for p in process:
        p.join()

    output_folder = 'results\dp'
    pth_files = ['Uniform.pickle', 'Gaussian.pickle']
    plot_accuracy_results(output_folder, 'plots/paper', pth_files)
    plot_accuracy_results(output_folder, 'plots/paper', pth_files, zoom_in=35)
    plot_times_results(output_folder, 'plots/paper')


if __name__ == '__main__':
    main()
