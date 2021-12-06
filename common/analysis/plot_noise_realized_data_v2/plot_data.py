from toolbox import load_file
from constants import dataset_noise


all_data, _ = load_file(0)

some_data = all_data[0:10]


from matplotlib import pyplot as plt
import os

NOISE_REALIZATION = 0

plots_dir = "/home/sstjaernholm/nuNet/common/analysis/plot_noise_realized_data_v2/plots"
if not os.path.isdir(plots_dir):
    os.mkdir(plots_dir)
# os.mkdir(plots_dir)

for idx, data in enumerate(some_data):
    
    idx_true = idx

    idx_true_plots_dir = f"{plots_dir}/{idx_true}_plots"
    if not os.path.isdir(idx_true_plots_dir):
        os.mkdir(idx_true_plots_dir)

    # Getting x axis (1 step is 0.5 ns)
    x_axis_double = range(int(len(data[0])))
    x_axis = [float(x)/2 for x in x_axis_double]

    # Plotting
    fig, axs = plt.subplots(5)
    fig.suptitle(f'Plot of 4 LPDA & 1 dipole of SouthPole data for event {idx_true} in file {0} for {"noisy" if dataset_noise else f"noise realized (idx {NOISE_REALIZATION})"}')

    for i in range(5):
        axs[i].plot(x_axis, data[i])
        axs[i].set_xlim([min(x_axis), max(x_axis)])
        if i != 4:
            axs[i].set_title(f'LPDA {i+1}')

    axs[4].set_title('Dipole')

    for ax in axs.flat:
        ax.set(xlabel='time (ns)', ylabel=f'signal (μV)')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig.set_size_inches(12, 10)

    plt.savefig(f"{idx_true_plots_dir}/signal_file{0}_event{idx_true}{'' if dataset_noise else f'_realization{NOISE_REALIZATION}'}.png")

    fig.clear()
    plt.close(fig)
