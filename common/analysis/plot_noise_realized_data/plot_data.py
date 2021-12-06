from constants import dataset_noise
from generator import TrainDataset
import tensorflow as tf


if dataset_noise:
    from generator_NO_NOISE_REALIZATIONS import TrainDataset
    n_noise_iterations = 1
else: 
    from generator import TrainDataset, n_noise_iterations

# Get dataset
dataset_train = tf.data.Dataset.range(1).prefetch(1).interleave(TrainDataset, deterministic=True)

# print(dataset_train.take(1))

# # for data, labels in dataset_train.take(1):  # only take first element of dataset
for data, labels in dataset_train:
    numpy_data = data.numpy()
    numpy_labels = labels.numpy()

    break

print("Dnne!")

from matplotlib import pyplot as plt
import os

NOISE_REALIZATION = 2

plots_dir = "/home/sstjaernholm/nuNet/common/analysis/plot_noise_realized_data/plots"
# os.mkdir(plots_dir)

for idx, data in enumerate(numpy_data):
    
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

    plt.savefig(f"{idx_true_plots_dir}/signal_file{0}_event{idx_true}{'' if dataset_noise else f'_realization{idx + NOISE_REALIZATION}'}.png")

    fig.clear()
    plt.close(fig)
