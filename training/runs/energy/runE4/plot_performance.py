# Imports
import numpy as np
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
import os
import time
import pickle
from scipy import stats
from radiotools import helper as hp
from NuRadioReco.utilities import units
from toolbox import load_file, calculate_percentage_interval, get_pred_energy_diff_data, models_dir
import argparse
from termcolor import colored
from constants import datapath, data_filename, label_filename, plots_dir
# -------

# Parse arguments
parser = argparse.ArgumentParser(description='Plot energy resolution')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")

args = parser.parse_args()
run_id = args.run_id

# Save the run name and filename
run_name = f"run{run_id}"
filename = f"model_history_log_{run_name}.csv"

print(colored(f"Plotting energy resolution for {run_name}...", "yellow"))

# Make sure plots folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Models folder
saved_model_dir = models_dir(run_name)

# Make sure predicted file exists, otherwise run evaluator
prediction_file = f'{saved_model_dir}/model.{run_name}.h5_predicted.pkl'
if not os.path.isfile(prediction_file):
    print("Prediction file does not exist, running evaluator...")
    os.system(f"python evaluator.py {run_id}")

# Get angle difference data
energy_difference_data = get_pred_energy_diff_data(run_name)

# Redefine N
N = energy_difference_data.size

# Calculate 68 %
energy_68 = calculate_percentage_interval(energy_difference_data, 0.68)

delta_log_E_string = r"$\Delta(\log_{10}\:E)$"
# fig, ax = php.get_histogram(predicted_nu_energy[:, 0], bins=np.arange(17, 20.1, 0.05), xlabel="predicted energy")
fig, ax = php.get_histogram(energy_difference_data, bins=np.linspace(-1.5, 1.5, 90),
                            xlabel=delta_log_E_string)
# ax.plot(xl, N*stats.rayleigh(scale=scale, loc=loc).pdf(xl))
plt.title(f"Energy resolution for {run_name} with\n68 % interval of {delta_log_E_string} at {energy_68:.2f}")
fig.savefig(f"{plots_dir}/energy_resolution_{run_name}.png")

# # plt.show()

# SNR_bins = np.append(np.arange(1, 20, 1), [10000])
# SNR_means = np.arange(1.5, 20.5, 1)

# mean = stats.binned_statistic(max_LPDA[:, 0] / 10., angle_difference_data, bins=SNR_bins)[0]
# std = stats.binned_statistic(max_LPDA[:, 0] / 10., angle_difference_data, bins=SNR_bins, statistic='std')[0]
# fig, ax = plt.subplots(1, 1)
# # ax.errorbar(SNR_means, mean, yerr=std, fmt="o")
# ax.plot(SNR_means, mean, "o")
# # ax.set_ylim(0, 0.4)
# ax.set_xlabel("max SNR LPDA")
# ax.set_ylabel("angular resolution")
# fig.tight_layout()
# fig.savefig(f"{plots_dir}/mean_maxSNRLPDA_{run_name}.png")
# # plt.show()

print(colored(f"Saved energy resolution for {run_name}!", "green", attrs=["bold"]))
print("")