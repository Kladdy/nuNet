# GPU allocation
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
# --------------

# Imports
import numpy as np
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
from tensorflow import keras
import os
import time
from radiotools import helper as hp
from NuRadioReco.utilities import units
import pickle
import argparse
from termcolor import colored
from toolbox import load_file, models_dir
from constants import datapath, data_filename, label_filename, test_file_ids
# -------

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate energy resolution')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")

args = parser.parse_args()
run_id = args.run_id

# Save the run name and filename
run_name = f"run{run_id}"
filename = f"model_history_log_{run_name}.csv"

# Models folder
saved_model_dir = models_dir(run_name)

print(colored(f"Evaluating energy resolution for {run_name}...", "yellow"))

if run_id == "E12.1":
    with open(f"shift_files/ALVAREZ-HAD_energy_difference_means.npy", "rb") as f:
        nu_energy_bins = np.load(f)
        binned_resolution_nu_energy = np.load(f)
elif run_id == "E9.1":
    with open(f"shift_files/ARZ-HAD_energy_difference_means.npy", "rb") as f:
        nu_energy_bins = np.load(f)
        binned_resolution_nu_energy = np.load(f)
elif run_id == "E13.1":
    with open(f"shift_files/ARZ-EM_energy_difference_means.npy", "rb") as f:
        nu_energy_bins = np.load(f)
        binned_resolution_nu_energy = np.load(f)

print(nu_energy_bins)
print(binned_resolution_nu_energy)

mask = np.isnan(binned_resolution_nu_energy)
nu_energy_bins = nu_energy_bins[~mask]
binned_resolution_nu_energy = binned_resolution_nu_energy[~mask]

print(nu_energy_bins)
print(binned_resolution_nu_energy)

# Get log10 of energy in bins
nu_energy_bins = np.log10(nu_energy_bins)

# https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest_and_return_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_bias_by_log10_shower_energy(log10_shower_energy):
    nearest_idx = find_nearest_and_return_index(nu_energy_bins, log10_shower_energy)

    bias_for_nearest_idx = binned_resolution_nu_energy[nearest_idx]
    return bias_for_nearest_idx




# Load the model
model = keras.models.load_model(f'{saved_model_dir}/model.{run_name}.h5')

# Load test file data and make predictions
    # Load first file
data, shower_energy_log10 = load_file(test_file_ids[0])

    # Then load rest of files
if len(test_file_ids) > 1:
    for test_file_id in test_file_ids:
        if test_file_id != test_file_ids[0]:
            data_tmp, shower_energy_log10_tmp = load_file(test_file_id)

            data = np.concatenate((data, data_tmp))
            shower_energy_log10 = np.concatenate((shower_energy_log10, shower_energy_log10_tmp))

shower_energy_log10_predict = model.predict(data)


# Shifting all values
print("Shifting all values...")
print(shower_energy_log10_predict.shape)
print(shower_energy_log10_predict)
shower_energy_log10_predict = np.array([shower_energy_log10_predict[i] - get_bias_by_log10_shower_energy(shower_energy_log10_predict[i]) for i in range(len(shower_energy_log10_predict))])
print(shower_energy_log10_predict.shape)
print(shower_energy_log10_predict)
print("Done shifting! :-)")

# Save predicted angles
with open(f'{saved_model_dir}/ENERGY_SHIFTET.model.{run_name}.h5_predicted.pkl', "bw") as fout:
    pickle.dump([shower_energy_log10_predict, shower_energy_log10], fout, protocol=4)

print(colored(f"Done evaluating energy resolution for {run_name}!", "green", attrs=["bold"]))
print("")