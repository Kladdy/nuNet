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
parser = argparse.ArgumentParser(description='Evaluate angular resolution')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")

args = parser.parse_args()
run_id = args.run_id

# Save the run name and filename
run_name = f"run{run_id}"
filename = f"model_history_log_{run_name}.csv"

# Models folder
saved_model_dir = models_dir(run_name)

print(colored(f"Evaluating angular resolution for {run_name}...", "yellow"))

# Load the model
model = keras.models.load_model(f'{saved_model_dir}/model.{run_name}.h5')

# Load test file data and make predictions
    # Load first file
data, nu_direction = load_file(test_file_ids[0])

    # Then load rest of files
if len(test_file_ids) > 1:
    for test_file_id in test_file_ids:
        if test_file_id != test_file_ids[0]:
            data_tmp, nu_direction_tmp = load_file(test_file_id)

            data = np.concatenate((data, data_tmp))
            nu_direction = np.concatenate((nu_direction, nu_direction_tmp))

nu_direction_predict = model.predict(data)

# Print out norms
print("The following are the norms of prediction!")
normed_nu_direction = np.array([np.linalg.norm(v) for v in nu_direction_predict])
print(normed_nu_direction)

# Save predicted angles
with open(f'{saved_model_dir}/model.{run_name}.h5_predicted.pkl', "bw") as fout:
    pickle.dump([nu_direction_predict, nu_direction], fout, protocol=4)

print(colored(f"Done evaluating angular resolution for {run_name}!", "green", attrs=["bold"]))
print("")