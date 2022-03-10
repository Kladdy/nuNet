# Imports
import os
import subprocess
import numpy as np
import time
from constants import dataset
from radiotools import helper as hp
from radiotools import stats
from NuRadioReco.utilities import units
import pickle
# -------

def common_dir():
    return f"common"

def models_dir(run_name):
    return f"{common_dir()}/models/{run_name}"

def calculate_percentage_interval(energy_difference_data, percentage=0.68):
    # Redefine N
    N = energy_difference_data.size
    weights = np.ones(N)

    # Take abs due to the fact that the energy difference can be negative
    energy = stats.quantile_1d(np.abs(energy_difference_data), weights, percentage)

    # OLD METHOD -------------------------------
    # Calculate Rayleigh fit
    # loc, scale = stats.rayleigh.fit(angle)
    # xl = np.linspace(angle.min(), angle.max(), 100) # linspace for plotting

    # Calculate 68 %
    #index_at_68 = int(0.68 * N)
    #angle_68 = np.sort(angle_difference_data)[index_at_68]
    # ------------------------------------------

    return energy

def get_pred_energy_diff_data(run_name, do_return_data=False):
    prediction_file = f'{models_dir(run_name)}/model.{run_name}.h5_predicted.pkl'
    with open(prediction_file, "br") as fin:
        shower_energy_log10_predict, shower_energy_log10 = pickle.load(fin)

    # Remove extra dimension of array (it comes from the model)
    shower_energy_log10_predict = np.squeeze(shower_energy_log10_predict)

    # Only pick first 100000 data
    # N = 100000
    # nu_direction_predict = nu_direction_predict[:N]
    # nu_direction = nu_direction[:N]
    energy_difference_data = np.array([ shower_energy_log10_predict[i] - shower_energy_log10[i] for i in range(len(shower_energy_log10))])

    if do_return_data:
        return energy_difference_data, shower_energy_log10_predict, shower_energy_log10
    else:
        return energy_difference_data

def find_68_interval(run_name):
    energy_difference_data = get_pred_energy_diff_data(run_name)

    energy_68 = calculate_percentage_interval(energy_difference_data, 0.68)

    return energy_68

