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

def root_git_dir():
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')

def common_dir():
    return f"{root_git_dir()}/common"

def models_dir(run_name):
    return f"{common_dir()}/models/{run_name}"

# Loading data and label files
def load_file(i_file, norm=1e-6):
    # Load 500 MHz filter
    filt = np.load(f"{common_dir()}/bandpass_filters/500MHz_filter.npy")

    #     t0 = time.time()
    #     print(f"loading file {i_file}", flush=True)
    data = np.load(os.path.join(dataset.datapath, f"{dataset.data_filename}{i_file:04d}.npy"), allow_pickle=True)
    data = np.fft.irfft(np.fft.rfft(data, axis=-1) * filt, axis=-1)
    data = data[:, :, :, np.newaxis]

    labels_tmp = np.load(os.path.join(dataset.datapath, f"{dataset.label_filename}{i_file:04d}.npy"), allow_pickle=True)
    #     print(f"finished loading file {i_file} in {time.time() - t0}s")
    shower_energy_data = np.array(labels_tmp.item()["shower_energy"])

    # check for nans and remove them
    idx = ~(np.isnan(data))
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    data = data[idx, :, :, :]
    shower_energy_data = shower_energy_data[idx]
    data /= norm

    # Get log10 of energy
    shower_energy_log10 = np.log10(shower_energy_data)

    return data, shower_energy_log10

# Loading data and label files and also other properties
def load_file_all_properties(i_file, norm=1e-6):
    t0 = time.time()
    print(f"loading file {i_file}", flush=True)

    # Load 500 MHz filter
    filt = np.load(f"{common_dir()}/bandpass_filters/500MHz_filter.npy")

    data = np.load(os.path.join(dataset.datapath, f"{dataset.data_filename}{i_file:04d}.npy"), allow_pickle=True)
    data = np.fft.irfft(np.fft.rfft(data, axis=-1) * filt, axis=-1)
    data = data[:, :, :, np.newaxis]
    
    labels_tmp = np.load(os.path.join(dataset.datapath, f"{dataset.label_filename}{i_file:04d}.npy"), allow_pickle=True)
    print(f"finished loading file {i_file} in {time.time() - t0}s")
    nu_zenith_data = np.array(labels_tmp.item()["nu_zenith"])
    nu_azimuth_data = np.array(labels_tmp.item()["nu_azimuth"])
    nu_direction_data = hp.spherical_to_cartesian(nu_zenith_data, nu_azimuth_data)

    nu_energy_data = np.array(labels_tmp.item()["nu_energy"])
    nu_flavor_data = np.array(labels_tmp.item()["nu_flavor"])
    shower_energy_data = np.array(labels_tmp.item()["shower_energy"])

    # check for nans and remove them
    idx = ~(np.isnan(data))
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    data = data[idx, :, :, :]
    data /= norm

    nu_zenith_data = nu_zenith_data[idx]
    nu_azimuth_data = nu_azimuth_data[idx]
    nu_direction_data = nu_direction_data[idx]
    nu_energy_data = nu_energy_data[idx]
    nu_flavor_data = nu_flavor_data[idx]
    shower_energy_data = shower_energy_data[idx]

    return data, nu_direction_data, nu_zenith_data, nu_azimuth_data, nu_energy_data, nu_flavor_data, shower_energy_data


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

def get_pred_energy_diff_data(run_name):
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

    return energy_difference_data

def find_68_interval(run_name):
    energy_difference_data = get_pred_energy_diff_data(run_name)

    energy_68 = calculate_percentage_interval(energy_difference_data, 0.68)

    return energy_68

