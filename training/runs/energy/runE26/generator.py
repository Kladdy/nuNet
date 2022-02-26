# Imports
import os
import numpy as np
import tensorflow as tf
import time
from toolbox import load_file
from constants import datapath, data_filename, label_filename, n_files, n_files_val, dataset_name, dataset_em, dataset_noise
# -------

np.set_printoptions(precision=4)

# n_files and n_files_val comes from dataset in constants.py
n_files_test = 3
norm = 1e-6

n_files_train = n_files - n_files_val - n_files_test
list_of_file_ids_train = np.arange(n_files_train, dtype=np.int)
list_of_file_ids_val = np.arange(n_files_train, n_files_train + n_files_val, dtype=np.int)
list_of_file_ids_test = np.arange(n_files_train + n_files_val, n_files, dtype=np.int)
n_events_per_file = 100000
batch_size = 64

print(f"training on {n_files_train} files ({n_files_train/n_files*100:.1f}%), validating on {n_files_val} files ({n_files_val/n_files*100:.1f}%), testing on {n_files_test} files ({n_files_test/n_files*100:.1f}%)")


# WEIGHTS - WEIGHING BY ENERGY DISTRIBUTION
MAX_WEIGHT = 10

if dataset_name == "ALVAREZ" and dataset_em == False and dataset_noise == True:
    dataset_to_use = "ALVAREZ-HAD"
    
if dataset_name == "ARZ" and dataset_em == False and dataset_noise == True:
    dataset_to_use = "ARZ-HAD"

if dataset_name == "ARZ" and dataset_em == True and dataset_noise == True:
    dataset_to_use = "ARZ-EM"
    
with open(f"weights/{dataset_to_use}_weights.npy", "rb") as f:  
    file_contents = np.load(f)
    WEIGHTING_energy_list = file_contents[:, 0]
    WEIGHTING_weight_list = file_contents[:, 1]
    WEIGHTING_count_list = file_contents[:, 2]
    
print("energies: ", WEIGHTING_energy_list)
print("weights: ", WEIGHTING_weight_list)

# https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest_and_return_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_weight_by_log10_shower_energy(log10_shower_energy):
    nearest_idx = find_nearest_and_return_index(WEIGHTING_energy_list, log10_shower_energy)

    weight_for_nearest_idx = WEIGHTING_weight_list[nearest_idx]
    if weight_for_nearest_idx > MAX_WEIGHT:
        return MAX_WEIGHT
    else:
        return weight_for_nearest_idx


class TrainDataset(tf.data.Dataset):

    def _generator(file_id):
        if((file_id + 1) == n_files_train):
#             print("reshuffling")
            np.random.shuffle(list_of_file_ids_train)

        # Opening the file
        i_file = list_of_file_ids_train[file_id]
        data, shower_energy_log10 = load_file(i_file, norm)
        num_samples = data.shape[0]
        rand_ids = np.arange(num_samples, dtype=np.int)
        np.random.shuffle(rand_ids)
        n_batches = num_samples // batch_size
        for i_batch in range(n_batches):
            # Reading data (line, record) from the file
            y = shower_energy_log10[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size]]
            x = data[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size], :, :, :]
            # calculate weights for all the targets
            sample_weights = [get_weight_by_log10_shower_energy(target)**2 for target in y]
            yield x, y, sample_weights

    def __new__(cls, file_id):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float64, tf.dtypes.float64, tf.dtypes.float64),
            output_shapes=((batch_size, 5, 512, 1), (batch_size, ), (batch_size, )),
            args=(file_id,)
        )


class ValDataset(tf.data.Dataset):

    def _generator(file_id):
        if((file_id + 1) == n_files_val):
            np.random.shuffle(list_of_file_ids_val)

        # Opening the file
        i_file = list_of_file_ids_val[file_id]
        data, shower_energy_log10 = load_file(i_file, norm)
        num_samples = data.shape[0]
        rand_ids = np.arange(num_samples, dtype=np.int)
        np.random.shuffle(rand_ids)
        n_batches = num_samples // batch_size
        for i_batch in range(n_batches):
            # Reading data (line, record) from the file
            y = shower_energy_log10[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size]]
            x = data[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size], :, :, :]
            yield x, y

    def __new__(cls, file_id):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float64, tf.dtypes.float64),
            output_shapes=((batch_size, 5, 512, 1), (batch_size, )),
            args=(file_id,)
        )

