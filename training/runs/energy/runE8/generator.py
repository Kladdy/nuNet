# Imports
import os
import numpy as np
import tensorflow as tf
import time
from toolbox import load_file
from NuRadioReco.utilities import units
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelBandPassFilter
from constants import datapath, data_filename, label_filename, n_files, n_files_val
from scipy import constants as scipy_constants
from NuRadioReco.utilities import fft
# -------

n_noise_iterations = 20
n_channels = 5

np.set_printoptions(precision=4)

# n_files and n_files_val comes from dataset in constants.py
n_files_test = 3
norm = 1e-6

n_files_train = n_files - n_files_val - n_files_test
list_of_file_ids_train = np.arange(n_files_train, dtype=np.int)
list_of_file_ids_val = np.arange(n_files_train, n_files_train + n_files_val, dtype=np.int)
list_of_file_ids_test = np.arange(n_files_train + n_files_val, n_files, dtype=np.int)
n_events_per_file = 100000 * n_noise_iterations
batch_size = 64



# details of the MC data set to be able to calculate filter
sampling_rate = 2 * units.GHz
max_freq = 0.5 * sampling_rate
n_samples = 512
noise_temperature = 300
ff = np.fft.rfftfreq(n_samples, 1/sampling_rate)
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
filt = channelBandPassFilter.get_filter(ff, 0, 0, None, [0 * units.MHz, 800 * units.MHz], "butter", 10)
filt *= channelBandPassFilter.get_filter(ff, 0, 0, None, [80 * units.MHz, 1000 * units.GHz], "butter", 5)
bandwidth = np.trapz(np.abs(filt) ** 2, ff)
Vrms = (noise_temperature * 50 * scipy_constants.k * bandwidth / units.Hz) ** 0.5
noise_amplitude = Vrms / (bandwidth / (max_freq)) ** 0.5
# the unit of the data set is micro volts
noise_amplitude /= units.micro

print(f"noise temparture {noise_temperature:.0f}K, bandwidth {bandwidth/units.MHz}MHz, Vrms -> {Vrms/units.micro/units.V:.3g}muV, noise amplitude before filter {noise_amplitude:.2f}muV")

seed = np.random.randint(0, 2 ** 32 - 1)
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin(seed=seed)


print(f"training on {n_files_train} files ({n_files_train/n_files*100:.1f}%), validating on {n_files_val} files ({n_files_val/n_files*100:.1f}%), testing on {n_files_test} files ({n_files_test/n_files*100:.1f}%)")

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
            
            # create new array strucures that contain the different noise realizations for a batch of data
            tmp_shape = np.array(data.shape)
            tmp_shape[0] = batch_size * n_noise_iterations
            xx = np.zeros(tmp_shape)
            tmp_shape = np.array(shower_energy_log10.shape)
            tmp_shape[0] = batch_size * n_noise_iterations
            yy = np.zeros(tmp_shape)

            for i_event in range(batch_size):
                y = shower_energy_log10[rand_ids[i_batch * batch_size + i_event]]
                for i_noise in range(n_noise_iterations):
                    yy[i_event * n_noise_iterations + i_noise] = y
                    for i_channel in range(n_channels):
                        #print(i_event, i_noise, i_channel)
                        x = data[rand_ids[i_batch * batch_size + i_event], i_channel, :, 0]
                        noise_fft = channelGenericNoiseAdder.bandlimited_noise(0, None, n_samples, sampling_rate, noise_amplitude, 
                                                                       type='rayleigh', 
                                                                       time_domain=False, bandwidth=None)
                        tmp = x + fft.freq2time(noise_fft * filt, sampling_rate)  # apply filter to generated noise and add to noiseless trace
                        xx[i_event * n_noise_iterations + i_noise, i_channel,:,0] = tmp
            # now loop over the new xx,yy arrays and return it in chunks of batch_size
            rand_ids2 = np.arange(n_noise_iterations * batch_size, dtype=np.int)
            np.random.shuffle(rand_ids2)
            for i_noise in range(n_noise_iterations):
                xxx = xx[rand_ids2[i_noise * batch_size:(i_noise + 1) * batch_size], :, :, :]
                yyy = yy[rand_ids2[i_noise * batch_size:(i_noise + 1) * batch_size]]
                #print("xxx shape", xxx.shape)
                #print("yyy shape", yyy.shape)
                yield xxx, yyy

    def __new__(cls, file_id):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float64, tf.dtypes.float64),
            output_shapes=((batch_size, 5, 512, 1), (batch_size, )),
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
            
            # create new array strucures that contain the different noise realizations for a batch of data
            tmp_shape = np.array(data.shape)
            tmp_shape[0] = batch_size * n_noise_iterations
            xx = np.zeros(tmp_shape)
            tmp_shape = np.array(shower_energy_log10.shape)
            tmp_shape[0] = batch_size * n_noise_iterations
            yy = np.zeros(tmp_shape)

            for i_event in range(batch_size):
                y = shower_energy_log10[rand_ids[i_batch * batch_size + i_event]]
                for i_noise in range(n_noise_iterations):
                    yy[i_event * n_noise_iterations + i_noise] = y
                    for i_channel in range(n_channels):
                        #print(i_event, i_noise, i_channel)
                        x = data[rand_ids[i_batch * batch_size + i_event], i_channel, :, 0]
                        noise_fft = channelGenericNoiseAdder.bandlimited_noise(0, None, n_samples, sampling_rate, noise_amplitude, 
                                                                       type='rayleigh', 
                                                                       time_domain=False, bandwidth=None)
                        tmp = x + fft.freq2time(noise_fft * filt, sampling_rate)  # apply filter to generated noise and add to noiseless trace
                        xx[i_event * n_noise_iterations + i_noise, i_channel,:,0] = tmp
            # now loop over the new xx,yy arrays and return it in chunks of batch_size
            rand_ids2 = np.arange(n_noise_iterations * batch_size, dtype=np.int)
            np.random.shuffle(rand_ids2)
            for i_noise in range(n_noise_iterations):
                xxx = xx[rand_ids2[i_noise * batch_size:(i_noise + 1) * batch_size], :, :, :]
                yyy = yy[rand_ids2[i_noise * batch_size:(i_noise + 1) * batch_size]]
                #print("xxx shape", xxx.shape)
                #print("yyy shape", yyy.shape)
                yield xxx, yyy

    def __new__(cls, file_id):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float64, tf.dtypes.float64),
            output_shapes=((batch_size, 5, 512, 1), (batch_size, )),
            args=(file_id,)
        )

