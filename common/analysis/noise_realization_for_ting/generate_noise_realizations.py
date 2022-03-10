import os
import numpy as np
#import tensorflow as tf
import time
#from toolbox import load_file
from constants import datapath, n_files, n_files_val, dataset, dataset_name, dataset_em
import datasets
import argparse
from NuRadioReco.utilities import units
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelBandPassFilter
from scipy import constants
from radiotools import helper as hp
from NuRadioReco.utilities import units
import pickle
from NuRadioReco.utilities import fft
import logging
from toolbox import common_dir
logger = logging.getLogger()                                                                                                                       
logger.setLevel(logging.WARNING) 

np.set_printoptions(precision=4)

n_files_test = 3
norm = 1e-6
n_files_train = n_files - n_files_val - n_files_test
list_of_file_ids_train = np.arange(n_files_train, dtype=np.int)
list_of_file_ids_val = np.arange(n_files_train, n_files_train + n_files_val, dtype=np.int)
list_of_file_ids_test = np.arange(n_files_train + n_files_val, n_files, dtype=np.int)
n_events_per_file = 100000
batch_size = 64

print(f"training on {n_files_train} files ({n_files_train/n_files*100:.1f}%), validating on {n_files_val} files ({n_files_val/n_files*100:.1f}%), testing on {n_files_test} files ({n_files_test/n_files*100:.1f}%)")
steps_per_epoch = n_files_train * (n_events_per_file // batch_size)
n_batches_per_file = n_events_per_file // batch_size
print(f"steps_per_epoch {steps_per_epoch}, n_batches_per_file {n_batches_per_file}")
n_noise_iterations = 10
n_channels = 5

# details of the MC data set to be able to calculate filter
sampling_rate = 2 * units.GHz
max_freq = 0.5 * sampling_rate
n_samples = 512
noise_temperature = 300
ff = np.fft.rfftfreq(n_samples, 1/sampling_rate)
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
# filt = channelBandPassFilter.get_filter(ff, 0, 0, None, [0 * units.MHz, 800 * units.MHz], "butter", 10)
# filt *= channelBandPassFilter.get_filter(ff, 0, 0, None, [80 * units.MHz, 1000 * units.GHz], "butter", 5)
filt = channelBandPassFilter.get_filter(ff, 0, 0, None, [80 * units.MHz, 1000 * units.GHz], "butter", 5)
bandwidth = np.trapz(np.abs(filt) ** 2, ff)
Vrms = (noise_temperature * 50 * constants.k * bandwidth / units.Hz) ** 0.5
noise_amplitude = Vrms / (bandwidth / (max_freq)) ** 0.5
# the unit of the data set is micro volts
noise_amplitude /= units.micro

print(f"noise temparture {noise_temperature:.0f}K, bandwidth {bandwidth/units.MHz}MHz, Vrms -> {Vrms/units.micro/units.V:.3g}muV, noise amplitude before filter {noise_amplitude:.2f}muV")

seed = np.random.randint(0, 2 ** 32 - 1)
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin(seed=seed)

def load_one_event(i_file, i_event, norm=1e-6):
    # Load 500 MHz filter
    filt = np.load(f"{common_dir()}/bandpass_filters/500MHz_filter.npy")

    t0 = time.time()
    print(f"loading file {i_file}", flush=True)

    # Load data
    data = np.load(os.path.join(dataset.datapath, f"{dataset.data_filename}{i_file:04d}.npy"), allow_pickle=True, mmap_mode="r")
    
    # Pick out single event data
    data = data[i_event:i_event+1, :, :]
    
    # Apply bandpass filter
    data = np.fft.irfft(np.fft.rfft(data, axis=-1) * filt, axis=-1)
    data = data[:, :, :, np.newaxis]

    # Load lables
    labels_tmp = np.load(os.path.join(dataset.datapath, f"{dataset.label_filename}{i_file:04d}.npy"), allow_pickle=True)
    print(f"finished loading file {i_file} in {time.time() - t0}s")

    shower_energy_data = np.array(labels_tmp.item()["shower_energy"])

    # Pick out the single event label
    shower_energy_data = shower_energy_data[i_event:i_event+1]

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


def realize_noise(data, log10_shower_energy, n_noise_iterations):
    print("Realizing noise...")
    num_samples = data.shape[0]
    ids = np.arange(num_samples, dtype=np.int)

    # create new array strucures that contain the different noise realizations for a batch of data
    tmp_shape = np.array(data.shape)
    tmp_shape[0] = num_samples * n_noise_iterations
    xx = np.zeros(tmp_shape)
    tmp_shape = np.array(log10_shower_energy.shape)
    tmp_shape[0] = num_samples * n_noise_iterations
    yy = np.zeros(tmp_shape)

    for i_event in range(num_samples):
        y = log10_shower_energy[ids[i_event]]
        for i_noise in range(n_noise_iterations):
            yy[i_event * n_noise_iterations + i_noise] = y
            for i_channel in range(n_channels):
                #print(i_event, i_noise, i_channel)
                x = data[ids[i_event], i_channel, :, 0]
                noise_fft = channelGenericNoiseAdder.bandlimited_noise(0, None, n_samples, sampling_rate, noise_amplitude, 
                                                                type='rayleigh', 
                                                                time_domain=False, bandwidth=None)
                tmp = x + fft.freq2time(noise_fft * filt, sampling_rate)  # apply filter to generated noise and add to noiseless trace
                xx[i_event * n_noise_iterations + i_noise, i_channel,:,0] = tmp

    return xx, yy


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate data')
    parser.add_argument("i_file", type=int ,help="the id of the file")
    parser.add_argument("i_event", type=int ,help="the id of the event")
    parser.add_argument("n_noise_iterations", type=int ,help="amount of noise relizations")

    args = parser.parse_args()
    i_file = args.i_file
    i_event = args.i_event
    n_noise_iterations = args.n_noise_iterations

    from matplotlib import pyplot as plt
    print("Starting to generate noise realizations...")

    data, log10_shower_energy = load_one_event(i_file, i_event)

    noise_realized_data, noise_realized_direction = realize_noise(data, log10_shower_energy, n_noise_iterations)

    # Make sure plots folder exists
    if not os.path.exists(f"plots/file_{i_file}_event_{i_event}"):
        os.makedirs(f"plots/file_{i_file}_event_{i_event}")

    for j in range(n_noise_iterations):
        print(f"Plotting noise realization {j}")
        data_event = noise_realized_data[j,:,:,:]

        # Getting x axis (1 step is 0.5 ns)
        x_axis_double = range(int(len(data_event[0])))
        x_axis = [float(x)/2 for x in x_axis_double]

        fig, axs = plt.subplots(5, 1)
        for i in range(5):
            axs[i].plot(x_axis, data_event[i,:,0])
            axs[i].set_xlim([min(x_axis), max(x_axis)])
            if i != 4:
                axs[i].set_title(f'LPDA {i+1}')

        axs[4].set_title('Dipole')

        for ax in axs.flat:
            ax.set(xlabel='time (ns)', ylabel=f'signal (μV)')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        #plt.show()
        fig.set_size_inches(12, 10)
        fig.suptitle(f'Plot of 4 LPDA & 1 dipole for ARZ2020 (had.) for event {i_event} in file {i_file}\nwith 500 MHz bandpass, noise realization {j}')

        plt.savefig(f"plots/file_{i_file}_event_{i_event}/file_{i_file}_event_{i_event}_noise_realization_{j}.png")


    # plot noiseless signal aswell...
    print(f"Plotting noiseless...")

    noiseless_data = data[0,:,:,:]

    # Getting x axis (1 step is 0.5 ns)
    x_axis_double = range(int(len(noiseless_data[0])))
    x_axis = [float(x)/2 for x in x_axis_double]

    fig, axs = plt.subplots(5, 1)
    for i in range(5):
        axs[i].plot(x_axis, noiseless_data[i,:,0])
        axs[i].set_xlim([min(x_axis), max(x_axis)])
        if i != 4:
            axs[i].set_title(f'LPDA {i+1}')

    axs[4].set_title('Dipole')

    for ax in axs.flat:
        ax.set(xlabel='time (ns)', ylabel=f'signal (μV)')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    #plt.show()
    fig.set_size_inches(12, 10)
    fig.suptitle(f'Plot of 4 LPDA & 1 dipole for ARZ2020 (had.) for event {i_event} in file {i_file}\nwith 500 MHz bandpass, noiseless')

    plt.savefig(f"plots/file_{i_file}_event_{i_event}/file_{i_file}_event_{i_event}_noiseless.png")

    # Also plot noisy data
    dataset_noise = True

    dataset = datasets.Dataset(dataset_name, dataset_em, dataset_noise)
    data_noisy, nu_direction_noisy = load_one_event(i_file, i_event)

    print(f"Plotting noisy...")

    noisy_data = data_noisy[0,:,:,:]

    # Getting x axis (1 step is 0.5 ns)
    x_axis_double = range(int(len(noisy_data[0])))
    x_axis = [float(x)/2 for x in x_axis_double]

    fig, axs = plt.subplots(5, 1)
    for i in range(5):
        axs[i].plot(x_axis, noisy_data[i,:,0])
        axs[i].set_xlim([min(x_axis), max(x_axis)])
        if i != 4:
            axs[i].set_title(f'LPDA {i+1}')

    axs[4].set_title('Dipole')

    for ax in axs.flat:
        ax.set(xlabel='time (ns)', ylabel=f'signal (μV)')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    #plt.show()
    fig.set_size_inches(12, 10)
    fig.suptitle(f'Plot of 4 LPDA & 1 dipole for ARZ2020 (had.) for event {i_event} in file {i_file}\nwith 500 MHz bandpass, noisy')

    plt.savefig(f"plots/file_{i_file}_event_{i_event}/file_{i_file}_event_{i_event}_noisy.png")