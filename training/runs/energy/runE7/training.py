# GPU allocation
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
# --------------

# Imports
import os
import numpy as np
import pickle
import argparse
from termcolor import colored
import time
from toolbox import load_file, find_68_interval, models_dir
from radiotools import helper as hp
from PIL import Image

#from scipy import stats
import wandb
from wandb.keras import WandbCallback
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, Flatten, Dropout
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, AveragePooling1D, Input, Flatten
from tensorflow.keras.layers import BatchNormalization, Lambda, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence, plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from generator import TrainDataset, ValDataset, n_events_per_file, n_files_train, batch_size, n_noise_iterations
from constants import run_version, dataset_name, datapath, data_filename, label_filename, plots_dir, project_name, n_files, n_files_val, dataset_em, dataset_noise, test_file_ids
# -------

# Values
feedback_freq = 1 # Only train on 1/feedback_freq of data per epoch
architectures_dir = "architectures"
learning_rate = 0.00005
epochs = 100
loss_function = "mean_absolute_error"
es_patience = 7
es_min_delta = 0.0001 # Old value: es_min_delta = 0.0001
# ------

# Parse arguments
parser = argparse.ArgumentParser(description='Neural network for neutrino energy reconstruction')
parser.add_argument("run_id", type=str ,help="the id of the run, eg '3.2' for run3.2")

args = parser.parse_args()
run_id = args.run_id

# Save the run name
run_name = f"run{run_id}"

# Make sure run_name is compatible with run_version
this_run_version = run_name.split(".")[0]
this_run_id = run_name.split(".")[1]
assert this_run_version == run_version, f"run_version ({run_version}) does not match the run version for this run ({this_run_version})"

# Models folder
saved_model_dir = models_dir(run_name)

# Make sure saved_models folder exists
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)

# Make sure architectures folder exists
if not os.path.exists(f"{saved_model_dir}/{architectures_dir}"):
    os.makedirs(f"{saved_model_dir}/{architectures_dir}")

# Initialize wandb
run = wandb.init(project=project_name,
                 group=run_version,
                 config={  # and include hyperparameters and metadata
                     "learning_rate": learning_rate,
                     "epochs": epochs,
                     "batch_size": batch_size,
                     "loss_function": loss_function,
                     "architecture": "CNN",
                     "dataset": dataset_name
                 })
run.name = run_name
config = wandb.config
    
# Send dataset params to wandb
wandb.log({f"dataset_name": dataset_name,
            f"dataset_em": dataset_em,
            f"dataset_noise": dataset_noise,
            f"test_file_ids": test_file_ids,
            f"datapath": datapath,
            f"data_filename": data_filename,
            f"label_filename": label_filename,
            f"n_files": n_files,
            f"n_files_val": n_files_val })

# Model params
conv2D_filter_size = 5
pooling_size = 4
amount_Conv2D_layers_per_block = 3 
amount_Conv2D_blocks = 4
conv2D_filter_amount = 32
activation_function = "relu"

# Send model params to wandb
wandb.log({f"conv2D_filter_amount": conv2D_filter_amount})
wandb.log({f"conv2D_filter_size": conv2D_filter_size})
wandb.log({f"pooling_size": pooling_size})
wandb.log({f"amount_Conv2D_blocks": amount_Conv2D_blocks})
wandb.log({f"amount_Conv2D_layers_per_block": amount_Conv2D_layers_per_block})
wandb.log({f"activation_function": activation_function})

# ----------- Create model -----------
model = Sequential()

# Conv2D block 1
model.add(Conv2D(conv2D_filter_amount, (1, conv2D_filter_size), strides=(1, 1), padding='same', activation=activation_function, input_shape=(5, 512, 1)))

for _ in range(amount_Conv2D_layers_per_block-1):
    model.add(Conv2D(conv2D_filter_amount, (1, conv2D_filter_size), strides=(1, 1), padding='same', activation=activation_function))

# MaxPooling to reduce size
model.add(AveragePooling2D(pool_size=(1, pooling_size)))

for i in range(amount_Conv2D_blocks-1):
    # Conv2D block
    for _ in range(amount_Conv2D_layers_per_block):
        model.add(Conv2D(conv2D_filter_amount*2**(i+1), (1, conv2D_filter_size), strides=(1, 1), padding='same', activation=activation_function))

    # MaxPooling to reduce size
    model.add(AveragePooling2D(pool_size=(1, pooling_size)))

# Batch normalization
model.add(BatchNormalization())

# Flatten prior to dense layers
model.add(Flatten())

# Dense layers (fully connected)
model.add(Dense(1024, activation=activation_function))
model.add(Dense(1024, activation=activation_function))
model.add(Dense(512, activation=activation_function))
model.add(Dense(256, activation=activation_function))
model.add(Dense(128, activation=activation_function))

# model.add(Dense(512, activation=activation_function))
# # model.add(Dropout(.1))
# model.add(Dense(256, activation=activation_function))
# # model.add(Dropout(.1))
# model.add(Dense(128, activation=activation_function))
# # model.add(Dropout(.1))

# Output layer
model.add(Dense(1))

model.compile(loss=config.loss_function,
              optimizer=Adam(lr=config.learning_rate))
model.summary()
# ------------------------------------

# Save the model (for opening in eg Netron)
#model.save(f'{saved_model_dir}/{architectures_dir}/model_architecture_{run_name}.h5')
plot_model(model, to_file=f'{saved_model_dir}/{architectures_dir}/model_architecture_{run_name}.png', show_shapes=True)
model_json = model.to_json()
with open(f'{saved_model_dir}/{architectures_dir}/model_architecture_{run_name}.json', "w") as json_file:
    json_file.write(model_json)

# Send amount of parameters to wandb
trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

wandb.log({f"trainable_params": trainable_count})
wandb.log({f"non_trainable_params": non_trainable_count})

# Configuring CSV-logger
csv_logger = CSVLogger(os.path.join(saved_model_dir, f"model_history_log_{run_name}.csv"), append=True)

# Configuring callbacks
es = EarlyStopping(monitor="val_loss", patience=es_patience, min_delta=es_min_delta, verbose=1),
mc = ModelCheckpoint(filepath=os.path.join(saved_model_dir, f"model.{run_name}.h5"),
                                                    monitor='val_loss', verbose=1,
                                                    save_best_only=True, mode='auto',
                                                    save_weights_only=False)
wb = WandbCallback(save_model=False)
callbacks = [es, mc , wb, csv_logger]      

# Calculating steps per epoch and batches per file
steps_per_epoch = n_files_train // feedback_freq * (n_events_per_file // batch_size)
n_batches_per_file = n_events_per_file // batch_size
print(f"steps_per_epoch {steps_per_epoch}, n_batches_per_file {n_batches_per_file}")

# Configuring training dataset
dataset_train = tf.data.Dataset.range(n_files_train).prefetch(n_batches_per_file * 10).interleave(
        TrainDataset,
        cycle_length=2,
        num_parallel_calls=2,
        deterministic=False).repeat()

# Configuring validation dataset
dataset_val = tf.data.Dataset.range(n_files_val).prefetch(n_batches_per_file * 10).interleave(
        ValDataset,
        cycle_length=2,
        num_parallel_calls=2,
        deterministic=False)

# Configuring history
history = model.fit(x=dataset_train, steps_per_epoch=steps_per_epoch, epochs=config.epochs,
          validation_data=dataset_val, callbacks=callbacks)

# Dump history with pickle
with open(os.path.join(saved_model_dir, f'history_{run_name}.pkl'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Sleep for a few seconds to free up some resources...
time.sleep(5)

# Plot loss and evaluate
os.system(f"python plot_performance.py {run_id}")

# Calculate 68 % interval and sent to wandb
energy_68 = find_68_interval(run_name)

wandb.log({f"68 % interval": energy_68})

# Send angular resolution image to wandb
energy_res_image = Image.open(f"{plots_dir}/energy_resolution_{run_name}.png")
wandb.log({"energy_resolution": [wandb.Image(energy_res_image, caption=f"energy resolution for {run_name}")]})

# Plot resolution as a function of SNR, energy and azimuth and send to wandb
os.system(f"python resolution_plotter.py {run_id}")
log10_energy_diff_nu_enegy_image = Image.open(f"{plots_dir}/mean_log10_energy_difference_nu_energy_{run_name}.png")
log10_energy_diff_SNR_image = Image.open(f"{plots_dir}/mean_log10_energy_difference_SNR_{run_name}.png")
log10_energy_diff_nu_zenith_image = Image.open(f"{plots_dir}/mean_log10_energy_difference_nu_zenith_{run_name}.png")
log10_energy_diff_nu_azimuth_image = Image.open(f"{plots_dir}/mean_log10_energy_difference_nu_azimuth_{run_name}.png")
wandb.log({"angular_log10_energy_difference_nu_energy": [wandb.Image(log10_energy_diff_nu_enegy_image, caption=f"Angular resolution over nu_energy for {run_name}")]})
wandb.log({"angular_log10_energy_difference_snr": [wandb.Image(log10_energy_diff_SNR_image, caption=f"Angular resolution over SNR for {run_name}")]})
wandb.log({"angular_log10_energy_difference_nu_zenith": [wandb.Image(log10_energy_diff_nu_zenith_image, caption=f"Angular resolution nu_zenith for {run_name}")]})
wandb.log({"angular_log10_energy_difference_nu_azimuth": [wandb.Image(log10_energy_diff_nu_azimuth_image, caption=f"Angular resolution nu_azimuth for {run_name}")]})

run.join()

print(colored(f"Done training {run_name}!", "green", attrs=["bold"]))
print("")



