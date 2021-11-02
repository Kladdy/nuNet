from os import system
import subprocess
import math
import numpy as np
import time
from constants import run_version

class Screen:
    def __init__(self):
        pass

    def get_amount_of_screen_sessions(self):
        res = subprocess.run(['screen', '-ls'], stdout=subprocess.PIPE)
        res_bytes = res.stdout
        res_str = res_bytes.decode('utf-8')
        split_lines = res_str.splitlines()
        if split_lines[0][0] == "N": # If no sockets
            return 0 
        last_line = split_lines[-1]
        split_last_line = last_line.split()
        amount_of_screen_sessions = split_last_line[0] # get first
        return int(amount_of_screen_sessions)

    def initialize_training(self, run_id, arglist): # put exec sh in the end to stop the screen from dying
        system(f"screen -dmS {run_id} bash -c 'python training.py {run_id} {arglist}'")

class Configurations:
    class Hyperparameter:
        def __init__(self, name, values):
            self.name = name
            self.values = values

        def get_random_hyperparameter_value(self):
            rand_idx = np.random.randint(len(self.values))
            return rand_idx, self.values[rand_idx]

    def __init__(self):
        self.hyperparameters = []

    def contains_any_hyperparameters(self):
        if len(self.hyperparameters) != 0:
            return True
        else:
            return False

    def add_hyperparameter(self, name, values):
        hyp = self.Hyperparameter(name, values)
        self.hyperparameters.append(hyp)

    def get_amount_of_configurations(self):
        if self.contains_any_hyperparameters():
            hyperparameter_amounts = [len(hyp.values) for hyp in self.hyperparameters]
            return math.prod(hyperparameter_amounts)
        else:
            return 0

    def get_random_configuration(self):
        names_and_values = []
        for hyperparameter in self.hyperparameters:
            idx, value = hyperparameter.get_random_hyperparameter_value()
            names_and_values.append((hyperparameter.name, value))
        return names_and_values

    def get_random_arglist(self):
        args = []
        for name_and_value in self.get_random_configuration():
            args.append(f"--{name_and_value[0]} {name_and_value[1]}")
        return ' '.join(args)
        

configurations = Configurations()
configurations.add_hyperparameter("loss_function", ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "cosine_similarity"])
configurations.add_hyperparameter("activation_function", ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"])
configurations.add_hyperparameter("conv2D_filter_size", [3, 5, 7])
configurations.add_hyperparameter("pooling_size", [2, 4, 8])
configurations.add_hyperparameter("amount_Conv2D_layers_per_block", [1, 2, 3, 4, 5])
configurations.add_hyperparameter("amount_Conv2D_blocks", [1, 2, 3, 4, 5])
configurations.add_hyperparameter("conv2D_filter_amount", [8, 16, 32, 64])

def main():
    screen = Screen()
    print(f"We have {configurations.get_amount_of_configurations()} configurations available")

    run_verison_without_run = run_version[3:] # The run version, eg E3 or E5
    time_to_sleep = 60 # sleep this amount of seconds before checking if we should create a new training session
    max_amount_of_screen_sessions = 10
    run_id_base = 8 # The run_id to start at, eg 1 for E3.1 or 20 for E3.20 

    initial_amount_of_screen_sessions = screen.get_amount_of_screen_sessions()
    print(f"We already have {initial_amount_of_screen_sessions} screen sessions. We can thus have {max_amount_of_screen_sessions+initial_amount_of_screen_sessions} sessions in total, {max_amount_of_screen_sessions} of which being training sessions")

    while True:
        if screen.get_amount_of_screen_sessions() < max_amount_of_screen_sessions - initial_amount_of_screen_sessions:
            print(f"We can create a new training session! Initializing...")
            arglist = configurations.get_random_arglist()
            this_run_name = f"{run_verison_without_run}.{run_id_base}"
            screen.initialize_training(this_run_name, arglist)
            run_id_base += 1
            print(f"Training session with run_name {this_run_name} initialized!")
        
        time.sleep(time_to_sleep)    

if __name__ == "__main__":
    main()