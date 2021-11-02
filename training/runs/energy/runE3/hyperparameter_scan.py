from os import system
import subprocess
import math

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

class Configurations:
    class Hyperparameter:
        def __init__(self, name, values):
            self.name = name
            self.values = values

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

configurations = Configurations()
configurations.add_hyperparameter("loss_function", ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "cosine_similarity", "huber", "log_cosh"])
configurations.add_hyperparameter("activation_function", ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"])
configurations.add_hyperparameter("conv2D_filter_size", [3, 5, 7])
configurations.add_hyperparameter("pooling_size", [2, 4, 8])
configurations.add_hyperparameter("amount_Conv2D_layers_per_block", [1, 2, 3, 4, 5])
configurations.add_hyperparameter("amount_Conv2D_blocks", [1, 2, 3, 4, 5])
configurations.add_hyperparameter("conv2D_filter_amount", [8, 16, 32, 64])

def main():
    print(f"We have {configurations.get_amount_of_configurations()} configurations available")
    
    screen = Screen()
    print(screen.get_amount_of_screen_sessions())

if __name__ == "__main__":
    main()



# for _ in range(5):
#     res = system("screen -dm bash -c 'sleep 5; echo hello; sleep 5; echo hejdå; exec sh'")
#     print("result2 is: ", res)