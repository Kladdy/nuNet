import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Generate screen command')
parser.add_argument("run_version", type=str, help="the version of the run, eg '9' for run9")
parser.add_argument("runs", type=int ,help="amount of runs in the version")
parser.add_argument("time_offset", type=int, help="amount of minutes to offset time", default=0)

args = parser.parse_args()
run_version = args.run_version
runs = args.runs
time_offset = args.time_offset

run_training_strings = [f"sleep {1*run + time_offset}m && python training.py {run_version}.{run+1}" for run in range(runs)]

for run_string in run_training_strings: 
    print(run_string)