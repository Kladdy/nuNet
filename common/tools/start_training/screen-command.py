import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Generate screen command')
parser.add_argument("run_version", type=str ,help="the version of the run, eg '9' for run9")
parser.add_argument("runs", type=int ,help="amount of runs in the version")

args = parser.parse_args()
run_version = args.run_version
runs = args.runs

screen_strings = [f"screen -S {run_version}.{run+1}" for run in range(runs)]
command = " && ".join(screen_strings)

print(command)