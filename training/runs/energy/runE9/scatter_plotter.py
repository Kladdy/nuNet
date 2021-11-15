# %%
from toolbox import get_pred_energy_diff_data, models_dir
from matplotlib import pyplot as plt
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Scatter plotterer')
parser.add_argument("run_id", type=str ,help="the id of the run to be analyzed, eg '3.2' for run3.2")

args = parser.parse_args()
run_id = args.run_id

# Save the run name
run_name = f"run{run_id}"

# %%
energy_difference_data, shower_energy_log10_predict, shower_energy_log10 = get_pred_energy_diff_data(run_name, True)

# %%
delta_log_E_string = r"$\Delta(\log_{10}\:E)$"

xmin = min(shower_energy_log10_predict)
xmax = max(shower_energy_log10_predict)
ymin = min(shower_energy_log10)
ymax = max(shower_energy_log10)

fig  = plt.figure()
ax = fig.gca()
ax.plot(shower_energy_log10_predict, shower_energy_log10, '.', markersize=0.1)
ax.plot([xmin, xmax], [ymin, ymax], 'k--')

ax.set_xlabel(f"predicted {delta_log_E_string}")
ax.set_ylabel(f"true {delta_log_E_string}")

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

fig.tight_layout()

fig.savefig(f'{models_dir(run_name)}/scatter_{run_name}.png')
