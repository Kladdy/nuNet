


# Imports
import os
import sys
import numpy as np
import subprocess
import time
from radiotools import helper as hp
from radiotools import stats
from NuRadioReco.utilities import units
import pickle
# -------

# Imports for histogram2d
import math
import os
from matplotlib import colors as mcolors
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# -----------------------


def root_git_dir():
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')

def common_dir():
    return f"{root_git_dir()}/common"

def models_dir(run_name):
    return f"{common_dir()}/models/{run_name}"


def get_pred_energy_diff_data(run_name, do_return_data=False):
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

    if do_return_data:
        return energy_difference_data, shower_energy_log10_predict, shower_energy_log10
    else:
        return energy_difference_data

def get_histogram2d(x=None, y=None, z=None,
                bins=10, range=None,
                xscale="linear", yscale="linear", cscale="linear",
                normed=False, cmap=None, clim=(None, None),
                ax1=None, grid=True, shading='flat', colorbar={},
                cbi_kwargs={'orientation': 'vertical'},
                xlabel="", ylabel="", clabel="", title="",
                fname="hist2d.png"):
    """
    creates a 2d histogram
    Parameters
    ----------
    x, y, z :
        x and y coordinaten for z value, if z is None the 2d histogram of x and z is calculated
    numpy.histogram2d parameters:
        range : array_like, shape(2,2), optional
        bins : int or array_like or [int, int] or [array, array], optional
    ax1: mplt.axes
        if None (default) a olt.figure is created and histogram is stored
        if axis is give, the axis and a pcolormesh object is returned
    colorbar : dict
    plt.pcolormesh parameters:
        clim=(vmin, vmax) : scalar, optional, default: clim=(None, None)
        shading : {'flat', 'gouraud'}, optional
    normed: string
        colum, row, colum1, row1 (default: None)
    {x,y,c}scale: string
        'linear', 'log' (default: 'linear')
    """

    if z is None and (x is None or y is None):
        sys.exit("z and (x or y) are all None")

    if ax1 is None:
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)
    else:
        ax = ax1

    if z is None:
        z, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range)
        z = z.T
    else:
        xedges, yedges = x, y

    if normed:
        if normed == "colum":
            z = z / np.sum(z, axis=0)
        elif normed == "row":
            z = z / np.sum(z, axis=1)[:, None]
        elif normed == "colum1":
            z = z / np.amax(z, axis=0)
        elif normed == "row1":
            z = z / np.amax(z, axis=1)[:, None]
        else:
            sys.exit("Normalisation %s is not known.")

    color_norm = mpl.colors.LogNorm() if cscale == "log" else None
    vmin, vmax = clim
    im = ax.pcolormesh(xedges, yedges, z, shading=shading, vmin=vmin, vmax=vmax, norm=color_norm, cmap=cmap)

    if colorbar is not None:
        cbi = plt.colorbar(im, **cbi_kwargs)
        cbi.ax.tick_params(axis='both', **{"labelsize": 14})
        cbi.set_label(clabel)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    ax.set_title(title)

    return fig, ax, im


# def calculate_percentage_interval(energy_difference_data, percentage=0.68):
#     left_percentage = (1 - percentage) / 2 # For example left_percentage = 0.16 for percentage = 0.68
#     right_percentage = 1 - left_percentage # For example right_percentage = 0.84 for percentage = 0.68

#     # Redefine N
#     N = energy_difference_data.size

#     idx_left = round(left_percentage * N)
#     idx_right = round(right_percentage * N)

#     sorted_energy_difference_data = np.sort(energy_difference_data)

#     left_limit = sorted_energy_difference_data[idx_left]
#     right_limit = sorted_energy_difference_data[idx_right]

#     energy_difference_median = np.median(energy_difference_data)

#     left_abs_diff = np.abs(energy_difference_median - left_limit)
#     right_abs_diff = np.abs(energy_difference_median - right_limit)

#     energy = (left_abs_diff + right_abs_diff) / 2 # Calculate mean

#     return energy

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
    

def get_pred_energy_diff_data(run_name, do_return_data=False):
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

    if do_return_data:
        return energy_difference_data, shower_energy_log10_predict, shower_energy_log10
    else:
        return energy_difference_data



def get_2dhist_normalized_columns(X, Y, fig, ax, binsx, binsy, shading='flat', clim=(None, None), norm=None, cmap=None):
    """
    creates a 2d histogram where the number of entries are normalized to 1 per column
    Parameters
    ----------
    X: array
        x values
    Y: array
        y values
    fig: figure instance
        the figure to plot in
    ax: axis instance
        the axis to plot in
    binsx: array
        the x bins
    binsy: array
        the y bins
    shading: string
        fill style {'flat', 'gouraud'}, see matplotlib documentation (default flat)
    clim: tuple, list
        limits for the color axis (default (None, None))
    norm: None or Normalize instance (e.g. matplotlib.colors.LogNorm()) (default None)
        normalization of the color scale
    cmap: string or None
        the name of the colormap
    Returns
    --------
    pcolormesh object, colorbar object
    """
    H, xedges, yedges = np.histogram2d(X, Y, bins=[binsx, binsy])
    np.nan_to_num(H)
    # Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
    Hmasked = H
    H_norm_rows = Hmasked / np.outer(Hmasked.sum(axis=1, keepdims=True), np.ones(H.shape[1]))

    # ax.set_xlim(16.3, 19)
    # ax.set_ylim(16.3, 19)

    # if run_id == "E13.1":
    #     ax.set_xlim(17, 19)
    #     ax.set_ylim(17, 19)

    max_value_in_range = 0
    for i, x in enumerate(xedges[:-1]):
        for j, y in enumerate(yedges[:-1]):
            if 16.3 < x < 19 and 16.3 < y < 19:
                if H_norm_rows[i, j] > max_value_in_range:
                    max_value_in_range = H_norm_rows[i, j]

    print(max_value_in_range)

    if(cmap is not None):
        cmap = plt.get_cmap(cmap)

    vmin, vmax = clim
    pc = ax.pcolormesh(xedges, yedges, H_norm_rows.T, shading=shading, vmin=vmin, vmax=vmax , norm=norm, cmap=cmap)
    # cb = fig.colorbar(pc, ax=ax, orientation='vertical')
    cb = None

    return pc, cb