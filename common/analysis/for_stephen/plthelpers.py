import inspect, re
import math
import os
from matplotlib import colors as mcolors
from past.builtins import xrange

from scipy import optimize

import matplotlib.pyplot as plt
import numpy as np
import radiotools.stats

def plot_hist_stats(ax, data, weights=None, posx=0.05, posy=0.95, overflow=None,
                    underflow=None, rel=False,
                    additional_text="", additional_text_pre="",
                    fontsize=12, color="k", va="top", ha="left",
                    median=True, quantiles=True, mean=True, std=True, N=True,
                    single_sided=False):
    data = np.array(data)
    textstr = additional_text_pre
    if (textstr != ""):
        textstr += "\n"
    if N:
        textstr += "$N=%i$\n" % data.size
    if not single_sided:
        tmean = data.mean()
        tstd = data.std()
        if weights is not None:

            def weighted_avg_and_std(values, weights):
                """
                Return the weighted average and standard deviation.

                values, weights -- Numpy ndarrays with the same shape.
                """
                average = np.average(values, weights=weights)
                variance = np.average((values - average) ** 2, weights=weights)  # Fast and numerically precise
                return (average, variance ** 0.5)

            tmean, tstd = weighted_avg_and_std(data, weights)

    #     import SignificantFigures as serror
        if mean:
            if weights is None:
    #             textstr += "$\mu = %s \pm %s$\n" % serror.formatError(tmean,
    #                                                 tstd / math.sqrt(data.size))
                textstr += "$\mu = {:.2f}$, ".format(tmean)
            else:
                textstr += "$\mu = {:.2f}$, ".format(tmean)
        if std:
            if rel:
                textstr += "$\sigma = %.2f$ (%.1f\%%)\n" % (tstd, tstd / tmean * 100.)
            else:
                textstr += "$\sigma = %.2f$\n" % (tstd)
        if median:
            tweights = np.ones_like(data)
            if weights is not None:
                tweights = weights
            if quantiles:
                q1 = radiotools.stats.quantile_1d(data, tweights, 0.16)
                q2 = radiotools.stats.quantile_1d(data, tweights, 0.84)
                median = radiotools.stats.median(data, tweights)
    #             median_str = serror.formatError(median, 0.05 * (np.abs(median - q2) + np.abs(median - q1)))[0]
                textstr += "$\mathrm{median} = %.2f^{+%.2f}_{-%.2f}$\n" % (median, np.abs(median - q2),
                                                                           np.abs(median - q1))
            else:
                textstr += "$\mathrm{median} = %.2f $\n" % radiotools.stats.median(data, tweights)
        
    else:
        if(weights is None):
            w = np.ones_like(data)
        else:
            w = weights
        q68 = radiotools.stats.quantile_1d(data, weights=w, quant=.68)
        q95 = radiotools.stats.quantile_1d(data, weights=w, quant=.95)
        textstr += "$\sigma_\mathrm{{68}}$ = {:.1f}$^\circ$\n".format(q68)
        textstr += "$\sigma_\mathrm{{95}}$ = {:.1f}$^\circ$\n".format(q95)

    if(overflow):
        textstr += "$\mathrm{overflows} = %i$\n" % overflow
    if(underflow):
        textstr += "$\mathrm{underflows} = %i$\n" % underflow

    textstr += additional_text
    textstr = textstr[:-1]

    props = dict(boxstyle='square', facecolor='w', alpha=0.5)
    ax.text(posx, posy, textstr, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment=va, ha=ha, multialignment='left',
            bbox=props, color=color)