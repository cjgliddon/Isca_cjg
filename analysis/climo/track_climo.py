# Script for generating "robust and resistant" numerical summary measures of 
# cyclone and anticyclone tracks, as well as graphical visualizations of stats.

import numpy as np
from matplotlib import colormaps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import KernelDensity
import os
from os.path import join
from scipy import stats

# loading my modules...
import sys
sys.path.append('..')
from track import *
from experiment import PredictabilityExperiment

# ------------------------------- Basic functions ---------------------------------

def trim_var(a, proportiontocut, axis=None):
    """
    Returns the variance of an array after trimming a specified fraction of extreme values.

    Parameters
    ----------
    a : array-like
        The data to be analyzed.
    proportiontocut : float
        Fraction of the most positive and most negative elements to remove. When the specified 
        proportion does not result in an integer number of elements, the number of elements to 
        trim is rounded down.
    """
    a = np.asarray(a)

    if np.size(a) == 0:
        return np.full(a.shape, np.nan)

    if axis is None:
        a = np.ravel(a)
        axis = 0

    nobs = a.shape[axis]
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if (lowercut > uppercut):
        raise ValueError("Proportion too big.")

    atmp = np.partition(a, (lowercut, uppercut - 1), axis)

    sl = [slice(None)] * atmp.ndim
    sl[axis] = slice(lowercut, uppercut)
    trimmed = atmp[tuple(sl)]
    return np.var(trimmed, axis=axis)

def print_rr_description(arr, unit=None, trim_frac=0.1):
    """
    Prints a variety of "robust and resistant" (see Wilks, 2006) summary statistics for a given
    array of data.
    
    Parameters
    ----------
    arr : array-like
        The data to be analyzed.
    unit : str, optional
        The units of the data (default: None)
    trim_frac : float, optional
        The fraction of the data to be "trimmed" from the trimmed mean and variance metrics
        (default: 0.1)
    """
    quartiles = np.quantile(arr, (0.25, 0.5, 0.75))
    trimmed_mean = stats.trim_mean(arr, proportiontocut=trim_frac)
    trimmed_var = trim_var(arr, proportiontocut=trim_frac)

    trimean = (quartiles[0] + 2*quartiles[1] + quartiles[2])/4
    iqr = quartiles[2] - quartiles[0]
    mad = np.median(
        np.abs(arr - quartiles[1])
        )
    yk_index = (quartiles[2] + quartiles[0] - 2*quartiles[1])/iqr

    print("")    
    print(f"Number of data pts      : {np.size(arr)}")
    print(f"(Trimming fraction used : {trim_frac})")
    print("")
    print(f"Median                  : {quartiles[1]} " + unit)
    print(f"Trimean                 : {trimean} " + unit)
    print(f"Trimmed mean            : {trimmed_mean} " + unit)
    print("")
    print(f"Interquartile range     : {iqr} " + unit)
    print(f"Mean absolute deviation : {mad} " + unit)
    print(f"Trimmed variance        : {trimmed_var} ({unit})^2")
    print("")
    print(f"Yule-Kendall index      : {yk_index}")
    print("")

    return

# TODO: create KDE constructor function helper

# ------------------------------- Main script functions ---------------------------------

def print_all_stats(exp):
    feature_types = ["cycs", "acycs"]
    variables = ["MSLP", "vor850", "Z500", "Z250"]
    units = ["hPa", "10^-5 s^-1", "m", "m"]
    int_metric = "nearby_maximum" 

    for ft in feature_types:
        for var, u in zip(variables, units):
            td = exp.get_control_tracks(var, ft)
            print(f"Calculating statistics for {var}_{ft}...")

            peak_ints = np.asarray(
                [np.max(tr.intensity[int_metric]) for tr in td.values()]
            )
            print("Printing robust and resistant statistics for track intensities...")
            print_rr_description(peak_ints, unit=u, trim_frac=0.1)

            durations = np.asarray(
                [tr.length*0.25 for tr in td.values()]
            )
            print("Printing robust and resistant statistics for track durations...")
            print_rr_description(durations, unit="days", trim_frac=0.1)

            displacements = np.asarray(
                [tr.get_displacement() for tr in td.values()]
            )
            print("Printing robust and resistant statistics for track displacements...")
            print_rr_description(displacements, unit="degrees", trim_frac=0.1)

            if do_filtering:
                td = filter_tracks(td, filter_method='etc_only')
                print(f"Filtering performed on track dictionary (etc-only)")

            # Statistics for paired data

            rrank_dur = stats.spearmanr(durations, peak_ints)
            rrank_dis = stats.spearmanr(displacements, peak_ints)
            rrank_durdis = stats.spearmanr(durations, displacements)
            ktau_dur = stats.kendalltau(durations, peak_ints)
            ktau_dis = stats.kendalltau(displacements, peak_ints)
            ktau_durdis = stats.kendalltau(durations, displacements)
            print(f"Spearman rank correlation coef. (durations, intensities): {rrank_dur.statistic}")
            print(f"Spearman rank correlation coef. (displacements, intensities): {rrank_dis.statistic}")
            print(f"Spearman rank correlation coef. (durations, displacements): {rrank_durdis.statistic}")

            print(f"Kendall's tau (durations, intensities): {ktau_dur.statistic}")
            print(f"Kendall's tau (displacements, intensities): {ktau_dis.statistic}")
            print(f"Kendall's tau (durations, displacements): {ktau_durdis.statistic}")

            print("--------------------------------------------------------------------")
            print("Common (non-robust/resistant) statistics:")
            print(f"Mean peak intensity: {np.mean(peak_ints)}")
            print(f"Peak intensity variance: {np.std(peak_ints)**2}")
            print(" *** ")

def plot_kdes_2x2(exp):
    feature_types = ["cycs", "acycs"]
    variables = ["MSLP", "vor850", "Z500", "Z250"]
    units = ["hPa", "$10^-5$ s$^-1$", "m", "m"]
    int_metric = "nearby_maximum" 

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
    for i, (var, ax, u) in zip(variables, np.ravel(axes), units):
        td_a = exp.get_control_tracks("acycs", var)
        peak_ints_a = np.asarray(
                [np.max(tr.intensity[int_metric]) for tr in td_a.values()]
            )
        td_c = exp.get_control_tracks("cycs",  var)
        peak_ints_c = np.asarray(
                [np.max(tr.intensity[int_metric]) for tr in td_c.values()]
            )

        # generate KDEs: set up horizontal domain
        int_min = 0
        int_max = np.max([np.max(peak_ints_a), np.max(peak_ints_c)])
        int_plot = np.linspace(int_min, int_max, 500)[:, np.newaxis]

        # generate KDEs and pdf data
        # note that score_samples returns log of the pdf, so we take np.exp of the output
        kde_a = KernelDensity(bandwidth=1.0, kernel='gaussian').fit(peak_ints_a[:, np.newaxis])
        pdf_a = np.exp(kde_a.score_samples(int_plot))
        kde_c = KernelDensity(bandwidth=1.0, kernel='gaussian').fit(peak_ints_c[:, np.newaxis])
        pdf_c = np.exp(kde_c.score_samples(int_plot))

        # plot data
        ax.plot(int_plot[:, 0], pdf_a, label='acycs')
        ax.plot(int_plot[:, 0], pdf_c, label='cycs')
        ax.set_xlabel('peak intensity ({u})')
        ax.set_ylabel('pdf')
        ax.set_xlim(int_min, int_max)
        ax.set_ylim(bottom=0)
        ax.set_title(var)
        ax.legend(loc='best', framealpha=0.9)        
    
    return fig, ax

if __name__ == "__main__":
    expname = sys.argv[1]
    my_exp = PredictabilityExperiment(expname)
    print_all_stats(my_exp)
    fig, ax = plot_kdes_2x2(my_exp)
    plt.savefig("intensity_kdes.png")