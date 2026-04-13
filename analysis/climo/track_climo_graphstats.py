# Script for generating graphical summaries of "robust and resistant" cyclone and anticyclone track statistics.

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

GFDL_STORAGE='/orcd/data/talia_tb/001/aqua_gcm_runs'

# a function for producing a "summary graphic" of assorted statistical visualizations
def create_exp_summary_graphic_single(expname, target_dir=None, do_etc_filter=True):
    """
    TODO
    """
    if target_dir == None:
        target_dir = join('figures', 'rr_summary_graphics', expname)
    os.makedirs(target_dir, exist_ok=True)

    exproot = join(GFDL_STORAGE, expname)
    feature_vars = ("MSLP", "vor850")
    feature_types = ("cycs", "acycs")
    feature_units = ("hPa", "10^-5 s^-1")
    statqtys = ("peak_intensity", "duration")

    for i, fv in enumerate(feature_vars):
        for k, qty in enumerate(statqtys):
            # separate figures for intensity and duration statistics
            fig = plt.figure(figsize=(12,9), layout='compressed')

            gs = gridspec.GridSpec(2, 4, figure=fig)

            # Top left quadrant: two subplots side-by-side (horizontally)
            ax1 = fig.add_subplot(gs[0, 0])  # left subplot in top-left quadrant
            ax2 = fig.add_subplot(gs[0, 1])  # right subplot in top-left quadrant
            # Top right quadrant: single subplot
            ax3 = fig.add_subplot(gs[0, 2:4])
            # Bottom left quadrant: single subplot
            ax4 = fig.add_subplot(gs[1, 0:2])
            # Bottom right quadrant: single subplot
            ax5 = fig.add_subplot(gs[1, 2:4])

            ds_list = []
            for j, ft in enumerate(feature_types):
                print(f"Plotting statistics for {fv}_{ft}...")
                path_to_tracks = join(exproot, "tracks", fv, f"{fv}_{ft}.pickle")
                td = load_trackdict_from_file(path_to_tracks)
                if do_etc_filter:
                    td = filter_tracks(td, filter_method='etc_only')
                
                if qty == 'peak_intensity':
                    ds = np.asarray(
                        [np.max(tr.intensity) for tr in td.values()]
                    )
                    print("Intensities")
                elif qty == 'duration':
                    ds = np.asarray(
                        [tr.length*0.25 for tr in td.values()]
                    )
                    print("Duration")
                
                ds_list.append(ds)

            for ax in (ax2,ax3,ax4,ax5):
                if qty == 'peak_intensity':
                    ax.set_xlabel(f'Peak intensity {feature_units[i]}')
                    histbins = ['scott', 'scott']
                elif qty == 'duration':
                    ax.set_label(f'Duration (days)')
                    histbins = [np.arange(2.0, np.max(ds)+0.25, 0.5) for ds in ds_list] # histogram binning should follow discretization of time series

            iv_label = f"Peak intensity {feature_units[i]}" if qty == 'peak_intensity' else "Duration (days)"
            # Box plots
            labels=feature_types
            ax1.boxplot(ds_list, sym='bx', tick_labels=labels)
            ax1.set_ylabel(iv_label)
            ax1.set_title("box & whisker")

            # histograms
            ax3.hist(ds_list[0], bins=histbins[0])
            qs = np.quantile(ds_list[0], (0.25, 0.5, 0.75))
            for q in qs:
                ax3.axvline(q, color='k', linestyle='dashed')

            ax5.hist(ds_list[1], bins=histbins[1])
            qs = np.quantile(ds_list[1], (0.25, 0.5, 0.75))
            for q in qs:
                ax5.axvline(q, color='k', linestyle='dashed')

            ax_max = np.max(
                ( ax3.get_xlim()[-1], ax5.get_xlim()[-1] )
                )
            for ax in (ax3, ax5):
                ax.set_ylabel("counts")
                ax.set_xlabel(iv_label)
                ax.set_xlim(0, ax_max)      # make sure plots have same x-range
            ax3.set_title('cycs histogram')
            ax5.set_title('acycs histogram')

            for j, ds in enumerate(ds_list):
                # plot cdfs
                ax2.ecdf(ds, label=feature_types[j])            

                # generate & plot kdes
                ds_reshaped = ds[:,np.newaxis]
                if qty == 'peak_intensity':
                    xmin = 1
                elif qty == 'duration':
                    xmin = 2
                X_plot = np.linspace(xmin, np.max(ds), 500)[:,np.newaxis]
                kde = KernelDensity(bandwidth=1.0, kernel='gaussian').fit(ds_reshaped)
                log_dens=kde.score_samples(X_plot)
                ax4.plot(X_plot[:,0], np.exp(log_dens), label=feature_types[j])

            ax2.set_xlim(0, ax2.get_xlim()[-1])
            ax2.set_xlabel(iv_label)
            ax2.set_ylabel('cdf')
            ax2.legend()

            ax4.set_xlim(xmin, ax4.get_xlim()[-1])
            ax4.set_ylim(0, ax4.get_ylim()[-1])
            ax4.set_xlabel(iv_label)
            ax4.set_ylabel('kde')
            ax4.set_title('Gaussian kernel density estimate')
            ax4.legend()
            fig.suptitle(f"Summary metrics for {expname} {qty}")

            plt.savefig(join(target_dir, f"{feature_vars[i]}_{qty}_stats.png"), format="png")
            plt.close()

    
    return

def plot_kdes(expnames, tr_varia, pl_varia="intensity", do_etc_filter=True,
              my_cmap='Set1', xlabel='Peak intensity', target_dir=None, savename=None):
    """
    TODO
    """

    if target_dir == None:
        target_dir = join('figures', 'rr_summary_graphics')
    if savename == None:
        savename = f'kde_comparison_{tr_varia}.png'

    ds_list_cyc = []; ds_list_acyc = []

    if pl_varia == "intensity":
        td_to_ds = lambda td: np.asarray(
                    [np.max(tr.intensity) for tr in td.values()]
                )
    elif pl_varia == "duration":
        td_to_ds = np.asarray(
                    [tr.length*0.25 for tr in td.values()]
                )
    else:
        raise Exception("pl_varia must be set to 'intensity' or 'duration'")

    for expn in expnames:
        print(f"Loading data for experiment {expn}...")
        exproot = join(GFDL_STORAGE, expn)
        td_c = load_trackdict_from_file(join(exproot, "tracks", tr_varia, f"{tr_varia}_cycs.pickle" ))
        td_a = load_trackdict_from_file(join(exproot, "tracks", tr_varia, f"{tr_varia}_acycs.pickle"))
        if do_etc_filter:
            td_c = filter_tracks(td_c, filter_method='etc_only')
            td_a = filter_tracks(td_a, filter_method='etc_only')
        ds_list_cyc.append(  td_to_ds(td_c) )
        ds_list_acyc.append( td_to_ds(td_a) )


    # get a list of colors for our experiments
    my_colors = colormaps[my_cmap].colors

    
    fig = plt.figure(); fig = plt.axes()
    for j in range(len(expnames)):
        # reshape datasets for kde calculation
        ds_c = ds_list_cyc[j][  :,np.newaxis ]
        ds_a = ds_list_acyc[j][ :,np.newaxis ]

        # set appropriate lower bound on data
        if pl_varia == "intensity":
            xmin = 1
        elif pl_varia == "duration":
            xmin = 2
        xmax = np.max( np.concatenate([ds_c, ds_a], axis=-1) )
        X_plot = np.linspace(xmin, xmax, 500)[:,np.newaxis]

        kde_c = KernelDensity(bandwidth=1.0, kernel='gaussian').fit(ds_c)
        log_dens_c=kde_c.score_samples(X_plot)
        ax.plot(X_plot[:,0], np.exp(log_dens_c), linestyle='solid', color=my_colors[j], alpha=0.5, label=expnames[j])

        kde_a = KernelDensity(bandwidth=1.0, kernel='gaussian').fit(ds_a)
        log_dens_a=kde_a.score_samples(X_plot)
        ax.plot(X_plot[:,0], np.exp(log_dens_a), linestyle='dashed', color=my_colors[j], alpha=0.5)

    ax.set_xlim(xmin, ax.get_xlim()[-1])
    ax.set_ylim(0,    ax.get_ylim()[-1])
    ax.set_xlabel(xlabel)
    ax.legend()

    plt.savefig(join(target_dir, savename), format="png")
    plt.close()
    return

def plot_ivd_scatter(expname, tr_varia='vor850', do_etc_filter=True, target_dir=None, savename=None):
    """
    TODO
    """
    if target_dir == None:
        target_dir = join("figures", "rr_summary_graphics")
    if savename == None:
        savename = f"{expname}_ivd_scatter.png"
    exproot = join(GFDL_STORAGE, expname)

    td_c = load_trackdict_from_file(join(exproot, "tracks", tr_varia, f"{tr_varia}_cycs.pickle" ))
    td_a = load_trackdict_from_file(join(exproot, "tracks", tr_varia, f"{tr_varia}_acycs.pickle"))
    if do_etc_filter:
        td_c = filter_tracks(td_c, filter_method='etc_only')
        td_a = filter_tracks(td_a, filter_method='etc_only')
    xy_c = [np.asarray([np.max(tr.intensity) for tr in td_c.values()]),
            np.asarray([tr.length*0.25       for tr in td_c.values()])]
    xy_a = [np.asarray([np.max(tr.intensity) for tr in td_a.values()]),
            np.asarray([tr.length*0.25       for tr in td_a.values()])]

    fig = plt.figure(figsize=(10, 6)); ax = plt.axes()
    ax.scatter(xy_c[1], xy_c[0], marker='o', alpha=0.5, label='cyc')
    ax.scatter(xy_a[1], xy_a[0], marker='x', alpha=0.5, label='acyc')
    ax.set_xlabel('Duration (days)')
    ax.set_xlim(1, ax.get_xlim()[-1])
    ax.set_ylabel('Peak intensity ($10^{-5}$ s$^{-1}$)')
    ax.legend()
    ax.set_title(f"{expname} intensity vs. duration")
    plt.savefig(join(target_dir, savename), format="png")    
    plt.close()

    return

def plot_ivs_scatter(expname, tr_varia='vor850', do_etc_filter=True, target_dir=None, savename=None):
    """
    TODO
    """
    if target_dir == None:
        target_dir = join("figures", "rr_summary_graphics")
    if savename == None:
        savename = f"{expname}_ivs_scatter.png"
    exproot = join(GFDL_STORAGE, expname)

    td_c = load_trackdict_from_file(join(exproot, "tracks", tr_varia, f"{tr_varia}_cycs.pickle" ))
    td_a = load_trackdict_from_file(join(exproot, "tracks", tr_varia, f"{tr_varia}_acycs.pickle"))
    if do_etc_filter:
        td_c = filter_tracks(td_c, filter_method='etc_only')
        td_a = filter_tracks(td_a, filter_method='etc_only')
    xy_c = [np.asarray([np.max(tr.intensity) for tr in td_c.values()]),
            np.asarray([tr.get_displacement() for tr in td_c.values()])]
    xy_a = [np.asarray([np.max(tr.intensity) for tr in td_a.values()]),
            np.asarray([tr.get_displacement() for tr in td_a.values()])]

    fig = plt.figure(figsize=(10, 6)); ax = plt.axes()
    ax.scatter(xy_c[1], xy_c[0], marker='o', alpha=0.5, label='cyc')
    ax.scatter(xy_a[1], xy_a[0], marker='x', alpha=0.5, label='acyc')
    ax.set_xlabel('Displacement (geodes. deg.)')
    ax.set_xlim(1, ax.get_xlim()[-1])
    ax.set_ylabel('Peak intensity ($10^{-5}$ s$^{-1}$)')
    ax.legend()
    ax.set_title(f"{expname} intensity vs. displacement")
    plt.savefig(join(target_dir, savename), format="png")    
    plt.close()

    return

def plot_dvs_scatter(expname, tr_varia='vor850', do_etc_filter=True, target_dir=None, savename=None):
    """
    TODO
    """
    if target_dir == None:
        target_dir = join("figures", "rr_summary_graphics")
    if savename == None:
        savename = f"{expname}_dvs_scatter.png"
    exproot = join(GFDL_STORAGE, expname)

    td_c = load_trackdict_from_file(join(exproot, "tracks", tr_varia, f"{tr_varia}_cycs.pickle" ))
    td_a = load_trackdict_from_file(join(exproot, "tracks", tr_varia, f"{tr_varia}_acycs.pickle"))
    if do_etc_filter:
        td_c = filter_tracks(td_c, filter_method='etc_only')
        td_a = filter_tracks(td_a, filter_method='etc_only')
    xy_c = [np.asarray([tr.length*0.25        for tr in td_c.values()]),
            np.asarray([tr.get_displacement() for tr in td_c.values()])]
    xy_a = [np.asarray([tr.length*0.25        for tr in td_a.values()]),
            np.asarray([tr.get_displacement() for tr in td_a.values()])]

    fig = plt.figure(figsize=(6, 6)); ax = plt.axes()
    ax.scatter(xy_c[0], xy_c[1], marker='o', alpha=0.5, label='cyc')
    ax.scatter(xy_a[0], xy_a[1], marker='x', alpha=0.5, label='acyc')
    ax.set_xlabel('Duration (days)')
    ax.set_xlim(1, ax.get_xlim()[-1])
    ax.set_ylabel('Displacement (geodes. deg.)')
    ax.legend()
    ax.set_title(f"{expname} duration vs. displacement")
    plt.savefig(join(target_dir, savename), format="png")    
    plt.close()

    return

# if __name__ == "__main__":
#     expname = sys.argv[1]
#     print(f"Calculating statistics of control-run tracks for experiment {expname}.")
# 
#     create_exp_summary_graphic_single(expname)

# if __name__ == "main__":
# 
#     # list of experiment names
#     variable = sys.argv[1]      # "intensity" or "duration"
#     tr_varia = sys.argv[2]      # "MSLP" or "vor850"
#     expnames = sys.argv[3:]
# 
#     plot_kdes(expnames, tr_varia, pl_varia=variable)

if __name__ == "__main__":
    expname = sys.argv[1]
    print(f"Calculating statistics of control-run tracks for experiment {expname}.")
    create_exp_summary_graphic_single(expname)
    plot_ivs_scatter(expname)
    plot_dvs_scatter(expname)
