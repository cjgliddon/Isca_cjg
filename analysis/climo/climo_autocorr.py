"""
A script for calculating and plotting the autocorrelation function for
the time series of various atmospheric variables measured at selected
latitudes. 
"""

import numpy as np
from matplotlib import colormaps
import matplotlib.pyplot as plt
import os
from os.path import join
import sys
# from scipy.signal import correlate
import xarray as xr

GFDL_STORAGE = os.environ['GFDL_STORAGE']

def get_acorr_ts(data, nlags):
    ndata = np.size(data)
    data_subs0 = data[:ndata - nlags]
    sd0 = np.std(data_subs0)
    corrs = np.zeros(nlags)
    for i in range(nlags):
        data_subsi = data[i:ndata-nlags+i]
        sdi = np.std(data_subsi)
        m = np.stack((data_subs0, data_subsi), axis=0)
        cov = np.cov(m)[0,1]        # (get cross-correlation)
        corrs[i] = cov/(sd0*sdi)
    return corrs


if __name__ == "__main__":
    print("Got into Main")

    expname = sys.argv[1]
    my_cmap = 'Set1'
    nlags = 14*4

    my_colors = colormaps[my_cmap].colors
    target_dir = join("figures", "rr_summary_graphics")
    savename = f"{expname}_ll_acorr.png"

    exproot = join(GFDL_STORAGE, expname)

    # figure 1: low-level synoptic variables
    synop_ds = xr.open_dataset(join(exproot, 'postprocessed', 'base_anoms.nc'))

    latitudes = np.asarray((0.0, 20.0, 40.0, 60.0, 80.0))
    # select a random longitude for each latitude
    nlons = synop_ds.lon.shape
    lon_indices = np.random.randint(0, nlons, size=latitudes.shape)

    ps  = synop_ds['ps']
    vor = synop_ds['vor'].isel(pfull=-1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
    axes[0].set_title('ps')
    axes[1].set_title('vor850')

    for ax in axes:
        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('$r_k$')
    
    fig.suptitle(f'{expname} low-level synoptic autocorrelations')

    t_points = np.arange(0, 0.25*nlags, 0.25)
    for i, lat in enumerate(latitudes):
        ps_ts  = ps.sel(lat=lat, method='nearest').isel(lon=lon_indices[i]).data
        vor_ts = vor.sel(lat=lat, method='nearest').isel(lon=lon_indices[i]).data

        ps_acorr  = get_acorr_ts(ps_ts, nlags=nlags)
        vor_acorr = get_acorr_ts(vor_ts, nlags=nlags)

        axes[0].plot(t_points, ps_acorr, color=my_colors[i], label=str(lat))
        axes[1].plot(t_points, vor_acorr, color=my_colors[i], label=str(lat))
    
    for ax in axes:
        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('$r_k$')
        ax.set_xlim(t_points[0], t_points[-1])
        ax.legend()

    plt.savefig(join(target_dir, savename), format="png")
    plt.close()

    # figure 2: pv
    pv_vars = ('pv250', 'pv500', 'pv850')
    pv_dict = {k: dict({
                    'datafile': f'{k}_anoms.nc',
                })
               for k in pv_vars}
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout='constrained')
    for i, var in enumerate(pv_vars):
        ax = axes[i]
        ax.set_title(var)
        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('$r_k$')

        ds = xr.open_dataset(join(exproot, 'postprocessed', pv_dict[var]['datafile']))
        for j, lat in enumerate(latitudes):
            pv_ts = ds['pv'].sel(lat=lat, method='nearest').isel(lon=lon_indices[j]).data
            pv_acorr  = get_acorr_ts(pv_ts, nlags=nlags)
            ax.plot(t_points, pv_acorr, color=my_colors[j], label=str(lat))
        ax.legend()
        ax.set_xlim(t_points[0], t_points[-1])


    savename=f"{expname}_pv_acorr.png"
    plt.savefig(join(target_dir, savename), format="png")
    plt.close()


