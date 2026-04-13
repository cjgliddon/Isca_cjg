import cftime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import os
from os.path import join
import sys
import xarray as xr

import pdb

# my modules
sys.path.append('..')
sys.path.append('../..')
from ensemble import EnsembleSet
from pv_decomposition import get_decomp_coefficients_single
import track as trk
from tools import extract_window
from visualizations.animations import animate_window

# ---------------------------- PREFATORY FUNCTIONS ---------------------------------

def animate_track(tr, da_dicts, figsize = None, lonlims = None, latlims = None):
    """
    """
    # We need to "tile" background file fields to be consistent with unwrapping
    for _dict in da_dicts:
        lons, lats, times = _dict['data'].lon.data, _dict['data'].lat.data, _dict['data'].time.data
        tiled_data = np.tile(_dict['data'].data, (1,1,3))
        _dict['data_tiled'] = tiled_data
    n_lon = len(lons)
    lons_unwrapped = np.tile(lons, 3)
    lons_unwrapped[0:n_lon]         -= 360.0
    lons_unwrapped[2*n_lon:3*n_lon] += 360.0

    # convert time coordinates to simple floats for consistency with track object
    times = cftime.date2num(times, 'days since 0001-01-01')

    # get coordinates of track trajectory
    tr_time = tr.time; tr_lat = tr.lat
    tr_lon = np.unwrap(tr.lon, period=360)
    latmin, latmax = np.min(tr_lat), np.max(tr_lat)
    lonmin, lonmax = np.min(tr_lon), np.max(tr_lon)

    if figsize == None:
        aspect_ratio = (lonmin - lonmax)/(latmin - latmax)
        figsize=(5*aspect_ratio, 5)
    
    if lonlims == None:
        lonlims = (lonmin-1, lonmax+1)
    if latlims == None:
        latlims = (latmin-1, latmax+1)

    # set up figure
    fig = plt.figure(figsize = figsize, layout='constrained', dpi=300); ax = plt.axes() 
    where_time = lambda t: np.where(times == t)[0][0]   # helper function for time indexing the main arrays  
 
    i = 0
    j = where_time(tr_time[i])
    for _dict in da_dicts:
        data = _dict['data_tiled']
        plot_func = _dict['plot_func']
        out = plot_func(lons_unwrapped, lats, data[j] * _dict['plot_scale'], **_dict['plot_kws'])
        if ('cbar_kws' in _dict):
            fig.colorbar(out, **_dict['cbar_kws'])
    ax.set_xlim(lonlims)
    ax.set_ylim(latlims)
    ax.set_xlabel('Lon')
    ax.set_ylabel('Lat')
    ax.set_title(f'time = {tr_time[i]}')

    track_kws = dict(linewidth=2, color='k')

    def update(i):
        ax.cla()
        j = where_time(tr_time[i])
        for _dict in da_dicts:
            data = _dict['data_tiled']
            plot_func = _dict['plot_func']
            out = plot_func(lons_unwrapped, lats, data[j] * _dict['plot_scale'], **_dict['plot_kws'])
        ax.plot(tr_lon[:i+1], tr_lat[:i+1], **track_kws)
        print(tr_time[i], times[j])

        ax.set_xlim(lonlims)
        ax.set_ylim(latlims)
        ax.set_xlabel('Lon')
        ax.set_ylabel('Lat')
        ax.set_title(f'time = {tr_time[i]}')
    
    anim = FuncAnimation(fig, update, frames=tr.length)
    return anim


GFDL_STORAGE = os.environ['GFDL_STORAGE']
expname = "default"
exproot = join(GFDL_STORAGE, expname)

tracks = trk.load_trackdict_from_file(join(exproot, 'tracks', 'vor850', 'vor850_cycs.pickle'))
t0 = 679.0      # track start time
tracks = {k: v for k, v in tracks.items() if v.time[0] == t0}

if True:
    plot_dicts = [
        dict({
            'var'        : 'dpv850_dt',
            'data'       : xr.open_dataset(join(exproot, 'postprocessed', 'dpv850_dt.nc'))['pv'],
            'plot_func'  : plt.contourf,
            'plot_scale' : 1e6*86400,       # PVU/day
            'plot_kws'   : dict(cmap='RdBu_r', levels=np.arange(-10, 10.1, 0.5), extend='both'),
            'cbar_kws'   : dict(label='dPV/dt (PVU/d)'),
        }),
        dict({
            'var'        : 'pv850',
            'data'       : xr.open_dataset(join(exproot, 'postprocessed', 'pv850_anoms.nc'))['pv'],
            'plot_func'  : plt.contour,
            'plot_scale' : 1e6,        # PVU
            'plot_kws'   : dict(colors='k', linewidths=1, levels=np.arange(-10, 10.1, 0.5)),
        }),
        dict({
            'var'        : 'vor850',
            'data'       : xr.open_dataset(join(exproot, 'postprocessed', 'base_anoms.nc'))['vor'].sel(pfull=850.0, method='nearest'),
            'plot_func'  : plt.contourf,
            'plot_scale' : 1e5,
            'plot_kws'   : dict(cmap='RdBu_r', levels=np.arange(-30, 30.1, 2), extend='both'),
            'cbar_kws'   : dict(label='$ζ_850$ ($10^{-5}$ s$^{-1}$)'),
        })
    ]

    for k, v in tracks.items():
        anim = animate_window(v, plot_dicts[0:2])
        anim.save(f'track_window_{k}_test.gif', fps=1.5)
        del anim

        anim = animate_track(v, plot_dicts[1:])
        anim.save(f'track_anim_{k}_test.gif', fps=2)
        del anim

# We'll pursue a further analysis of track 2042, a fairly long-lived NH cyclone.

import copy 

N = 29
tr_snipped = copy.deepcopy(tracks[2042])
tr_snipped.time = tr_snipped.time[:N]
tr_snipped.lon = tr_snipped.lon[:N]
tr_snipped.lat = tr_snipped.lat[:N]
tr_snipped.length = N
print(tr_snipped.lat)

ds       = join(exproot, 'postprocessed', 'pv850.nc')
ds_climo = join(exproot, 'postprocessed', 'pv850_mean.nc')
ds_tdv   = join(exproot, 'postprocessed', 'dpv850_dt.nc')

results = get_decomp_coefficients_single(ds, ds_climo, ds_tdv, tr_snipped, 'cyc', return_full_results=True)
coeffs  = get_decomp_coefficients_single(ds, ds_climo, ds_tdv, tr_snipped, 'cyc', return_full_results=False)

# before we plot, let's get the dpv/dt anomaly field
dpv_dt = xr.open_dataset(join(exproot, 'postprocessed', 'dpv850_dt.nc'))['pv']
dpv_dt_anom = dpv_dt
pv_anom = xr.open_dataset(join(exproot, 'postprocessed', 'pv850_anoms.nc'))['pv']

fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), layout='constrained')
times = np.asarray(list(results.keys()))
dates = cftime.num2date(times, 'days since 0001-01-01', calendar='360_day')
ravax = axes.ravel()

scale_fac = 1e6*86400

i = 0
ax = ravax[0]
window = extract_window(dpv_dt, tr_snipped.lon[i], tr_snipped.lat[i], dates[i], window_size=50)
cm = ax.contourf(window.lon, window.lat, window * 1e6 * 86400, 
                    levels=np.arange(-2, 2.1, 0.2), extend='both', cmap='RdBu_r')
fig.colorbar(cm, ax=ax)

now_result = results[times[i]]

ax = ravax[1]
cm = ax.contourf(now_result['int'] * 1e6 * 86400,
                    levels=np.arange(-0.1, 0.11, 1.0E-2), extend='both', cmap='RdBu_r')
fig.colorbar(cm, ax=ax)

ax = ravax[2]
cm = ax.contourf(now_result['prop'] * 1e6 * 86400, levels=np.arange(-1.0, 1.1, 0.1), extend='both', cmap='RdBu_r')
fig.colorbar(cm, ax=ax)

ax = ravax[3]
cm = ax.contourf(now_result['def'] * 1e6 * 86400, levels=np.arange(-0.5, 0.55, 0.05), extend='both', cmap='RdBu_r')
fig.colorbar(cm, ax=ax)

ax = ravax[4]
ax.plot(times[:], coeffs['beta'][:])
xlim = ax.get_xlim()
ylim4 = ax.get_ylim()

ax = ravax[5]
ax.plot(times[:], coeffs['ax'][:], label='ax')
ax.plot(times[:], coeffs['ay'][:], label='ay')
ylim5 = ax.get_ylim()
ax.legend()

def update(i):
    for ax in ravax:
        ax.cla()

    ax = ravax[0]
    window = extract_window(dpv_dt, tr_snipped.lon[i], tr_snipped.lat[i], dates[i], window_size=50)
    window2 = extract_window(pv_anom, tr_snipped.lon[i], tr_snipped.lat[i], dates[i], window_size=50)
    cm = ax.contourf(window.lon, window.lat, window * 1e6 * 86400, 
                    levels=np.arange(-2, 2.1, 0.2), extend='both', cmap='RdBu_r')
    ct = ax.contour(window.lon, window.lat, window2*1e6, levels=np.arange(-5, 5.05, 0.25), colors='k')

    now_result = results[times[i]]

    ax = ravax[1]
    cm = ax.contourf(now_result['int'] * 1e6 * 86400,
                    levels=np.arange(-0.1, 0.11, 0.01), extend='both', cmap='RdBu_r')

    ax = ravax[2]
    cm = ax.contourf(now_result['prop'] * 1e6 * 86400, levels=np.arange(-1.0, 1.1, 0.1), extend='both', cmap='RdBu_r')

    ax = ravax[3]
    cm = ax.contourf(now_result['def'] * 1e6 * 86400, levels=np.arange(-0.5, 0.55, 0.05), extend='both', cmap='RdBu_r')

    ax = ravax[4]
    ax.plot(times[:i+1], coeffs['beta'][:i+1], label='beta')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim4)
    ax.legend()

    ax = ravax[5]
    ax.plot(times[:i+1], coeffs['ax'][:i+1], label='ax')
    ax.plot(times[:i+1], coeffs['ay'][:i+1], label='ay')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim5)
    ax.legend()

anim = FuncAnimation(fig, update, frames=tr_snipped.length)
anim.save('test_decomp_anim.gif')