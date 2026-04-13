from cftime import num2date, date2num
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
from os.path import join
from scipy.stats import linregress
import sys
import xarray as xr

import pdb

sys.path.append('..')
from track import load_trackdict_from_file, filter_tracks
from tools import extract_window

GFDL_STORAGE = os.environ['GFDL_STORAGE']
expname      = 'held_suarez'
exproot      = join(GFDL_STORAGE, expname)

# get vorticity anomaly data and TRACK trajectories
subdir    = join(exproot, 'ensembles', '360.0', 'spec_mag_0.02', 'b01')
data_file = join(subdir, 'pert_anoms.nc')
ctrk_file = join(subdir, 'vor850', 'vor850_cycs.pickle')
atrk_file = join(subdir, 'vor850', 'vor850_acycs.pickle')

# data_file = join(exproot, 'postprocessed', 'base_anoms.nc')
# ctrk_file = join(exproot, 'tracks', 'vor850', 'vor850_cycs.pickle')
# atrk_file = join(exproot, 'tracks', 'vor850', 'vor850_acycs.pickle')

c_tracks  = load_trackdict_from_file(ctrk_file)
a_tracks  = load_trackdict_from_file(atrk_file)
ds        = xr.open_dataset(data_file)
ps        = ds['ps']
vor       = ds['vor'].sel(pfull=850.0, method='nearest')
# vor_trunc = xr.open_dataset("/home/cgliddon/agcm_tracking_tools/TRACK-1.5.4/outdat/vor_filt.nc")['var'].isel(level=0)
# vor = vor_trunc.assign_coords(time=vor0['time'])

# pdb.set_trace()
# filter tracks
c_tracks = filter_tracks(c_tracks)
a_tracks = filter_tracks(a_tracks)

def get_along_track_intensity(tr, darr, method='linear_interpolation', window_size=5):
    """
    Arguments
    ---------
    tr : Track 
        A track object
    darr : xr.DataArray
        An array of (lat, lon, time) data
    method : str
        Method of obtaining the along-track timeseries. Options include:
        > `'nearest_neighbor'`: Gets value of data array at grid point nearest to track coordinates.
        > `'linear_interpolation'`: Linearly interpolates values of data array to position of grid point.
        > `'nearby_maximum'`: Finds the maximum magnitude of the data array within a window of size `window_size` around the track's position.
            (Window size must be specified.)
        > `'window_avg'`: Finds the mean magnitude of the data array within a window of size `window_size` around the track's position.
    window_size : float, optional
        The height and width of the window used for averaging/extremum-finding in the window-using methods, in degrees.

    Returns
    -------
    timeseries : np.array
        A 1-D Numpy array of the same length as the track.
    """
    timeseries = np.zeros(tr.time.shape)
    lons, lats = tr.lon, tr.lat
    ts = num2date(tr.time, 'days since 0001-01-01', calendar='360_day')
    
    # flag whether method requires use of a window
    uses_window = True if method in ['nearby_maximum', 'window_avg'] else False

    for i, (t, lon, lat) in enumerate(zip(ts, lons, lats)):
        if uses_window:
            darr_window = extract_window(darr, lon, lat, t, window_size=window_size)
            if method == 'nearby_maximum':
                value = np.nanmax(np.abs(darr_window.values))
            elif method == 'window_avg':
                # we neglect the latitude weighting of the window values because the size of the window is small
                value = np.abs(np.nanmean(darr_window.values))
        else:
            darr_loc = darr.sel(time=t)
            if method == 'nearest_neighbor':
                value = np.abs(darr_loc.sel(lon=lon, lat=lat, method='nearest').values)
            elif method == 'linear_interpolation':
                value = np.abs(darr_loc.interp(lon=lon, lat=lat, method='linear').values)
        timeseries[i] = value

    return timeseries

def intensity_validation_anim(tr, pcm_data, ctr_data, int_curves, cmap='RdBu_r', suptitle=None):
    """

    """
    import pdb

    # data preparation: time coordinate conversion; "unwrapping" lon axis of track and gridded data
    ttime_num = tr.time
    ttime = num2date(ttime_num, 'days since 0001-01-01', calendar='360_day')
    tlon = np.unwrap(tr.lon, period=360)
    tlat = tr.lat

    dlon = pcm_data['lon'].values
    dlat = pcm_data['lat'].values
    nlon = len(dlon)

    # extract only data array times coinciding with the track
    pcm_data = pcm_data.sel(time=ttime)
    ctr_data = ctr_data.sel(time=ttime)

    lon_uw = np.tile(dlon, 3)
    lon_uw[0:nlon] -= 360; lon_uw[2*nlon:3*nlon] += 360
    ctr_uw = np.tile(ctr_data.values, (1,1,3))
    pcm_uw = np.tile(pcm_data.values, (1,1,3))

    int_curve_array = np.asarray(list(int_curves.values()))

    # pre-compute plot ranges
    tlon_range = np.min(tlon), np.max(tlon)
    tlat_range = np.min(tlat), np.max(tlat)
    ctr_mag    = np.max(np.abs(ctr_uw))
    pcm_mag    = np.max(np.abs(pcm_uw))
    n_ctrs = 20
    int_curve_array = np.asarray(list(int_curves.values()))
    plot_ranges = dict(
        x1=(tlon_range[0] - 2, tlon_range[1] + 2),
        y1=(tlat_range[0] - 2, tlat_range[1] + 2),
        x2=(ttime_num[0], ttime_num[-1]),
        y2=(0, np.max(int_curve_array)),
    )

    # initialize figure
    j = 0       # time index
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), width_ratios=[2, 1], layout='constrained')
    ax = axes[0]
    cm = ax.pcolormesh(lon_uw, dlat, pcm_uw[j], vmin=-pcm_mag, vmax=pcm_mag, cmap=cmap)
    fig.colorbar(cm, ax = ax)
    if suptitle is not None:
        fig.suptitle(suptitle)

    def update(j):
        ax = axes[0]
        ax.cla()
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.pcolormesh(lon_uw, dlat, pcm_uw[j], vmin=-pcm_mag, vmax=pcm_mag, cmap=cmap)
        ax.contour(lon_uw, dlat, ctr_uw[j], levels=np.linspace(-ctr_mag, ctr_mag, n_ctrs), colors='0.5', linewidths=1.0)
        # plot tracks 
        ax.plot(tlon[:j+1],tlat[:j+1], linewidth=2, color='k')
        ax.set_xlim(plot_ranges['x1'])
        ax.set_ylim(plot_ranges['y1'])
        ax.set_title(f"time = {ttime_num[j]}")

        ax = axes[1]
        ax.cla()
        ax.set_xlabel('Time')
        ax.set_ylabel('Intensity')
        for label, series in int_curves.items():
            ax.plot(ttime_num[:j+1], series[:j+1], label=label)
        ax.legend()
        ax.set_xlim(plot_ranges['x2'])
        ax.set_ylim(plot_ranges['y2'])

    anim = FuncAnimation(fig, update, frames=len(ttime_num))
    return anim

def test():
    from pprint import pprint 

    # randomly select a cyclone track
    k = list(c_tracks.keys())[np.random.randint(len(c_tracks))]
    tr = c_tracks[k]
    
    print(f"Calculating along-track timeseries for cyclone track {k}...")
    methods = ["nearest_neighbor", "linear_interpolation", "nearby_maximum", "window_avg"]
    timeseries = {m : get_along_track_intensity(tr, vor, method=m) for m in methods}
    print("(lon, lat):")
    pprint([(tr.lon[i], tr.lat[i]) for i in range(tr.length)])
    print("TRACK intensity output:")
    pprint(tr.intensity)
    print("Vorticity time series from various methods:")
    pprint(timeseries)

def main(plot_trajectories=False, plot_diffscatter=False, plot_peakscatter=False):

    methods = ["nearest_neighbor", "linear_interpolation", "nearby_maximum", "window_avg"]

    # calculate vorticity time series for all cyclone/anticyclone tracks, both with and without interpolation
    cvor_timeseries = dict()
    for k, tr in c_tracks.items():
        vor_timeseries = {m : get_along_track_intensity(tr, vor, method=m) for m in methods}
        cvor_timeseries[k] = vor_timeseries

    avor_timeseries = dict()
    for k, tr in a_tracks.items():
        vor_timeseries = {m : get_along_track_intensity(tr, vor, method=m) for m in methods}
        avor_timeseries[k] = vor_timeseries

    if plot_trajectories:
        # a cute visualization: a two-panel animation with one panel showing the 
        # track trajectory against the vorticity and MSLP anomaly fields, and the other
        # showing the intensity curves as they evolve over time
        iks_to_plot = np.random.randint(len(c_tracks), size=3)
        for ik in iks_to_plot:
            k = list(c_tracks.keys())[ik]
            print(f"Plotting cyclone track {k}...")
            tr        = c_tracks[k]
            plot_data = {m: ts * 1e5 for m, ts in cvor_timeseries[k].items()}
            tr_ints   = tr.intensity
            plot_data['TRACK'] = tr_ints

            pdb.set_trace()

            anim = intensity_validation_anim(tr, pcm_data=vor, ctr_data=ps, 
                                            int_curves=plot_data, suptitle=f"Cyclone track {k} intensity")
            anim.save(f"graphics/cyc_{k}_int_anim.gif", fps=1.5)
            
        iks_to_plot = np.random.randint(len(a_tracks), size=3)
        for ik in iks_to_plot:
            k = list(a_tracks.keys())[ik]
            print(f"Plotting anticyclone track {k}...")
            tr             = a_tracks[k]
            plot_data = {m: ts * 1e5 for m, ts in avor_timeseries[k].items()}
            tr_ints   = tr.intensity
            plot_data['TRACK'] = tr_ints

            anim = intensity_validation_anim(tr, pcm_data=vor, ctr_data=ps, 
                                            int_curves=plot_data, suptitle=f"Anticyclone track {k} intensity")
            anim.save(f"graphics/acyc_{k}_int_anim.gif", fps=1.5)
        
#    if plot_diffscatter:
#        # construct arrays of points for scatterplot
#        cyc_int     = []
#        cyc_raw_ni  = []
#        cyc_raw_i   = []
#        acyc_int    = []
#        acyc_raw_ni = []
#        acyc_raw_i  = []
#
#        for k in c_tracks.keys():
#            trint  = c_tracks[k].intensity
#            vor_ni = np.abs(cvor_no_interp[k]) * 1e5 
#            vor_i  = np.abs(cvor_interp[k]) * 1e5
#
#            for j in range(len(trint)):
#                cyc_int.append(trint[j])
#                cyc_raw_ni.append(vor_ni[j])
#                cyc_raw_i.append(vor_i[j])
#
#        for k in a_tracks.keys():
#            trint  = a_tracks[k].intensity
#            vor_ni = np.abs(avor_no_interp[k]) * 1e5
#            vor_i  = np.abs(avor_interp[k]) * 1e5
#
#            for j in range(len(trint)):
#                acyc_int.append(trint[j])
#                acyc_raw_ni.append(vor_ni[j])
#                acyc_raw_i.append(vor_i[j])
#
#        # do linear regression on pairs of data
#        data_pairs = [(cyc_int, cyc_raw_ni), (acyc_int, acyc_raw_ni), 
#                      (cyc_int, cyc_raw_i) , (acyc_int, acyc_raw_i)  ]
#        linreg_results = []
#        x_to_plot = np.linspace(0, 40)
#        for xdata, ydata in data_pairs:
#            result = linregress(xdata, ydata)
#            result_dict = dict({
#                'y_to_plot': x_to_plot*result.slope + result.intercept,
#                'm'        : result.slope,
#                'b'        : result.intercept,
#                'r2'       : result.rvalue**2,
#                'delta_m'  : result.stderr,
#                'delta_b'  : result.intercept_stderr,
#            })
#            linreg_results.append(result_dict)
#
#        # make the plot
#        fig, axes  = plt.subplots(2, 2, figsize=(10, 10), layout='constrained')
#
#        titles     = ['cyclones (no interp.)', 'anticyclones (no interp.)', 'cyclones (interp.)', 'anticyclones (interp.)']
#
#        fig.suptitle('Feature intensity: TRACK output vs. raw model data')
#        for i, ax in enumerate(axes.ravel()):
#            lr_res = linreg_results[i]
#            ax.set_xlabel('TRACK ($10^{-5}$ s$^{-1}$)')
#            ax.set_ylabel('raw ($10^{-5}$ s$^{-1}$)')
#            ax.set_title(titles[i])
#            # plot scatter
#            ax.scatter(*data_pairs[i], c='k', alpha=0.2)
#            ax.set_xlim(ax.get_xlim())
#            ax.set_ylim(ax.get_ylim())
#            # plot linear regression and "perfect" 1-1 reference
#            # ax.plot(x_to_plot, lr_res['y_to_plot'], linestyle='dashed', c='blue', 
#            #        label=f"$m={lr_res['m']}$ +/- {lr_res['delta_m']}; $R^2 = {lr_res['r2']}$")
#            ax.plot(np.linspace(0, 40), np.linspace(0, 40), linestyle='dashed', c='0.5')
#            # ax.legend()
#
#        plt.savefig('graphics/track_vs_raw_scatter.png', format='png')
#        plt.close()
#
#    if plot_peakscatter:
#        # construct arrays of points for scatterplot
#        cyc_int     = []
#        cyc_raw_ni  = []
#        cyc_raw_i   = []
#        acyc_int    = []
#        acyc_raw_ni = []
#        acyc_raw_i  = []
#
#        for k in c_tracks.keys():
#            int_tpeak = 0.25*(1 + np.argmax(c_tracks[k].intensity))
#            ni_tpeak  = 0.25*(1 + np.argmax(cvor_no_interp[k]))
#            i_tpeak   = 0.25*(1 + np.argmax(cvor_interp[k]))
#            cyc_int.append(int_tpeak)
#            cyc_raw_ni.append(ni_tpeak)
#            cyc_raw_i.append(i_tpeak)
#
#        for k in a_tracks.keys():
#            int_tpeak = 0.25*(1 + np.argmax(a_tracks[k].intensity))
#            ni_tpeak  = 0.25*(1 + np.argmax(avor_no_interp[k]))
#            i_tpeak   = 0.25*(1 + np.argmax(avor_interp[k]))
#            acyc_int.append(int_tpeak)
#            acyc_raw_ni.append(ni_tpeak)
#            acyc_raw_i.append(i_tpeak)
#
#        # do linear regression on pairs of data
#        data_pairs = [(cyc_int, cyc_raw_ni), (acyc_int, acyc_raw_ni), 
#                      (cyc_int, cyc_raw_i) , (acyc_int, acyc_raw_i)  ]
#        linreg_results = []
#        x_to_plot = np.linspace(0, 40)
#        for xdata, ydata in data_pairs:
#            result = linregress(xdata, ydata)
#            result_dict = dict({
#                'y_to_plot': x_to_plot*result.slope + result.intercept,
#                'm'        : result.slope,
#                'b'        : result.intercept,
#                'r2'       : result.rvalue**2,
#                'delta_m'  : result.stderr,
#                'delta_b'  : result.intercept_stderr,
#            })
#            linreg_results.append(result_dict)
#
#        fig, axes = plt.subplots(2, 2, figsize=(10, 10), layout='constrained')
#
#        titles       = ['cyclones (no interp.)', 'anticyclones (no interp.)', 'cyclones (interp.)', 'anticyclones (interp.)']
#
#
#        fig.suptitle('Time of peak intensity: TRACK output vs. raw model data')
#        for i, ax in enumerate(axes.ravel()):
#            ax.set_xlabel('TRACK (days)')
#            ax.set_ylabel('raw (days)')
#            ax.set_title(titles[i])
#            # plot scatter
#            ax.scatter(*data_pairs[i], c='k', alpha=0.2)
#            ax.set_xlim(ax.get_xlim())
#            ax.set_ylim(ax.get_ylim())
#            # plot linear regression and "perfect" 1-1 reference
#            # ax.plot(x_to_plot, lr_res['y_to_plot'], linestyle='dashed', c='blue', 
#            #        label=f"$m={lr_res['m']}$ +/- {lr_res['delta_m']}; $R^2 = {lr_res['r2']}$")
#            ax.plot(np.linspace(0, 40), np.linspace(0, 40), linestyle='dashed', c='0.5')
#            # ax.legend()
#
#        plt.savefig('graphics/track_vs_raw_scatter_tpeak.png', format='png')
#        plt.close()        

    




if __name__ == '__main__':
    test()
    main(plot_trajectories=True)
    
