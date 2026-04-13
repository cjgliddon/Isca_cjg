import numpy as np
import cftime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
import pdb

sys.path.append('..')
from tools import extract_window

def get_feature_snapshot(td, t, size_scaling=None):
    """
    Helper function for animate_tracks_and_synop.

    Arguments
    ---------
    td : dict
        A dictionary whose values are Track objects
    t : float
        A time in days
    size_scaling : int or None, optional

    Returns
    -------
    x, y : np.arr
    intensity: np.arr or None
    """
    x, y, intensity = [], [], []
    for tr in td.values():
        for j in range(tr.length):
            if tr.time[j] == t:
                x.append(tr.lon[j])
                y.append(tr.lat[j])
                if size_scaling == None:
                    intensity = None
                else:
                    intensity.append(tr.intensity[j])            
    return x, y, intensity

def animate_tracks_and_synop(tdicts, ds, intensity_scale=None, colors=['k', 'm', 'g'], markers=['o', 's', 'x'], 
                             time_range=None, plot_anoms=False, save_prefix=None):
    """
    Arguments
    ---------
    tdicts : dict or iterable of dicts
        A dictionary (or iterable of dictionaries) whose keys are integers and whose values are 
        Track objects. 
    ds : xarray.Dataset
        Dataset whose time coordinates must span the range of times included in the track
        dictionary, and which must have variables 'ps' (surface pressure) and 'vor' 
        (vorticity).
    intensity_scale : iterable of floats or None, optional
        If not None, defines the factors by which to rescale the size of the track markers for
        each dictionary in tdicts.
    colors : iterable of strings, optional
        Colors of the track markers for each track dictionary in tdicts. 
    markers : iterable of strings
        Shapes of the track markers for each track dictionary in tdicts.
    time_range : tuple or None
        If specified, gives the initial and final times of the animation. If not specified,
        the animation is presumed to be over the range of times of "ds".
    plot_anoms : bool, optional
        If True, removes the climatology (time mean) from the dataset. Default is False.
    save_prefix : str, optional
        Prefix to the animation's save-out filename. 
    """
    # extract DataArrays of surface pressure and vorticity
    ps_da = ds['ps']
    vor_da = ds['vor'].sel(pfull=850.0, method='nearest')

    # convert data to anomalies if requested
    if plot_anoms:
        ps_da = ps_da - ps_da.mean(dim='time')
        vor_da = vor_da - vor_da.mean(dim='time')
    
    # if we have only one track dictionary, convert to a one-element list
    if type(tdicts) == dict:
        tdicts = [tdicts]

    if intensity_scale == None:
        intensity_scale = [None]*len(tdicts)

    # find initial and final times
    t_float = lambda t: cftime.date2num(t, "days since 0001-01-01")
    t_date = lambda t: cftime.num2date(t, "days since 0001-01-01", calendar="360_day")
    if time_range == None:
        t_init, t_fin = t_float(ds['time'].data[0]), t_float(ds['time'].data[-1])
    else:
        t_init, t_fin = time_range[0], time_range[-1]
    delta_t = 0.25      # timestep length, in days
    num_t = int((t_fin - t_init) / 0.25) + 1

    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')

    # initial frame
    i_f = 0
    time_fl = t_init + i_f*delta_t
    time_dt = t_date(time_fl)
    cm = ax.contourf(ds.lon, ds.lat, vor_da.sel(time=time_dt)*1.0E5, levels=np.arange(-20, 21, 2), extend='both', cmap='coolwarm')
    ct = ax.contour(ds.lon, ds.lat, ps_da.sel(time=time_dt)*0.01, levels=20, colors='0.5', linewidths=0.75)
        
    for j, td in enumerate(tdicts):
        x, y, intensity = get_feature_snapshot(td, time_fl)
        ax.autoscale(enable=False)
        ax.scatter(x, y, s=intensity, c=colors[j], marker=markers[j])
    fig.colorbar(cm, label='Vorticity ($\mathrm{10^{-5}\ s^{-1}}$)')
    ax.set_title(f'time = {time_fl} d')

    def update(i_f):
        ax.cla()
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        time_fl = t_init + i_f*delta_t
        time_dt = t_date(time_fl)

        cm = ax.contourf(ds.lon, ds.lat, vor_da.sel(time=time_dt)*1.0E5, levels=np.arange(-20, 21, 2), extend='both', cmap='coolwarm')
        ct = ax.contour(ds.lon, ds.lat, ps_da.sel(time=time_dt)*0.01, levels=20, colors='0.5', linewidths=0.75)
            
        for j, td in enumerate(tdicts):
            x, y, intensity = get_feature_snapshot(td, time_fl, size_scaling=intensity_scale[j])
            ax.autoscale(enable=False)
            ax.scatter(x, y, s=intensity, c=colors[j], marker=markers[j])
        ax.set_title(f'time = {time_fl} d')

    anim = FuncAnimation(fig, update, frames=num_t)
    if save_prefix == None:
        anim.save('track_synop_anim.gif')
    else:
        anim.save(f'{save_prefix}_track_synop_anim.gif')
    return anim

def animate_track(tr, ds, save_prefix=None, show_anoms=False, figsize=None, lonlims=None, latlims=None, frames=None, loop_no=0):
    time_b = cftime.date2num(ds.time, 'days since 0001-01-01')
    ps = ds['ps']; vor = ds['vor'].sel(pfull=850.0, method='nearest')
    lats = ds['lat'].data
    if show_anoms:
        ps = ps - ps.mean(dim='time')
        vor = vor - vor.mean(dim='time')
    # need to "tile" background file fields to be consistent with unwrapping
    Nlon = len(ds['lon'])
    lons_unwrapped = np.tile(ds['lon'].data, 3)
    lons_unwrapped[0:Nlon] -= 360
    lons_unwrapped[2*Nlon:3*Nlon] += 360

    vor_unwrapped = np.tile(vor.data, (1,1,3))
    ps_unwrapped = np.tile(ps.data, (1,1,3))
    # print(f"Unwrapped surface pressure shape: {ps_unwrapped.shape}")

    time = tr.time; tlat = tr.lat; tlon = np.unwrap(tr.lon, period=360)
    minlat, maxlat = np.min(tlat), np.max(tlat)
    minlon, maxlon = np.min(tlon), np.max(tlon)
    print(f"Lat range: {minlat, maxlat}")
    print(f"Lon range: {minlon, maxlon}")
    if figsize == None:
        aspect_ratio = (maxlon - minlon)/(maxlat - minlat)
        figsize=(5*aspect_ratio, 5)

    if save_prefix == None:
        save_prefix = f"track_{hash(tr)}"

    # set up figure
    fig = plt.figure(figsize = figsize, layout='constrained', dpi=300); ax = plt.axes()
    cm = ax.pcolormesh(vor.lon, vor.lat, vor.isel(time=-1)*1E5, vmin=-40, vmax=40, cmap='coolwarm')
    fig.colorbar(cm, label='Vorticity (10$^{-5}$ s$^{-1}$)')

    # dummy function for calculating the "index offset" between two times
    dt = 0.25
    get_offset = lambda t1, t2: int((t1 - t2) / dt)
    base_offset = get_offset(time[0], time_b[0])

    def update(tj):
        ax.cla()
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.pcolormesh(lons_unwrapped, lats, vor_unwrapped[tj+base_offset]*1E5, vmin=-40, vmax=40, cmap='coolwarm')
        ax.contour(lons_unwrapped, lats, ps_unwrapped[tj+base_offset]*0.01, 
                   levels=np.arange(-100, 100, 5), colors='0.5', linewidths=1)
        # plot tracks 
        ax.plot(tlon[:tj+1],tlat[:tj+1], linewidth=2, color='k')
        if lonlims == None:
            ax.set_xlim(minlon-1, maxlon+1)
        else:
            ax.set_xlim(lonlims)
        if latlims == None:
            ax.set_ylim(minlat-1, maxlat+1)
        else:
            ax.set_ylim(latlims)
        ax.set_title(f"time = {time[tj]}")
    
    if frames == None:
        frames = len(time)
    anim = FuncAnimation(fig, update, frames=frames)
    anim.save(f"{save_prefix}_animation.gif")
    return anim

def animate_matched_tracks(md, ds, save_prefix, figsize=None, lonlims=None, latlims=None):
    """
    Given a dictionary of matched tracks and the path of a "background file", automatically 
    produces GIFs for each set of matching control and perturbed tracks.

    :param dict md: matching track dictionary
    :param str bkgd_file: path to a .nc file of model output
    :param str savename_head:   the name head of the file to which the track saves out
    """
#    pdb.set_trace()
    time_b = cftime.date2num(ds.time, 'days since 0001-01-01')
    ps = ds['ps'] - ds['ps'].mean(dim=('time', 'lon'))
    # print(f"Surface pressure shape: {ps.shape}")
    vor = ds['vor'].sel(pfull=850.0, method='nearest')
    lats = ds['lat'].data

    # need to "tile" background file fields to be consistent with unwrapping
    Nlon = len(ds['lon'])
    lons_unwrapped = np.tile(ds['lon'].data, 3)
    lons_unwrapped[0:Nlon] -= 360
    lons_unwrapped[2*Nlon:3*Nlon] += 360

    vor_unwrapped = np.tile(vor.data, (1,1,3))
    ps_unwrapped = np.tile(ps.data, (1,1,3))
    # print(f"Unwrapped surface pressure shape: {ps_unwrapped.shape}")

    md_list = list(md.items())
    for i in range(len(md_list)):
        tc, tp_list = md_list[i]
        savename = save_prefix + f'_{tc.time[0]}_{i}.gif'


        # unpack all the relevant variables
        time_c, time_p = tc.time, [tp.time for tp in tp_list]
        latc, latp = tc.lat, [tp.lat for tp in tp_list]
        lonc, lonp = np.unwrap(tc.lon, period=360), [np.unwrap(tp.lon, period=360) for tp in tp_list]
        
        # find reasonable ranges for axis limits
        lat_list = latp + [latc]
        lon_list = lonp + [lonc]
        minlat, maxlat, minlon, maxlon = np.nan, np.nan, np.nan, np.nan
        for j in range(len(lat_list)):
            minlat = np.nanmin([minlat, np.min(lat_list[j])])
            minlon = np.nanmin([minlon, np.min(lon_list[j])])
            maxlat = np.nanmax([maxlat, np.max(lat_list[j])])
            maxlon = np.nanmax([maxlon, np.max(lon_list[j])])
        print(f"Lat range: {minlat, maxlat}")
        print(f"Lon range: {minlon, maxlon}")
        aspect_ratio = (maxlon - minlon)/(maxlat - minlat)

        # dummy function for calculating the "index offset" between two times
        dt = 0.25
        get_offset = lambda t1, t2: int((t1 - t2) / dt)

        base_offset = get_offset(time_c[0], time_b[0])
        p_offsets = [get_offset(time_c[0], tp[0]) for tp in time_p]

        # set up figure
        if figsize == None:
            figsize = (5*aspect_ratio, 5)
        fig = plt.figure(figsize = figsize, layout='constrained', dpi=150); ax = plt.axes()
        cm = ax.pcolormesh(vor.lon, vor.lat, vor.isel(time=-1)*1E5, vmin=-40, vmax=40, cmap='coolwarm')
        fig.colorbar(cm, label='Vorticity (10$^{-5}$ s$^{-1}$)')

        def update(tj):
            ax.cla()
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.pcolormesh(lons_unwrapped, lats, vor_unwrapped[tj+base_offset]*1E5, vmin=-40, vmax=40, cmap='coolwarm')
            ax.contour(lons_unwrapped, lats, ps_unwrapped[tj+base_offset]*0.01, 
                       levels=np.arange(-100, 100, 5), colors='0.5', linewidths=1)
            # plot tracks 
            ax.plot(lonc[:tj+1],latc[:tj+1], linewidth=2, color='k')
            for k in range(len(tp_list)):
                po = p_offsets[k]
                if tj+1+po > 0:
                    ax.plot(lonp[k][:tj+1+po], latp[k][:tj+1+po], linewidth=1, color='k', linestyle='dashed')
            if lonlims == None:
                ax.set_xlim(minlon-1, maxlon+1)
            else:
                ax.set_xlim(lonlims)
            if latlims == None:
                ax.set_ylim(minlat-1, maxlat+1)
            else:
                ax.set_ylim(latlims)
            ax.set_title(f"time = {time_c[tj]}")
        n_frames = len(time_c)
        anim = FuncAnimation(fig, update, frames=n_frames)
        anim.save(savename)
        plt.close()

    return

def animate_matched_tracks_pv(md, ds, save_prefix, figsize=None, lonlims=None, latlims=None, pv_scale=1):
    """
    Given a dictionary of matched tracks and the path of a "background file", automatically 
    produces GIFs for each set of matching control and perturbed tracks.

    :param dict md: matching track dictionary
    :param str ds: xarray dataset
    :param str savename_head:   the name head of the file to which the track saves out
    """
#    pdb.set_trace()
    time_b = cftime.date2num(ds.time, 'days since 0001-01-01')
    pv = ds['pv']
    # print(f"Surface pressure shape: {ps.shape}")
    lats = ds['lat'].data

    # need to "tile" background file fields to be consistent with unwrapping
    Nlon = len(ds['lon'])
    lons_unwrapped = np.tile(ds['lon'].data, 3)
    lons_unwrapped[0:Nlon] -= 360
    lons_unwrapped[2*Nlon:3*Nlon] += 360

    pv_unwrapped = np.tile(pv.data, (1,1,3))
    # print(f"Unwrapped surface pressure shape: {ps_unwrapped.shape}")

    md_list = list(md.items())
    for i in range(len(md_list)):
        tc, tp_list = md_list[i]
        savename = save_prefix + f'pv250_{tc.time[0]}_{i}.gif'


        # unpack all the relevant variables
        time_c, time_p = tc.time, [tp.time for tp in tp_list]
        latc, latp = tc.lat, [tp.lat for tp in tp_list]
        lonc, lonp = np.unwrap(tc.lon, period=360), [np.unwrap(tp.lon, period=360) for tp in tp_list]
        
        # find reasonable ranges for axis limits
        lat_list = latp + [latc]
        lon_list = lonp + [lonc]
        minlat, maxlat, minlon, maxlon = np.nan, np.nan, np.nan, np.nan
        for j in range(len(lat_list)):
            minlat = np.nanmin([minlat, np.min(lat_list[j])])
            minlon = np.nanmin([minlon, np.min(lon_list[j])])
            maxlat = np.nanmax([maxlat, np.max(lat_list[j])])
            maxlon = np.nanmax([maxlon, np.max(lon_list[j])])
        print(f"Lat range: {minlat, maxlat}")
        print(f"Lon range: {minlon, maxlon}")
        aspect_ratio = (maxlon - minlon)/(maxlat - minlat)

        # dummy function for calculating the "index offset" between two times
        dt = 0.25
        get_offset = lambda t1, t2: int((t1 - t2) / dt)

        base_offset = get_offset(time_c[0], time_b[0])
        p_offsets = [get_offset(time_c[0], tp[0]) for tp in time_p]

        # set up figure
        if figsize == None:
            figsize = (5*aspect_ratio, 5)
        fig = plt.figure(figsize = figsize, layout='constrained', dpi=150); ax = plt.axes()
        cm = ax.pcolormesh(pv.lon, pv.lat, pv.isel(time=-1)*1E6, vmin=-pv_scale, vmax=pv_scale, cmap='coolwarm')
        fig.colorbar(cm, label='$PV_{250}$ (PVU)')

        def update(tj):
            ax.cla()
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.pcolormesh(lons_unwrapped, lats, pv_unwrapped[tj+base_offset]*1E6, vmin=-pv_scale, vmax=pv_scale, cmap='coolwarm')
#            ax.contour(lons_unwrapped, lats, ps_unwrapped[tj+base_offset]*0.01, 
#                       levels=np.arange(-100, 100, 5), colors='0.5', linewidths=1)
            # plot tracks 
            ax.plot(lonc[:tj+1],latc[:tj+1], linewidth=2, color='k')
            for k in range(len(tp_list)):
                po = p_offsets[k]
                if tj+1+po > 0:
                    ax.plot(lonp[k][:tj+1+po], latp[k][:tj+1+po], linewidth=1, color='k', linestyle='dashed')
            if lonlims == None:
                ax.set_xlim(minlon-1, maxlon+1)
            else:
                ax.set_xlim(lonlims)
            if latlims == None:
                ax.set_ylim(minlat-1, maxlat+1)
            else:
                ax.set_ylim(latlims)
            ax.set_title(f"time = {time_c[tj]}")
        n_frames = len(time_c)
        anim = FuncAnimation(fig, update, frames=n_frames)
        anim.save(savename)
        plt.close()

    return

def animate_window(tr, da_dicts, save_prefix=None, figsize=None, lonlims=None, latlims=None, save_out = False):
    """
    Arguments
    ---------
    tr : Track object
        The track for which we wish to produce an animation.
    da_dict : list of dictionaries
        List of dictionaries containing the data to be used for the
        animation's "background."
    save_prefix : str
    """
    fig = plt.figure(figsize=(6, 5), layout='constrained'); ax = plt.axes()

    i = 0
    t, lon, lat = tr.time[i], tr.lon[i], tr.lat[i]
    t_date = cftime.num2date(t, 'days since 0001-01-01', calendar='360_day')

    for _dict in da_dicts:
        da = _dict['data']
        print(da.min(), da.max())
        da_window = extract_window(da, lon, lat, time=t_date)
        plot_func = _dict['plot_func']
        out = plot_func(da_window.lon, da_window.lat, da_window * _dict['plot_scale'],
                                 **_dict['plot_kws'])
        if ('cbar_kws' in _dict):
            fig.colorbar(out, **_dict['cbar_kws']) 
    ax.set_xlabel('Lon')
    ax.set_ylabel('Lat')

    def update_window(i):
        ax.cla()
        
        t, lon, lat = tr.time[i], tr.lon[i], tr.lat[i]
        t_date = cftime.num2date(t, 'days since 0001-01-01', calendar='360_day')

        for _dict in da_dicts:
            da = _dict['data']
            da_window = extract_window(da, lon, lat, time=t_date)
            plot_func = _dict['plot_func']
            out = plot_func(da_window.lon, da_window.lat, da_window * _dict['plot_scale'],
                                     **_dict['plot_kws'])
        ax.set_xlabel('Lon')
        ax.set_ylabel('Lat')
        ax.set_title(f'(t, lon, lat) = ({t} d, {lon:.2f}°, {lat:.2f}°)')
    
    anim = FuncAnimation(fig, update_window, frames = tr.length)
    return anim
        

def test():

    from os.path import join
    import track as trk
    import xarray as xr

    GFDL_STORAGE = os.environ['GFDL_STORAGE']
    expname = "default"
    exproot = join(GFDL_STORAGE, expname)

    tracks = trk.load_trackdict_from_file(join(exproot, 'tracks', 'vor850', 'vor850_cycs.pickle'))
    t0 = 679.0      # track start time
    tracks = {k: v for k, v in tracks.items() if v.time[0] == t0}

    # datafields
    data = dict({
        'pvtend_250': xr.open_dataset(join(exproot, 'postprocessed', 'dpv250_dt.nc')),
        'pvtend_500': xr.open_dataset(join(exproot, 'postprocessed', 'dpv500_dt.nc')),
        'pvtend_850': xr.open_dataset(join(exproot, 'postprocessed', 'dpv850_dt.nc')),        
    })

    plot_dicts = [
        dict({
            'var'        : 'dpv250_dt',
            'data'       : xr.open_dataset(join(exproot, 'postprocessed', 'dpv250_dt.nc'))['pv'],
            'plot_func'  : plt.contourf,
            'plot_scale' : -1e6*86400,       # PVU/day
            'plot_kws'   : dict(cmap='RdBu_r', levels=np.arange(-10, 10.1, 0.5), extend='both'),
            'cbar_kws'   : dict(label='dPV/dt (PVU/d)'),
        }),
        dict({
            'var'        : 'pv250',
            'data'       : xr.open_dataset(join(exproot, 'postprocessed', 'pv250_anoms.nc'))['pv'],
            'plot_func'  : plt.contour,
            'plot_scale' : -1e6,        # PVU
            'plot_kws'   : dict(colors='k', linewidths=1, levels=np.arange(-10, 10.1, 0.5)),
        })
    ]

    for k, v in tracks.items():
        anim = animate_window(v, plot_dicts)
        anim.save(f'track_anim_{k}_test.gif', fps=1.5)


def synop_test():
    from os.path import join
    import track as trk
    import xarray as xr

    GFDL_STORAGE = os.environ['GFDL_STORAGE']
    expname = "rotation/omegax2"
    exproot = join(GFDL_STORAGE, expname)

    subdir = join(exproot, 'ensembles', '370.0', 'spec_mag_0.02', 'b09')
    ds = xr.open_dataset(join(subdir, 'pert_anoms.nc'))
    tdict = trk.load_trackdict_from_file(join(subdir, 'vor850', 'vor850_cycs.pickle'))
    animate_tracks_and_synop(tdict, ds, save_prefix='b09')

def match_anim_test():
    from os.path import join
    import track as trk
    from ensemble import EnsembleSet
    import xarray as xr

    GFDL_STORAGE = os.environ['GFDL_STORAGE']
    expname = "held_suarez"
    exproot = join(GFDL_STORAGE, expname)
    ds = xr.open_dataset(join(exproot, 'postprocessed', 'base_anoms.nc'))
    es = EnsembleSet(exproot, n_mems=20)
    for t0 in np.arange(360.0, 400.0, 10.0):
        md = es.match_tracks('MSLP_cycs', (t0+9.0, t0+9.0), t0)
        animate_matched_tracks(md, ds, 'anim_test_pres')


if __name__ == "__main__":
    print("Got into Main")

    match_anim_test()