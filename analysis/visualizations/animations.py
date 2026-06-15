# Functions for animating GCM output.

import numpy as np
import cftime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys

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

def animate_tracks_and_synop(tdicts, time_coords, contourf_da=None, contourf_kwargs=None,
                             contour_da=None, contour_kwargs=None, track_specs=None, 
                             time_range=None, save_prefix=None):
    """
    Animates the positions of meteorological features along trajectories alongside 
    synoptic fields.

    Arguments
    ---------
    tdicts : dict or iterable of dicts
        A dictionary (or iterable of dictionaries) whose keys are integers and whose values are 
        Track objects. 
    time_coords : xarray.DataArray
        Time coordinates spanning the range of times to animate. Typically ds['time'].
    contourf_da : xarray.DataArray or None, optional
        DataArray to be plotted as filled contours. Must have dimensions matching the 
        spatial coordinates used in plotting (typically 'lon' and 'lat').
    contourf_kwargs : dict or None, optional
        Keyword arguments to pass to ax.contourf(). Common options include:
        {'levels': np.arange(-20, 21, 2), 'cmap': 'coolwarm', 'extend': 'both'}
    contour_da : xarray.DataArray, list of DataArrays, or None, optional
        DataArray(s) to be plotted as line contours. Can be a single DataArray or a list
        of DataArrays.
    contour_kwargs : dict, list of dicts, or None, optional
        Keyword arguments to pass to ax.contour(). Can be a single dict (applied to all
        contour_da if a list) or a list of dicts (one per DataArray in contour_da).
        Common options include: {'levels': 20, 'colors': '0.5', 'linewidths': 0.75}
    track_specs : list of dicts or None, optional
        List of dictionaries specifying formatting for each track dictionary in tdicts.
        Each dict should contain: 'color', 'marker', and optionally 'size_scaling'.
        Example: [{'color': 'k', 'marker': 'o', 'size_scaling': None}, ...]
    time_range : tuple or None, optional
        If specified, gives the initial and final times of the animation as floats 
        (in days since reference date). If not specified, uses the full range of time_coords.
    save_prefix : str or None, optional
        Prefix to the animation's save-out filename. If None, saves as 'track_synop_anim.gif'.
    
    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        Animation object.
    """
    
    # if we have only one track dictionary, convert to a one-element list
    if type(tdicts) == dict:
        tdicts = [tdicts]
    
    # set up default track specifications
    if track_specs is None:
        track_specs = [{'color': 'k', 'marker': 'o', 'size_scaling': None} 
                       for _ in tdicts]
    
    # set up default plotting kwargs for filled contours
    if contourf_kwargs is None:
        contourf_kwargs = {}
    
    # handle contour_da and contour_kwargs - allow single or multiple DataArrays
    if contour_da is not None:
        if not isinstance(contour_da, list):
            contour_da = [contour_da]
        
        if contour_kwargs is None:
            contour_kwargs = [{} for _ in contour_da]
        elif not isinstance(contour_kwargs, list):
            # if a single dict is provided, apply it to all contour DataArrays
            contour_kwargs = [contour_kwargs for _ in contour_da]
        
        # ensure matching lengths
        if len(contour_da) != len(contour_kwargs):
            raise ValueError("contour_da and contour_kwargs must have the same length")
    else:
        contour_da = []
        contour_kwargs = []
    
    # time conversion utilities
    t_float = lambda t: cftime.date2num(t, "days since 0001-01-01")
    t_date = lambda t: cftime.num2date(t, "days since 0001-01-01", calendar="360_day")
    
    # find initial and final times
    if time_range is None:
        t_init, t_fin = t_float(time_coords.data[0]), t_float(time_coords.data[-1])
    else:
        t_init, t_fin = time_range[0], time_range[-1]
    
    delta_t = 0.25      # timestep length, in days
    num_t = int((t_fin - t_init) / 0.25) + 1

    # extract spatial coordinates from the first available DataArray
    spatial_da = contourf_da if contourf_da is not None else (contour_da[0] if contour_da else None)
    if spatial_da is None:
        raise ValueError("At least one of contourf_da or contour_da must be provided.")
    
    lon = spatial_da.lon
    lat = spatial_da.lat

    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')

    # initial frame
    i_f = 0
    time_fl = t_init + i_f*delta_t
    time_dt = t_date(time_fl)
    
    # plot filled contours
    cm = None
    if contourf_da is not None:
        cm = ax.contourf(lon, lat, contourf_da.sel(time=time_dt), **contourf_kwargs)
        fig.colorbar(cm, ax=ax)
    
    # plot line contours
    for da, kwargs in zip(contour_da, contour_kwargs):
        ct = ax.contour(lon, lat, da.sel(time=time_dt), **kwargs)
    
    # plot initial track positions
    for j, td in enumerate(tdicts):
        spec = track_specs[j]
        x, y, intensity = get_feature_snapshot(td, time_fl, size_scaling=spec.get('size_scaling'))
        ax.autoscale(enable=False)
        ax.scatter(x, y, s=intensity, c=spec['color'], marker=spec['marker'])
    
    ax.set_title(f'time = {time_fl} d')

    def update(i_f):
        ax.cla()
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        
        time_fl = t_init + i_f*delta_t
        time_dt = t_date(time_fl)

        # plot filled contours
        if contourf_da is not None:
            cm = ax.contourf(lon, lat, contourf_da.sel(time=time_dt), **contourf_kwargs)
        
        # plot line contours
        for da, kwargs in zip(contour_da, contour_kwargs):
            ct = ax.contour(lon, lat, da.sel(time=time_dt), **kwargs)
        
        # plot track positions
        for j, td in enumerate(tdicts):
            spec = track_specs[j]
            x, y, intensity = get_feature_snapshot(td, time_fl, size_scaling=spec.get('size_scaling'))
            ax.autoscale(enable=False)
            ax.scatter(x, y, s=intensity, c=spec['color'], marker=spec['marker'])
        
        ax.set_title(f'time = {time_fl} d')

    anim = FuncAnimation(fig, update, frames=num_t)
    
    if save_prefix is None:
        anim.save('track_synop_anim.gif')
    else:
        anim.save(f'{save_prefix}_track_synop_anim.gif')
    
    return anim