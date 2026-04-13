from cftime import num2date
import numpy as np
import sys

sys.path.append('..')
from tools import extract_window

methods = ["nearest_neighbor", "linear_interpolation", "nearby_maximum", "window_avg"]

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

def recompute_all(do_control=True, do_ensembles=True, time_range=None):
    pass

if __name__ == '__main__':
    recompute_all()