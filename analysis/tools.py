import numpy as np
import warnings
import xarray as xr

def binned_average_timeseries(time_series_list, bin_width=None, num_bins=None, 
                              min_samples=1, return_std=True, center_metric='mean'):
    """
    Compute time-binned average of multiple time series with irregular sampling.
    (Courtesy of Claude Sonnet 4.5)
    
    Parameters
    ----------
    time_series_list : list of tuples
        List of (times, values) tuples, where each is a numpy array.
        Each tuple represents one time series.
    bin_width : float, optional
        Width of each time bin. If None, uses num_bins instead.
    num_bins : int, optional
        Number of bins to create. Only used if bin_width is None.
        Default is 100 if neither bin_width nor num_bins is specified.
    min_samples : int, optional
        Minimum number of samples required in a bin to compute statistics.
        Bins with fewer samples will have NaN values. Default is 1.
    return_std : bool, optional
        If True, return standard deviation. If False, return standard error.
        Default is True.
    
    Returns
    -------
    bin_centers : numpy array
        Center time point of each bin
    avg_values : numpy array
        Average value in each bin
    uncertainty : numpy array
        Standard deviation (if return_std=True) or standard error of each bin
    sample_counts : numpy array
        Number of samples contributing to each bin
    """
    
    # Input validation
    if not time_series_list:
        raise ValueError("time_series_list cannot be empty")
    
    for i, (times, values) in enumerate(time_series_list):
        if len(times) != len(values):
            raise ValueError(f"Series {i}: times and values must have same length")
        if len(times) == 0:
            raise ValueError(f"Series {i}: cannot be empty")
    
    # Find global time range across all series
    all_times = np.concatenate([ts[0] for ts in time_series_list])
    t_min, t_max = all_times.min(), all_times.max()
    
    # Determine binning strategy
    if bin_width is None:
        if num_bins is None:
            num_bins = 100
        bin_width = (t_max - t_min) / num_bins
    
    # Create bin edges and centers
    bins = np.arange(t_min, t_max + bin_width, bin_width)
    #bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_edges = bins[:-1]

    # Initialize output arrays
    n_bins = len(bin_edges)
    avg_values = np.full(n_bins, np.nan)
    uncertainty = np.full(n_bins, np.nan) if center_metric == 'mean' else np.full((n_bins, 2), np.nan)
    sample_counts = np.zeros(n_bins, dtype=int)
    
    # Compute statistics for each bin
    for i in range(n_bins):
        bin_start, bin_end = bins[i], bins[i+1]
        values_in_bin = []
        
        # Collect all values falling in this bin across all series
        for times, values in time_series_list:
            mask = (times >= bin_start) & (times < bin_end)
            values_in_bin.extend(values[mask])
        
        # Compute statistics if we have enough samples.
        # Statistics are different depending on whether we use the mean as our
        # central metric (which assumes Gaussian behavior implicitly) or the median
        # (which does not).
        if len(values_in_bin) >= min_samples:
            values_array = np.array(values_in_bin)
            if center_metric == 'mean':
                avg_values[i] = np.mean(values_array)
            elif center_metric == 'median':
                avg_values[i] = np.median(values_array)
            sample_counts[i] = len(values_in_bin)

            if center_metric == 'mean':
                if len(values_in_bin) > 1:
                    std_dev = np.std(values_array, ddof=1)
                    if return_std:
                        uncertainty[i] = std_dev
                    else:
                        uncertainty[i] = std_dev / np.sqrt(len(values_in_bin))
                else:
                    uncertainty[i] = 0.0
            elif center_metric == 'median':
                if len(values_in_bin) > 3:
                    quartiles = np.quantile(values_array, [0.25, 0.75])
                    uncertainty[i] = quartiles
                else:
                    uncertainty[i] = np.nan

    return bin_edges, avg_values, uncertainty, sample_counts

def binned_average_timeseries_bootstrap(time_series_list, bin_width=None, num_bins=None, 
                              min_samples=1, center_metric='mean'):
    """
    Compute time-binned average of multiple time series with irregular sampling.
    (Courtesy of Claude Sonnet 4.5)
    
    Parameters
    ----------
    time_series_list : list of tuples
        List of (times, values) tuples, where each is a numpy array.
        Each tuple represents one time series.
    bin_width : float, optional
        Width of each time bin. If None, uses num_bins instead.
    num_bins : int, optional
        Number of bins to create. Only used if bin_width is None.
        Default is 100 if neither bin_width nor num_bins is specified.
    min_samples : int, optional
        Minimum number of samples required in a bin to compute statistics.
        Bins with fewer samples will have NaN values. Default is 1.
    center_metric : str, optional
        Metric used to estimate the central tendency of the dataset. Options
        include "mean" or "median".
    
    Returns
    -------
    bin_centers : numpy array
        Center time point of each bin
    avg_values : numpy array
        Average value in each bin
    uncertainty : numpy array
        Standard deviation (if return_std=True) or standard error of each bin
    sample_counts : numpy array
        Number of samples contributing to each bin
    """
    
    # Input validation
    if not time_series_list:
        raise ValueError("time_series_list cannot be empty")
    
    for i, (times, values) in enumerate(time_series_list):
        if len(times) != len(values):
            raise ValueError(f"Series {i}: times and values must have same length")
        if len(times) == 0:
            raise ValueError(f"Series {i}: cannot be empty")
    
    # Find global time range across all series
    all_times = np.concatenate([ts[0] for ts in time_series_list])
    t_min, t_max = all_times.min(), all_times.max()
    
    # Determine binning strategy
    if bin_width is None:
        if num_bins is None:
            num_bins = 100
        bin_width = (t_max - t_min) / num_bins
    
    # Create bin edges and centers
    bins = np.arange(t_min, t_max + bin_width, bin_width)
    #bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_edges = bins[:-1]

    # Initialize output arrays
    n_bins = len(bin_edges)
    avg_values = np.full(n_bins, np.nan)
    uncertainty = np.full(n_bins, np.nan) if center_metric == 'mean' else np.full((n_bins, 2), np.nan)
    sample_counts = np.zeros(n_bins, dtype=int)
    
    # Compute statistics for each bin
    for i in range(n_bins):
        bin_start, bin_end = bins[i], bins[i+1]
        values_in_bin = []
        
        # Collect all values falling in this bin across all series
        for times, values in time_series_list:
            mask = (times >= bin_start) & (times < bin_end)
            values_in_bin.extend(values[mask])
        
        # Compute statistics if we have enough samples.
        # Statistics are different depending on whether we use the mean as our
        # central metric (which assumes Gaussian behavior implicitly) or the median
        # (which does not).
        if len(values_in_bin) >= min_samples:
            values_array = np.array(values_in_bin)
            if center_metric == 'mean':
                avg_values[i] = np.mean(values_array)
            elif center_metric == 'median':
                avg_values[i] = np.median(values_array)
            sample_counts[i] = len(values_in_bin)

    return bin_edges, avg_values, uncertainty, sample_counts


def compute_geodesic_distance(coord1, coord2):
    '''
    Returns the geodesic angular distance on the surface of a sphere, in degrees, between two
    pairs of lat-lon coordinates coord1 and coord2 (array-like of length 2).
    '''
    lon1, lat1 = np.deg2rad(coord1[0]), np.deg2rad(coord1[1])
    lon2, lat2 = np.deg2rad(coord2[0]), np.deg2rad(coord2[1])
    uvec1 = np.array([np.cos(lat1)*np.cos(lon1), np.cos(lat1)*np.sin(lon1), np.sin(lat1)])
    uvec2 = np.array([np.cos(lat2)*np.cos(lon2), np.cos(lat2)*np.sin(lon2), np.sin(lat2)])
    cosine = np.sum(uvec1*uvec2,axis=0)
    if type(cosine) == np.float64:
        if cosine > 1.0:
            cosine = 1.0
        elif cosine < -1.0:
            cosine = -1.0
    elif type(cosine) == np.array:
        cosine[np.where(cosine > 1.0)] = 1.0
        cosine[np.where(cosine < -1.0)] = -1.0
#    try:
#        with warnings.catch_warnings():
#            warnings.simplefilter("error")
#            dist = np.rad2deg(np.arccos(cosine))
#    except Warning as e:
#        print(f"Caught warning as an exception: {e}")
#        # print(cosine)
#        dist = np.rad2deg(np.arccos(cosine))
#        # print(dist)
    dist = np.rad2deg(np.arccos(cosine))
    if type(dist) == np.array:
        dist[np.isnan(dist)] = 0.0
    elif type(dist) == np.float64:
        dist = 0.0 if np.isnan(dist) else dist
    return dist

def extract_window(ds, center_lon, center_lat, time=None, window_size=32):
    """
    Extract a window around a lat-lon coordinate and recenter to (0, 0).
    Handles edge cases: poles, antimeridian crossing, and longitude wrapping.
    (Courtesy of Claude Sonnet 4.5)
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with 'lat' and 'lon' coordinates
    center_lat : float
        Latitude of the center point (-90 to 90)
    center_lon : float
        Longitude of the center point (-180 to 180 or 0 to 360)
    window_size : float, optional
        Size of the window in degrees (default: 20)
    
    Returns
    -------
    xarray.Dataset
        Sliced dataset with recentered coordinates
    """
    # dummy dataset for time selection
    if time != None:
        ds1 = ds.sel(time=time)
    else: 
        ds1 = ds

    # get nearest grid point to center lon/lat
    center_lon = np.float64(ds['lon'].sel(lon=center_lon, method='nearest').data)
    center_lat = np.float64(ds['lat'].sel(lat=center_lat, method='nearest').data)

    half_window = window_size / 2
    
    # Normalize center_lon to [-180, 180]
    center_lon = ((center_lon + 180) % 360) - 180
    
    # Detect if dataset uses [0, 360] or [-180, 180] longitude convention
    lon_vals = ds.lon.values
    uses_360 = np.any(lon_vals > 180)
    
    # Convert center_lon to dataset convention
    if uses_360 and center_lon < 0:
        center_lon = center_lon + 360
    
    # Calculate desired latitude bounds
    lat_min_desired = center_lat - half_window
    lat_max_desired = center_lat + half_window
    
    # Clamp to valid lat range for data extraction
    lat_min = max(lat_min_desired, -90)
    lat_max = min(lat_max_desired, 90)
    
    # Slice latitude
    ds_window = ds1.sel(lat=slice(lat_min, lat_max))
    
    # Pad with NaNs if window extends beyond poles
    needs_padding = lat_min_desired < -90 or lat_max_desired > 90
    
    if needs_padding:
        # Get the latitude resolution from the dataset
        lat_res = np.abs(np.diff(ds.lat.values)).mean()
        
        # Create full desired latitude range
        n_points = int(window_size / lat_res) + 1
        lat_full = np.linspace(lat_min_desired, lat_max_desired, n_points)
        
        # Reindex to include out-of-bounds latitudes (fills with NaN)
        ds_window = ds_window.reindex(lat=lat_full, method='nearest', tolerance=lat_res/2)
    
    # Handle longitude with potential antimeridian crossing
    lon_min = center_lon - half_window
    lon_max = center_lon + half_window
    
    # Check if window crosses antimeridian
    if uses_360:
        # For [0, 360] convention
        if lon_min < 0:
            lon_min = lon_min + 360
        if lon_max > 360:
            lon_max = lon_max - 360
        
        if lon_min > lon_max:  # Crosses 0/360
            ds_left = ds_window.sel(lon=slice(lon_min, 360))
            ds_right = ds_window.sel(lon=slice(0, lon_max))
            ds_window = xr.concat([ds_left, ds_right], dim='lon')
        else:
            ds_window = ds_window.sel(lon=slice(lon_min, lon_max))
    else:
        # For [-180, 180] convention
        if lon_max > 180:
            lon_max = lon_max - 360
        if lon_min < -180:
            lon_min = lon_min + 360
            
        if lon_min > lon_max:  # Crosses antimeridian
            ds_left = ds_window.sel(lon=slice(lon_min, 180))
            ds_right = ds_window.sel(lon=slice(-180, lon_max))
            ds_window = xr.concat([ds_left, ds_right], dim='lon')
        else:
            ds_window = ds_window.sel(lon=slice(lon_min, lon_max))
    
    # Recenter longitude coordinates, handling wraparound
    lon_centered = ds_window.lon.values - center_lon
    
    # Normalize to [-180, 180]
    lon_centered = ((lon_centered + 180) % 360) - 180
    
    # Sort longitudes to maintain monotonic ordering
    lon_sort_idx = np.argsort(lon_centered)
    ds_window = ds_window.isel(lon=lon_sort_idx)
    lon_centered = lon_centered[lon_sort_idx]
    
    # Recenter coordinates to (0, 0)
    ds_window = ds_window.assign_coords({
        'lat': ds_window.lat - center_lat,
        'lon': lon_centered
    })

    return ds_window

def get_lat_indices(arg, resolution='T85'):
    """ A flexible method for obtaining grid indices corresponding to certain ranges of latitudes in the GCM 
        output.

        Arguments:
        :param arg:     A string specifying descriptively which latitude range to return indices for. Options include:
            - "ml": midlatitudes, 30°-60° in both hemispheres
            - "ml_NH": midlatitudes in the northern hemisphere
            - "ml_SH": midlatitudes in the southern hemisphere
            - "et": extratropics, poleward of 20° in both hemispheres
            - "et_NH": extratropics in the northern hemisphere
            - "et_SH": extratropics in the southern hemisphere
        :type arg:      str
        
        :param resolution:  (Optional) A string specifying the grid resolution. By default set to T85.

        Returns: the desired indices
    """
    if resolution == 'T85':
        lats = T85_lats
    
    if arg == 'ml':         # mid-latitudes
        indices = np.where((np.abs(lats) > 30) & (np.abs(lats) < 60))[0]
    elif arg == 'ml_NH':        # mid-latitudes, NH
        indices = np.where((lats > 30) & (lats < 60))[0]
    elif arg == 'ml_SH':
        indices = np.where((lats > -60) & (lats < -30))[0]
    elif arg == 'et':
        indices = np.where(np.abs(lats) > 20)[0]
    elif arg == 'et_NH':
        indices = np.where(lats > 20)[0]
    elif arg == 'et_SH':
        indices = np.where(lats < -20)[0]
    elif arg == '45N':
        indices = np.where((lats < 50) & (lats > 40))[0]
    elif arg == '45S':
        indices = np.where((lats > -50) & (lats < -40))[0]
    
    return indices

# ----------------- CONSTANTS --------------------
T85_lats = np.array([-88.927734, -87.538704, -86.14147 , -84.742386, -83.3426  , -81.94247 ,
       -80.542145, -79.14171 , -77.741196, -76.34063 , -74.940025, -73.53939 ,
       -72.13873 , -70.73806 , -69.33737 , -67.936676, -66.535965, -65.135254,
       -63.73453 , -62.3338  , -60.93307 , -59.532337, -58.1316  , -56.730858,
       -55.330112, -53.929367, -52.528618, -51.127865, -49.727116, -48.326363,
       -46.925606, -45.52485 , -44.124092, -42.723335, -41.322575, -39.921818,
       -38.521057, -37.120293, -35.719532, -34.31877 , -32.918007, -31.517244,
       -30.11648 , -28.715715, -27.31495 , -25.914186, -24.51342 , -23.112656,
       -21.71189 , -20.311123, -18.910357, -17.50959 , -16.108824, -14.708057,
       -13.30729 , -11.906524, -10.505756,  -9.104989,  -7.704221,  -6.303454,
        -4.902687,  -3.501919,  -2.101151,  -0.700384,   0.700384,   2.101151,
         3.501919,   4.902687,   6.303454,   7.704221,   9.104989,  10.505756,
        11.906524,  13.30729 ,  14.708057,  16.108824,  17.50959 ,  18.910357,
        20.311123,  21.71189 ,  23.112656,  24.51342 ,  25.914186,  27.31495 ,
        28.715715,  30.11648 ,  31.517244,  32.918007,  34.31877 ,  35.719532,
        37.120293,  38.521057,  39.921818,  41.322575,  42.723335,  44.124092,
        45.52485 ,  46.925606,  48.326363,  49.727116,  51.127865,  52.528618,
        53.929367,  55.330112,  56.730858,  58.1316  ,  59.532337,  60.93307 ,
        62.3338  ,  63.73453 ,  65.135254,  66.535965,  67.936676,  69.33737 ,
        70.73806 ,  72.13873 ,  73.53939 ,  74.940025,  76.34063 ,  77.741196,
        79.14171 ,  80.542145,  81.94247 ,  83.3426  ,  84.742386,  86.14147 ,
        87.538704,  88.927734])

T85_lons = np.array([  0.     ,   1.40625,   2.8125 ,   4.21875,   5.625  ,   7.03125,
         8.4375 ,   9.84375,  11.25   ,  12.65625,  14.0625 ,  15.46875,
        16.875  ,  18.28125,  19.6875 ,  21.09375,  22.5    ,  23.90625,
        25.3125 ,  26.71875,  28.125  ,  29.53125,  30.9375 ,  32.34375,
        33.75   ,  35.15625,  36.5625 ,  37.96875,  39.375  ,  40.78125,
        42.1875 ,  43.59375,  45.     ,  46.40625,  47.8125 ,  49.21875,
        50.625  ,  52.03125,  53.4375 ,  54.84375,  56.25   ,  57.65625,
        59.0625 ,  60.46875,  61.875  ,  63.28125,  64.6875 ,  66.09375,
        67.5    ,  68.90625,  70.3125 ,  71.71875,  73.125  ,  74.53125,
        75.9375 ,  77.34375,  78.75   ,  80.15625,  81.5625 ,  82.96875,
        84.375  ,  85.78125,  87.1875 ,  88.59375,  90.     ,  91.40625,
        92.8125 ,  94.21875,  95.625  ,  97.03125,  98.4375 ,  99.84375,
       101.25   , 102.65625, 104.0625 , 105.46875, 106.875  , 108.28125,
       109.6875 , 111.09375, 112.5    , 113.90625, 115.3125 , 116.71875,
       118.125  , 119.53125, 120.9375 , 122.34375, 123.75   , 125.15625,
       126.5625 , 127.96875, 129.375  , 130.78125, 132.1875 , 133.59375,
       135.     , 136.40625, 137.8125 , 139.21875, 140.625  , 142.03125,
       143.4375 , 144.84375, 146.25   , 147.65625, 149.0625 , 150.46875,
       151.875  , 153.28125, 154.6875 , 156.09375, 157.5    , 158.90625,
       160.3125 , 161.71875, 163.125  , 164.53125, 165.9375 , 167.34375,
       168.75   , 170.15625, 171.5625 , 172.96875, 174.375  , 175.78125,
       177.1875 , 178.59375, 180.     , 181.40625, 182.8125 , 184.21875,
       185.625  , 187.03125, 188.4375 , 189.84375, 191.25   , 192.65625,
       194.0625 , 195.46875, 196.875  , 198.28125, 199.6875 , 201.09375,
       202.5    , 203.90625, 205.3125 , 206.71875, 208.125  , 209.53125,
       210.9375 , 212.34375, 213.75   , 215.15625, 216.5625 , 217.96875,
       219.375  , 220.78125, 222.1875 , 223.59375, 225.     , 226.40625,
       227.8125 , 229.21875, 230.625  , 232.03125, 233.4375 , 234.84375,
       236.25   , 237.65625, 239.0625 , 240.46875, 241.875  , 243.28125,
       244.6875 , 246.09375, 247.5    , 248.90625, 250.3125 , 251.71875,
       253.125  , 254.53125, 255.9375 , 257.34375, 258.75   , 260.15625,
       261.5625 , 262.96875, 264.375  , 265.78125, 267.1875 , 268.59375,
       270.     , 271.40625, 272.8125 , 274.21875, 275.625  , 277.03125,
       278.4375 , 279.84375, 281.25   , 282.65625, 284.0625 , 285.46875,
       286.875  , 288.28125, 289.6875 , 291.09375, 292.5    , 293.90625,
       295.3125 , 296.71875, 298.125  , 299.53125, 300.9375 , 302.34375,
       303.75   , 305.15625, 306.5625 , 307.96875, 309.375  , 310.78125,
       312.1875 , 313.59375, 315.     , 316.40625, 317.8125 , 319.21875,
       320.625  , 322.03125, 323.4375 , 324.84375, 326.25   , 327.65625,
       329.0625 , 330.46875, 331.875  , 333.28125, 334.6875 , 336.09375,
       337.5    , 338.90625, 340.3125 , 341.71875, 343.125  , 344.53125,
       345.9375 , 347.34375, 348.75   , 350.15625, 351.5625 , 352.96875,
       354.375  , 355.78125, 357.1875 , 358.59375])