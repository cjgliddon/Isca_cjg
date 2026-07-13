# Functions for generating composites of synoptic fields around weather features in the *control* run.
# TODO: remove "allowed_starts" as an argument?

from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import os
from os.path import join
import pickle
import sys
import xarray as xr


sys.path.append('..')
import observable_functions as of
from track import *
from experiment import PredictabilityExperiment

import pdb


def extract_window(ds, center_lon, center_lat, time=None, window_size=32):
    """
    Extract a window around a lat-lon coordinate and recenter to (0, 0).
    Handles edge cases: poles, antimeridian crossing, and longitude wrapping.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset with 'lat' and 'lon' coordinates
    center_lat : float
        Latitude of the center point (-90 to 90)
    center_lon : float
        Longitude of the center point (-180 to 180 or 0 to 360)
    time : cftime.datetime, optional
        If provided, selects this time step from ds before windowing.
    window_size : float, optional
        Size of the window in degrees (default: 32)

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Sliced dataset/array with recentered coordinates
    """
    # time selection
    if time is not None:
        ds1 = ds.sel(time=time)
    else:
        ds1 = ds

    # snap center to nearest grid point
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
        lat_res = np.abs(np.diff(ds.lat.values)).mean()
        n_points = int(window_size / lat_res) + 1
        lat_full = np.linspace(lat_min_desired, lat_max_desired, n_points)
        ds_window = ds_window.reindex(lat=lat_full, method='nearest', tolerance=lat_res / 2)

    # Handle longitude with potential antimeridian crossing
    lon_min = center_lon - half_window
    lon_max = center_lon + half_window

    if uses_360:
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
    lon_centered = ((lon_centered + 180) % 360) - 180

    lon_sort_idx = np.argsort(lon_centered)
    ds_window = ds_window.isel(lon=lon_sort_idx)
    lon_centered = lon_centered[lon_sort_idx]

    ds_window = ds_window.assign_coords({
        'lat': ds_window.lat - center_lat,
        'lon': lon_centered
    })

    return ds_window


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_track_valid(tr, allowed_starts, lat_range, n_offset, int_metric='nearby_maximum'):
    """
    Check whether a track passes the filtering criteria and return its peak
    coordinates, or None if the track should be excluded.

    Returns
    -------
    tuple (t_date, lon, lat) or None
    """
    t0 = tr.time[0]
    if allowed_starts is None:
        pass
    elif t0 not in allowed_starts:
        return None

    t, lon, lat = tr.get_peak_coords(convert_date=False, n_offset=n_offset, metric=int_metric)
    if np.isnan(t):
        return None

    if not (lat_range[0] <= np.abs(lat) <= lat_range[1]):
        return None

    t_date = cftime.num2date(t, "days since 0001-01-01", calendar='360_day')
    return t_date, lon, lat


def _accumulate_window(da_window, lat, da_comp, invert_SH):
    """
    Flip axes / sign for Southern Hemisphere features and accumulate into
    `da_comp`.  Returns updated `da_comp`.
    """
    if lat < 0:
        da_window = da_window.copy()
        da_window.data = np.flip(da_window.data, axis=0)
        if invert_SH:
            da_window = -da_window

    if isinstance(da_comp, int):
        return da_comp + da_window
    else:
        da_comp.data = da_comp.data + da_window.data
        return da_comp


def _make_canonical_grid(da, tdict, allowed_starts, lat_range, window_size, int_metric='nearby_maximum'):
    """
    Extract a single valid window to obtain a canonical (lat, lon) reference
    grid.  Every subsequent window will have its coordinates replaced with
    these values, preventing floating-point drift from causing xr.concat to
    treat slightly-different coordinate values as distinct grid points.

    n_offset=0 is always used here: the canonical grid depends only on the
    spatial structure of the grid near a valid track position, not on which
    time step is selected.

    Parameters
    ----------
    da : xarray.DataArray
        Any DataArray drawn from the dataset (used only for its grid).
    tdict : dict
        Track dictionary.
    allowed_starts : set
        Pre-converted set of allowed start times.
    lat_range : tuple of float
    window_size : float

    Returns
    -------
    canonical_lat : np.ndarray
    canonical_lon : np.ndarray
    """
    for tr in tdict.values():
        result = _is_track_valid(tr, allowed_starts, lat_range, n_offset=0, int_metric=int_metric)
        if result is None:
            continue
        t_date, lon, lat = result
        ref = extract_window(da, lon, lat, time=t_date, window_size=window_size)
        return ref.lat.values.copy(), ref.lon.values.copy()
    raise ValueError("No valid tracks found – cannot construct canonical grid.")


def _assign_canonical_coords(da_window, canonical_lat, canonical_lon):
    """Replace the coordinate arrays of *da_window* with the canonical grid."""
    return da_window.assign_coords(lat=canonical_lat, lon=canonical_lon)


# ---------------------------------------------------------------------------
# Single-offset, single-variable composite
# ---------------------------------------------------------------------------

def get_peak_composites(tdict, ds, obs, allowed_starts, int_metric='nearby_maximum',
                        lat_range=(30, 60), time_offset=0.0,
                        use_anom_fields=False, invert_SH=False,
                        window_size=32, canonical_grid=None):
    """
    Composite of a single atmospheric variable at a single time offset from
    peak intensity.

    Parameters
    ----------
    tdict : dict
        Track dictionary whose values are Track objects.
    ds : xarray.Dataset
        Dataset containing the model output.
    obs : callable
        Function that accepts `ds` and returns an xarray.DataArray for the
        variable/level of interest.
    allowed_starts : iterable
        Times at which tracks are allowed to initialise.
    lat_range : tuple of float, optional
        (min_abs_lat, max_abs_lat) filter on peak latitude.
    time_offset : float, optional
        Time offset in days from peak intensity (positive = after peak).
    use_anom_fields : bool, optional
        If True, subtract the time-mean before compositing.
    invert_SH : bool, optional
        If True, negate the field for Southern Hemisphere features.
    window_size : float, optional
        Window size in degrees (default 32).
    canonical_grid : tuple of (np.ndarray, np.ndarray) or None, optional
        Pre-computed (canonical_lat, canonical_lon) arrays.  Should be passed
        in by `get_composite_timeseries` or `get_multivar_composites` so that
        a single reference grid is shared across all offsets and variables.
        If None, the grid is derived from the first valid track (safe when
        calling this function directly for a single composite).

    Returns
    -------
    da_comp : xarray.DataArray
        Composite DataArray.
    n_feat : int
        Number of features that entered the composite.
    """
    try:
        da = obs(ds)
    except:
        pdb.set_trace()
    if use_anom_fields:
        da = da - da.mean(dim='time')

    n_offset = int(4 * time_offset)
    allowed_starts = set(allowed_starts)

    if canonical_grid is None:
        canonical_grid = _make_canonical_grid(
            da, tdict, allowed_starts, lat_range, window_size,
            int_metric=int_metric)
    canonical_lat, canonical_lon = canonical_grid

    da_comp = 0
    n_feat = 0
    for key, tr in tdict.items():
        # print(tr.lat[0])
        result = _is_track_valid(tr, allowed_starts, lat_range, n_offset, int_metric=int_metric)
        if result is None:
            continue
        t_date, lon, lat = result
        da_window = extract_window(da, lon, lat, time=t_date, window_size=window_size)
        da_window = _assign_canonical_coords(da_window, canonical_lat, canonical_lon)
        da_comp = _accumulate_window(da_window, lat, da_comp, invert_SH)
        n_feat += 1

    if n_feat == 0:
        raise ValueError("No valid tracks found - composite is empty.")

    da_comp = da_comp / n_feat
    return da_comp, n_feat


# ---------------------------------------------------------------------------
# Time-series of composites (multiple offsets)
# ---------------------------------------------------------------------------

def get_composite_timeseries(tdict, ds, obs, allowed_starts, time_offsets, 
                              int_metric='nearby_maximum', lat_range=(30, 60),
                              use_anom_fields=False, invert_SH=False,
                              window_size=32, n_workers=1,
                              canonical_grid=None):
    """
    Build a time series of composites at multiple offsets relative to each
    track's peak intensity.

    Parameters
    ----------
    tdict : dict
        Track dictionary whose values are Track objects.
    ds : xarray.Dataset
        Dataset containing the model output.
    obs : callable
        Function that returns an xarray.DataArray from `ds`.
    allowed_starts : iterable
        Times at which tracks are allowed to initialise.
    time_offsets : array-like of float
        Time offsets in days from peak intensity at which to build composites.
    lat_range : tuple of float, optional
        (min_abs_lat, max_abs_lat) filter on peak latitude.
    use_anom_fields : bool, optional
        If True, subtract the time-mean before compositing.
    invert_SH : bool, optional
        If True, negate the field for Southern Hemisphere features.
    window_size : float, optional
        Window size in degrees (default 32).
    n_workers : int, optional
        Number of parallel workers.  1 (default) runs serially.
    canonical_grid : tuple of (np.ndarray, np.ndarray) or None, optional
        Pre-computed canonical grid, as returned by `_make_canonical_grid`.
        If None, it is computed once here from the first valid track.  Pass
        it explicitly from `get_multivar_composites` so that all variables
        share an identical grid.

    Returns
    -------
    da_stacked : xarray.DataArray
        Composites stacked along a new 'time_offset' dimension.
    n_feats : dict
        Mapping {time_offset: n_feat} with the feature counts.
    """
    time_offsets = list(time_offsets)
    allowed_starts = set(allowed_starts)

    # Compute canonical grid once for this call, before the offset loop.
    if canonical_grid is None:
        da_ref = obs(ds)
        canonical_grid = _make_canonical_grid(
            da_ref, tdict, allowed_starts, lat_range, window_size,
            int_metric=int_metric)

    if n_workers > 1:
        results = _parallel_composite_timeseries(
            tdict, ds, obs, allowed_starts, int_metric, time_offsets, 
            lat_range, use_anom_fields, invert_SH, window_size,
            n_workers, canonical_grid)
    else:
        results = {}
        for tau in time_offsets:
            da_comp, n_feat = get_peak_composites(
                tdict, ds, obs, allowed_starts, int_metric=int_metric,
                lat_range=lat_range, time_offset=tau,
                use_anom_fields=use_anom_fields, invert_SH=invert_SH,
                window_size=window_size, canonical_grid=canonical_grid)
            results[tau] = (da_comp, n_feat)

    composites = [results[tau][0] for tau in time_offsets]
    n_feats = {tau: results[tau][1] for tau in time_offsets}

    da_stacked = xr.concat(
        composites,
        dim=xr.DataArray(time_offsets, dims='time_offset', name='time_offset'))
    return da_stacked, n_feats


def _parallel_composite_timeseries(tdict, ds, obs, allowed_starts, int_metric,
                                   time_offsets,  lat_range,
                                   use_anom_fields, invert_SH,
                                   window_size, n_workers, canonical_grid):
    """Run get_peak_composites in parallel over time_offsets."""
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                get_peak_composites,
                tdict, ds, obs, allowed_starts, int_metric,
                lat_range, tau, use_anom_fields, invert_SH,
                window_size, canonical_grid
            ): tau
            for tau in time_offsets
        }
        for future in as_completed(futures):
            tau = futures[future]
            results[tau] = future.result()
    return results


# ---------------------------------------------------------------------------
# Multi-variable composite
# ---------------------------------------------------------------------------

def get_multivar_composites(tdict, obs_list, allowed_starts, 
                            int_metric='nearby_maximum',
                            time_offsets=None,
                            lat_range=(30, 60),
                            use_anom_fields=False, invert_SH=False,
                            window_size=32, n_workers=1):
    """
    Build composites for multiple atmospheric variables (potentially from
    different datasets) and return them as a single xarray.Dataset.

    The canonical lat/lon grid is computed once before any variable or offset
    loop, ensuring that all composites share identical coordinate values and
    can be safely concatenated and merged.

    Parameters
    ----------
    tdict : dict
        Track dictionary whose values are Track objects.
    obs_list : list of dict
        Each entry is a dict with keys:

        * ``'ds'``        – xarray.Dataset containing the raw model output
        * ``'obs'``       – callable that extracts a DataArray from ``ds``
        * ``'name'``      – str, name to give the variable in the output Dataset
        * ``'invert_SH'`` – (optional) bool, overrides the function-level
          ``invert_SH`` for this variable (useful for vorticity vs. pressure)

    allowed_starts : iterable
        Times at which tracks are allowed to initialise.
    time_offsets : array-like of float or None, optional
        If provided, composites are built at each offset and the output
        Dataset gains a ``'time_offset'`` dimension.  If None, a single
        composite is built (``time_offset=0.0``).
    lat_range : tuple of float, optional
        (min_abs_lat, max_abs_lat) filter on peak latitude.
    use_anom_fields : bool, optional
        If True, subtract the time-mean before compositing (applied to all
        variables).
    invert_SH : bool, optional
        Default invert-sign flag for SH features; can be overridden per
        variable via ``obs_list[i]['invert_SH']``.
    window_size : float, optional
        Window size in degrees (default 32).
    n_workers : int, optional
        Number of parallel workers for the time-offset loop (per variable).

    Returns
    -------
    ds_comp : xarray.Dataset
        Dataset containing one DataArray per variable (and optionally a
        ``'time_offset'`` dimension).
    n_feats : dict
        ``{varname: {time_offset: n_feat}}`` (or ``{varname: n_feat}`` when
        no ``time_offsets`` are given).
    """
    single_offset = time_offsets is None
    if single_offset:
        time_offsets = [0.0]

    allowed_starts = set(allowed_starts)

    # Build the canonical grid once using the first variable's DataArray.
    # All variables share the same track positions and window_size, so one
    # grid suffices for the entire analysis.
    first_entry = obs_list[0]
    da_ref = first_entry['obs'](first_entry['ds'])
    canonical_grid = _make_canonical_grid(
        da_ref, tdict, allowed_starts, lat_range, window_size,
        int_metric=int_metric)

    data_vars = {}
    n_feats = {}

    for entry in obs_list:
        varname = entry['name']
        ds_var  = entry['ds']
        obs     = entry['obs']
        inv_sh  = entry.get('invert_SH', invert_SH)

        da_ts, nf = get_composite_timeseries(
            tdict, ds_var, obs, allowed_starts,
            time_offsets=time_offsets,
            int_metric=int_metric,
            lat_range=lat_range,
            use_anom_fields=use_anom_fields,
            invert_SH=inv_sh,
            window_size=window_size,
            n_workers=n_workers,
            canonical_grid=canonical_grid)

        scalar_coords = [c for c in da_ts.coords
                        if da_ts[c].ndim == 0 and c not in da_ts.dims]
        da_ts = da_ts.drop_vars(scalar_coords)

        if single_offset:
            data_vars[varname] = da_ts.isel(time_offset=0, drop=True)
            n_feats[varname]   = nf[0.0]
        else:
            data_vars[varname] = da_ts
            n_feats[varname]   = nf

    ds_comp = xr.Dataset(data_vars)
    return ds_comp, n_feats


# ---------------------------------------------------------------------------
# Entry point / smoke test
# ---------------------------------------------------------------------------

GFDL_STORAGE = os.environ['GFDL_STORAGE']
GFDL_DATA    = os.environ['GFDL_DATA']
pv_identfunc = lambda ds: ds['pv']

def test_default():

    print("Running 'test_default'...")

    expname = 'default'
    exp     = PredictabilityExperiment(expname, root_folder=GFDL_STORAGE)
    vars = ['MSLP', 'vor850', 'Z250', 'Z500']
    feature_types = ['cycs', 'acycs']
    allowed_starts = np.arange(369.0, 1450.0, 10.0)
    save_dir = exp.anly_dir

    # list of fields we'll include in the composites
    obs_list = [
        {
            'ds':   exp.load_climo_field("base_ps.nc"),
            'obs':  of.surface_pressure,
            'name': 'ps',
            'invert_SH': False,
        },
        {
            'ds':   exp.load_climo_field("base_vor850.nc"),
            'obs':  of.vorticity_850,
            'name': 'vor850',
            'invert_SH': True,
        },
        {
            'ds':   exp.load_climo_field("anomaly_height250.nc"),
            'obs':  of.height_250,
            'name': 'Z250_a',
            'invert_SH': False,
        },
        {
            'ds':   exp.load_climo_field("anomaly_height500.nc"),
            'obs':  of.height_500,
            'name': 'Z500_a',
            'invert_SH': False,
        },
    ]

    for var in vars:
        for ft in feature_types:
            ct = exp.get_control_tracks(var, ft)
            ds_comp, n_feats = get_multivar_composites(ct, obs_list, allowed_starts,
                                                       time_offsets=np.arange(-2.0, 2.25, 0.25))
            pickle.dump((ds_comp, n_feats),
                        open(join(save_dir, f"{var}_{ft}_control_composites.pickle"), 'wb'))
            print(f"Feature composites for {var}_{ft} saved")




def test():
    expname   = sys.argv[1]
    feat_type = sys.argv[2]

    expdir = join(GFDL_STORAGE, expname)

#    obs_list = [
#        {
#            'ds':   xr.open_dataset(join(expdir, 'postprocessed', 'base_ps_vor.nc')),
#            'obs':  of.surface_pressure,
#            'name': 'ps',
#            'invert_SH': False,
#        },
#        {
#            'ds':   xr.open_dataset(join(expdir, 'postprocessed', 'base_ps_vor.nc')),
#            'obs':  of.vorticity_850,
#            'name': 'vor850',
#            'invert_SH': True,
#        },
#    ]

    obs_list = [
        {
            'ds':   xr.open_dataset(join(expdir, 'postprocessed', 'base_anoms.nc')),
            'obs':  of.surface_pressure,
            'name': 'ps',
            'invert_SH': False,
        },
        {
            'ds':   xr.open_dataset(join(expdir, 'postprocessed', 'base_ps_vor.nc')),
            'obs':  of.vorticity_850,
            'name': 'vor850',
            'invert_SH': True,
        },
        {
            'ds':   xr.open_dataset(join(expdir, 'postprocessed', 'pv250_anoms.nc')),
            'obs':  pv_identfunc,
            'name': 'pv250',
            'invert_SH': True,
        },
        {
            'ds':   xr.open_dataset(join(expdir, 'postprocessed', 'pv500_anoms.nc')),
            'obs':  pv_identfunc,
            'name': 'pv500',
            'invert_SH': True,
        },
        {
            'ds':   xr.open_dataset(join(expdir, 'postprocessed', 'pv850_anoms.nc')),
            'obs':  pv_identfunc,
            'name': 'pv850',
            'invert_SH': True,
        }
    ]

    track_var = feat_type[:feat_type.find("_")]
    basename  = f"{feat_type}.pickle"
    tdc = load_trackdict_from_file(join(expdir, 'tracks', track_var, basename))

    allowed_starts = np.arange(361.0, 371.0, 0.25)
    time_offsets   = np.arange(-2.0, 1.25, 0.25)   # ±5 days in 6-hourly steps

    ds_comp, n_feats = get_multivar_composites(
        tdc, obs_list, allowed_starts,
        time_offsets=time_offsets,
        lat_range=(30, 60))

    print(ds_comp)
    print("Feature counts:", n_feats)

def main():
    expname   = sys.argv[1]

    expdir = join(GFDL_STORAGE, expname)

    obs_list = [
        {
            'ds':   xr.open_dataset(join(expdir, 'control', 'postprocessed', 'base_anoms.nc')),
            'obs':  of.surface_pressure,
            'name': 'ps',
            'invert_SH': False,
        },
        {
            'ds':   xr.open_dataset(join(expdir, 'control', 'postprocessed', 'base_ps_vor.nc')),
            'obs':  of.vorticity_850,
            'name': 'vor850',
            'invert_SH': True,
        },
#        {
#            'ds':   xr.open_dataset(join(expdir, 'control', 'postprocessed', 'anomaly_temp850.nc')),  # surface temp. anomalies
#            'obs':  of.temperature_850,
#            'name': 'T850_p',
#            'invert_SH': False,
#        },
#        {
#            'ds':   xr.open_dataset(join(expdir, 'control', 'postprocessed', 'base_temp850.nc')),  # surface temp.
#            'obs':  of.temperature_850,
#            'name': 'T850',
#            'invert_SH': False,
#        },
        {
            'ds':   xr.open_dataset(join(expdir, 'control', 'postprocessed', 'anomaly_height500.nc')), # Z500 anomalies
            'obs':  of.height_500,
            'name': 'Z500_p',
            'invert_SH': False,
        },
        {
            'ds':   xr.open_dataset(join(expdir, 'control', 'postprocessed', 'base_height500.nc')), # Z500 anomalies
            'obs':  of.height_500,
            'name': 'Z500',
            'invert_SH': False,
        },
#        {
#            'ds':   xr.open_dataset(join(expdir, 'postprocessed', 'vedd_tedd_850.nc')), # v'T'
#            'obs':  of.temperature_850,
#            'name': "v'T'850",
#            'invert_SH': True,
#        },
#        {
#            'ds':   xr.open_dataset(join(expdir, 'control', 'postprocessed', 'pv250.nc')),
#            'obs':  of.pv,
#            'name': 'pv250_b',
#            'invert_SH': True,
#        },
#        {
#            'ds':   xr.open_dataset(join(expdir, 'control', 'postprocessed', 'pv250_anoms.nc')),
#            'obs':  of.pv,
#            'name': 'pv250',
#            'invert_SH': True,
#        },
#        {
#            'ds':   xr.open_dataset(join(expdir, 'control', 'postprocessed', 'pv500_anoms.nc')),
#            'obs':  of.pv,
#            'name': 'pv500',
#            'invert_SH': True,
#        },
#        {
#            'ds':   xr.open_dataset(join(expdir, 'control', 'postprocessed', 'pv850_anoms.nc')),
#            'obs':  of.pv,
#            'name': 'pv850',
#            'invert_SH': True,
#        }     
    ]

    feat_types = ['vor850_cycs', 'vor850_acycs']

    for feat_type in feat_types:
        track_var = feat_type[:feat_type.find("_")]
        basename  = f"{feat_type}.pickle"
        tdc = load_trackdict_from_file(join(expdir, 'control', 'tracks', basename))

        allowed_starts = np.arange(360.25, 1440.25, 0.25)
        time_offsets   = np.arange(-5.0, 5.25, 0.25)   # ±5 days in 6-hourly steps

        ds_comp, n_feats = get_multivar_composites(
            tdc, obs_list, allowed_starts,
            time_offsets=time_offsets,
            lat_range=(30, 60))

        print(ds_comp)
        print("Feature counts:", n_feats)

        pickle.dump((ds_comp, n_feats),
                    open(join(expdir, 'control', 'postprocessed', f'{feat_type}_ctrl_composite.pickle'), 'wb'))

def main2():
    expname   = sys.argv[1]

    expdir = join(GFDL_STORAGE, expname)

    obs_list = [
        {
            'ds':   xr.open_dataset(join(expdir, 'postprocessed', 'dpv250_dt.nc')),
            'obs':  of.pv,
            'name': 'dpv250_dt',
            'invert_SH': True,
        },
        {
            'ds':   xr.open_dataset(join(expdir, 'postprocessed', 'dpv500_dt.nc')),
            'obs':  of.pv,
            'name': 'dpv500_dt',
            'invert_SH': True,
        },
        {
            'ds':   xr.open_dataset(join(expdir, 'postprocessed', 'dpv850_dt.nc')),
            'obs':  of.pv,
            'name': 'dpv850_dt',
            'invert_SH': True,
        }     
    ]

    feat_types = ['vor850_cycs', 'vor850_acycs']

    for feat_type in feat_types:
        track_var = feat_type[:feat_type.find("_")]
        basename  = f"{feat_type}.pickle"
        tdc = load_trackdict_from_file(join(expdir, 'tracks', track_var, basename))

        allowed_starts = np.arange(360.25, 1440.25, 0.25)
        time_offsets   = np.arange(-5.0, 5.25, 0.25)   # ±5 days in 6-hourly steps

        ds_comp, n_feats = get_multivar_composites(
            tdc, obs_list, allowed_starts,
            time_offsets=time_offsets,
            lat_range=(30, 60))

        print(ds_comp)
        print("Feature counts:", n_feats)

        pickle.dump((ds_comp, n_feats),
                    open(join(expdir, 'control', 'postprocessed', f'{feat_type}_ctrl_pvtend_composite.pickle'), 'wb'))


if __name__ == '__main__':
    test_default()
#    main2()