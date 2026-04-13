"""
ensemble_composites.py
----------------------
Feature-centred composites computed over perturbed tracks in an EnsembleSet.

Public API
~~~~~~~~~~
get_ensemble_multivar_composites        Top-level entry point.  Returns a
                                        Dataset with dimensions
                                        (t0[, time_offset], lat, lon) — one
                                        composite per ensemble initialisation
                                        time.

reduce_composites                       Pool all t0 slices into a single
                                        track-count-weighted mean.

Design notes
~~~~~~~~~~~~
Granularity
    The finest output granularity is one composite per ensemble (i.e. per t0).
    Tracks from all members of a given ensemble are pooled together before
    dividing by the total track count for that ensemble.  To get a single
    fully-pooled composite, pass the result to `reduce_composites`.

Parallelisation
    Workers are dispatched over ensemble members.  Each worker opens its own
    .nc file, processes all valid tracks for all time offsets, and returns
    plain numpy arrays — avoiding pickling xarray objects across the process
    boundary.  The main process then reduces partial sums within each t0
    group before dividing.

Canonical grid
    Built once from the first valid track in the first member before any
    variable or offset loop, then shared with every worker so that xr.concat
    never sees floating-point coordinate mismatches.

obs_list schema
    Each entry is a dict with:
        'file'      : str      – basename of the .nc file in each member dir
        'obs'       : callable – extracts a DataArray from the dataset
        'name'      : str      – variable name in the output Dataset
        'invert_SH' : bool     – (optional) per-variable SH sign flip
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import os
from os.path import join
import sys
import xarray as xr

import pdb

sys.path.append('..')
import observable_functions as of
from control_composites import (
    extract_window,
    _is_track_valid,
    _make_canonical_grid,
    _assign_canonical_coords,
)
from ensemble import EnsembleSet
from track import load_trackdict_from_file

GFDL_STORAGE = os.environ.get('GFDL_STORAGE', '')


# ---------------------------------------------------------------------------
# Worker: one ensemble member, all time offsets, one variable
# ---------------------------------------------------------------------------

def _accumulate_member_for_offsets(mem_dir, file_basename, obs, tdict,
                                   allowed_starts, time_offsets, lat_range,
                                   use_anom_fields, invert_SH, window_size,
                                   canonical_lat, canonical_lon):
    """
    Open one member's data file, iterate over all valid tracks, and
    accumulate composite windows for every time offset.

    Designed to run in a subprocess; returns plain numpy arrays to avoid
    pickling xarray objects across the process boundary.

    Parameters
    ----------
    mem_dir : str
        Path to the ensemble member directory.
    file_basename : str
        Basename of the data file within `mem_dir`.
    obs : callable
        Observable function (must be a picklable module-level function).
    tdict : dict
        Track dictionary for this member.
    allowed_starts : set
    time_offsets : list of float
    lat_range : tuple of float
    use_anom_fields : bool
    invert_SH : bool
    window_size : float
    canonical_lat : np.ndarray
    canonical_lon : np.ndarray

    Returns
    -------
    accum : dict
        ``{tau: {'data': np.ndarray or None, 'n': int}}``
        'data' is the running element-wise sum of window arrays (same shape
        as the canonical grid), or None if no valid tracks were found for
        that offset.
    """
    ds = xr.open_dataset(join(mem_dir, file_basename))
    try:
        da = obs(ds)
    except:
        print(ds)
        pdb.set_trace()
    if use_anom_fields:
        da = da - da.mean(dim='time')

    accum = {tau: {'data': None, 'n': 0} for tau in time_offsets}

    for tr in tdict.values():
        for tau in time_offsets:
            n_offset = int(4 * tau)
            result = _is_track_valid(tr, allowed_starts, lat_range, n_offset)
            if result is None:
                continue
            t_date, lon, lat = result

            da_window = extract_window(da, lon, lat, time=t_date,
                                       window_size=window_size)
            da_window = _assign_canonical_coords(
                da_window, canonical_lat, canonical_lon)

            if lat < 0:
                da_window = da_window.copy()
                da_window.data = np.flip(da_window.data, axis=0)
                if invert_SH:
                    da_window = -da_window

            data = da_window.values
            if accum[tau]['data'] is None:
                accum[tau]['data'] = data.copy()
            else:
                accum[tau]['data'] += data
            accum[tau]['n'] += 1

    ds.close()
    return accum


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def _run_members(members, file_basename, obs, allowed_starts, time_offsets,
                 lat_range, use_anom_fields, invert_SH, window_size,
                 canonical_lat, canonical_lon, n_workers):
    """
    Run _accumulate_member_for_offsets over all members, serially or in
    parallel.  Returns a list of accum dicts in the same order as `members`.
    """
    if n_workers > 1:
        raw = [None] * len(members)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _accumulate_member_for_offsets,
                    mem_dir, file_basename, obs, tdict, allowed_starts,
                    time_offsets, lat_range, use_anom_fields, invert_SH,
                    window_size, canonical_lat, canonical_lon
                ): idx
                for idx, (mem_dir, tdict, _) in enumerate(members)
            }
            for future in as_completed(futures):
                raw[futures[future]] = future.result()
        return raw
    else:
        return [
            _accumulate_member_for_offsets(
                mem_dir, file_basename, obs, tdict, allowed_starts,
                time_offsets, lat_range, use_anom_fields, invert_SH,
                window_size, canonical_lat, canonical_lon)
            for mem_dir, tdict, _ in members
        ]


def _reduce_to_t0(raw_results, members, time_offsets,
                  canonical_lat, canonical_lon, varname):
    """
    Sum partial accumulators within each t0 group and divide by total track
    count to produce one normalised composite DataArray per t0.

    Parameters
    ----------
    raw_results : list of dict
        Per-member accumulators ``{tau: {'data': ndarray or None, 'n': int}}``,
        in the same order as `members`.
    members : list of (mem_dir, tdict, t0) tuples
    time_offsets : list of float
    canonical_lat, canonical_lon : np.ndarray
    varname : str

    Returns
    -------
    da : xarray.DataArray
        Dims ``(t0[, time_offset], lat, lon)``.  Single-offset case has no
        ``time_offset`` dimension.
    n_feats : dict
        ``{t0: {tau: int}}`` — total track count per t0 per offset.
    """
    # Group member indices by t0
    from collections import defaultdict
    t0_to_indices = defaultdict(list)
    for idx, (_, _, t0) in enumerate(members):
        t0_to_indices[t0].append(idx)

    unique_t0s = sorted(t0_to_indices)
    nlat = len(canonical_lat)
    nlon = len(canonical_lon)
    ntau = len(time_offsets)

    t0_composites = []
    n_feats = {}

    for t0 in unique_t0s:
        indices = t0_to_indices[t0]
        n_feats[t0] = {}
        tau_composites = []

        for tau in time_offsets:
            total_data = None
            total_n    = 0

            for idx in indices:
                entry = raw_results[idx][tau]
                if entry['data'] is not None:
                    total_data = entry['data'] if total_data is None \
                                 else total_data + entry['data']
                    total_n += entry['n']

            n_feats[t0][tau] = total_n
            if total_n > 0:
                tau_composites.append(total_data / total_n)
            else:
                tau_composites.append(np.full((nlat, nlon), np.nan))

        t0_composites.append(np.stack(tau_composites, axis=0))   # (T, lat, lon)

    # Stack t0 slices -> (t0, T, lat, lon)
    arr = np.stack(t0_composites, axis=0)

    if len(time_offsets) == 1:
        # Drop the time_offset dimension for the single-offset case
        da = xr.DataArray(
            arr[:, 0, :, :],
            coords={'t0': unique_t0s, 'lat': canonical_lat, 'lon': canonical_lon},
            dims=['t0', 'lat', 'lon'],
            name=varname)
        n_feats = {t0: nf[time_offsets[0]] for t0, nf in n_feats.items()}
    else:
        da = xr.DataArray(
            arr,
            coords={'t0': unique_t0s, 'time_offset': time_offsets,
                    'lat': canonical_lat, 'lon': canonical_lon},
            dims=['t0', 'time_offset', 'lat', 'lon'],
            name=varname)

    return da, n_feats


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def get_ensemble_multivar_composites(ens_set, feature_type, obs_list,
                                      allowed_starts, time_offsets=None,
                                      lat_range=(30, 60),
                                      use_anom_fields=False, invert_SH=False,
                                      window_size=32, n_workers=1):
    """
    Build feature-centred composites for multiple atmospheric variables,
    pooled across members within each ensemble initialisation time (t0).

    Parameters
    ----------
    ens_set : EnsembleSet
        The ensemble set object.
    feature_type : str
        Feature type string (e.g. ``'vor_extrema'``), used to locate the
        per-member track pickle files.
    obs_list : list of dict
        Each entry specifies one variable:

        * ``'file'``      – str, basename of the .nc file in each member dir
        * ``'obs'``       – callable (module-level) that extracts a DataArray
        * ``'name'``      – str, variable name in the output Dataset
        * ``'invert_SH'`` – bool (optional), per-variable SH sign flip

    allowed_starts : iterable
        Times at which tracks are allowed to initialise.
    time_offsets : array-like of float or None, optional
        Offsets in days from peak intensity.  If None a single snapshot at
        ``time_offset=0.0`` is returned and the output has no
        ``'time_offset'`` dimension.
    lat_range : tuple of float, optional
        (min_abs_lat, max_abs_lat) filter on peak latitude.
    use_anom_fields : bool, optional
        If True, subtract the per-member time-mean before compositing.
    invert_SH : bool, optional
        Default SH sign-flip flag; overridable per variable via obs_list.
    window_size : float, optional
        Window size in degrees (default 32).
    n_workers : int, optional
        Number of parallel worker processes (default 1 = serial).

    Returns
    -------
    ds_comp : xarray.Dataset
        Dimensions: ``(t0[, time_offset], lat, lon)``.
    n_feats : dict
        ``{varname: {t0: {tau: int}}}`` — track counts per variable, per
        ensemble, per offset.  For the single-offset case the innermost dict
        is replaced by a plain int.
    """
    single_offset = time_offsets is None
    if single_offset:
        time_offsets = [0.0]
    time_offsets   = list(time_offsets)
    allowed_starts = set(allowed_starts)

    track_var  = feature_type[:feature_type.find("_")]
    track_base = f"{feature_type}.pickle"

    # Collect (mem_dir, tdict, t0) tuples across all ensembles
    members = []
    for t0, ens in ens_set.ensemble_dict.items():
        for i_mem, mem_dir in ens.mem_dirs.items():
            tdict = load_trackdict_from_file(
                join(mem_dir, track_var, track_base), mem_id=i_mem)
            members.append((mem_dir, tdict, t0))

    if not members:
        raise ValueError("No ensemble members found in EnsembleSet.")

    # Build the canonical grid once from the first variable and first member
    first_entry = obs_list[0]
    ds_ref = xr.open_dataset(join(members[0][0], first_entry['file']))
    da_ref = first_entry['obs'](ds_ref)
    canonical_grid = _make_canonical_grid(
        da_ref, members[0][1], allowed_starts, lat_range, window_size)
    ds_ref.close()
    canonical_lat, canonical_lon = canonical_grid

    data_vars = {}
    n_feats   = {}

    for entry in obs_list:
        varname       = entry['name']
        file_basename = entry['file']
        obs           = entry['obs']
        inv_sh        = entry.get('invert_SH', invert_SH)

        raw_results = _run_members(
            members, file_basename, obs, allowed_starts, time_offsets,
            lat_range, use_anom_fields, inv_sh, window_size,
            canonical_lat, canonical_lon, n_workers)

        da, nf = _reduce_to_t0(
            raw_results, members, time_offsets,
            canonical_lat, canonical_lon, varname)

        scalar_coords = [c for c in da.coords
                        if da[c].ndim == 0 and c not in da.dims]
        da = da.drop_vars(scalar_coords)

        data_vars[varname] = da
        n_feats[varname]   = nf

    return xr.Dataset(data_vars), n_feats


# ---------------------------------------------------------------------------
# Optional pooling across all t0
# ---------------------------------------------------------------------------

def reduce_composites(ds_comp, n_feats):
    """
    Pool all t0 slices into a single track-count-weighted mean.

    Parameters
    ----------
    ds_comp : xarray.Dataset
        Output of `get_ensemble_multivar_composites`.
    n_feats : dict
        ``{varname: {t0: {tau: int}}}`` or ``{varname: {t0: int}}`` as
        returned alongside ``ds_comp``.

    Returns
    -------
    ds_pooled : xarray.Dataset
        Same variables as ``ds_comp`` but with the ``t0`` dimension collapsed.
    n_feats_total : dict
        ``{varname: {tau: int}}`` (or ``{varname: int}`` for single-offset).
    """
    has_tau = 'time_offset' in ds_comp.dims
    data_vars     = {}
    n_feats_total = {}

    for varname in ds_comp.data_vars:
        da = ds_comp[varname]           # (t0[, time_offset], lat, lon)
        nf = n_feats[varname]           # {t0: {tau: int}} or {t0: int}

        if has_tau:
            time_offsets = da.time_offset.values.tolist()
            # weights shape: (n_t0, n_tau)
            weights = np.array(
                [[nf[t0][tau] for tau in time_offsets]
                 for t0 in da.t0.values],
                dtype=float)            # (T0, T)
            data    = da.values         # (T0, T, lat, lon)
            w_full  = weights[:, :, np.newaxis, np.newaxis]
        else:
            # weights shape: (n_t0,)
            weights = np.array([nf[t0] for t0 in da.t0.values], dtype=float)
            data    = da.values         # (T0, lat, lon)
            w_full  = weights[:, np.newaxis, np.newaxis]

        w_sum = np.nansum(np.where(np.isnan(data), 0, w_full), axis=0)
        d_sum = np.nansum(data * w_full, axis=0)
        with np.errstate(invalid='ignore'):
            mean_data = np.where(w_sum > 0, d_sum / w_sum, np.nan)

        if has_tau:
            data_vars[varname] = xr.DataArray(
                mean_data,
                coords={'time_offset': time_offsets,
                        'lat': ds_comp.lat.values,
                        'lon': ds_comp.lon.values},
                dims=['time_offset', 'lat', 'lon'],
                name=varname)
            n_feats_total[varname] = {
                tau: int(weights[:, i].sum())
                for i, tau in enumerate(time_offsets)}
        else:
            data_vars[varname] = xr.DataArray(
                mean_data,
                coords={'lat': ds_comp.lat.values, 'lon': ds_comp.lon.values},
                dims=['lat', 'lon'],
                name=varname)
            n_feats_total[varname] = int(weights.sum())

    return xr.Dataset(data_vars), n_feats_total


# ---------------------------------------------------------------------------
# Entry point / smoke test
# ---------------------------------------------------------------------------

def test():
    expname   = sys.argv[1]
    feat_type = sys.argv[2]
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    expdir = join(GFDL_STORAGE, expname)

    t0_list = np.array([361.0, 371.0])
    ens_set = EnsembleSet(directory=expdir, n_mems=10, t0_list=t0_list)

    obs_list = [
        {'file': 'atmos_6_hourly.nc', 'obs': of.surface_pressure,
         'name': 'ps',     'invert_SH': False},
        {'file': 'atmos_6_hourly.nc', 'obs': of.vorticity_850,
         'name': 'vor850', 'invert_SH': True},
        {'file': 'pv250.nc',    'obs': of.pv,
         'name': 'pv250', 'invert_SH': True},
    ]

    allowed_starts = t0_list + 9.0
    time_offsets   = np.arange(-5.0, 5.25, 0.25)

    # Per-t0 composites
    ds_t0, n_feats = get_ensemble_multivar_composites(
        ens_set, feat_type, obs_list, allowed_starts,
        time_offsets=time_offsets, lat_range=(30, 60), n_workers=n_workers)
    print("Per-t0 dataset:\n", ds_t0)

    # Fully pooled
    ds_pooled, n_pooled = reduce_composites(ds_t0, n_feats)
    print("\nPooled dataset:\n", ds_pooled)


if __name__ == '__main__':
    test()