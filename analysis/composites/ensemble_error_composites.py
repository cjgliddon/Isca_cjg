import cftime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import os
from os.path import join
import pickle
import sys
import xarray as xr

sys.path.append('..')
from control_composites import (
    extract_window,
    _is_track_valid,
    _make_canonical_grid,
    _assign_canonical_coords,
)
from ensemble import EnsembleSet
import observable_functions as of
from track import load_trackdict_from_file

import pdb



def _accumulate_member_for_offsets(t0_mem, ctrl_path, bkgd_path, mem_dir, pert_basename, 
                                   obs, tdict, time_offsets, lat_range,
                                   use_anom_fields, window_size,
                                   canonical_lat, canonical_lon,
                                   err_metric='mae'):
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
    window_size : float
    canonical_lat : np.ndarray
    canonical_lon : np.ndarray
    err_metric : str

    Returns
    -------
    accum : dict
        ``{tau: {'data': np.ndarray or None, 'n': int}}``
        'data' is the running element-wise sum of window arrays (same shape
        as the canonical grid), or None if no valid tracks were found for
        that offset.
    """
    ctrl_da = obs(xr.open_dataset(ctrl_path))
    bkgd_da = pickle.load(open(bkgd_path, 'rb')) if bkgd_path is not None else None     # pickled object is a DataArray
    pert_da = obs(xr.open_dataset(join(mem_dir, pert_basename)))

    if use_anom_fields:
        for da in (ctrl_da, pert_da):
            da = da - da.mean(dim='time')

    accum = {tau: {'data': None, 'n': 0} for tau in time_offsets}

    for ctr in tdict.keys():
        for tau in time_offsets:
            n_offset = int(4 * tau)
            cresult = _is_track_valid(ctr, None, lat_range, n_offset)
            if cresult is None:
                continue
            ct_date, clon, clat = cresult
            # pdb.set_trace()

            cda_window = extract_window(ctrl_da, clon, clat, time=ct_date,
                                       window_size=window_size)
            cda_window = _assign_canonical_coords(
                cda_window, canonical_lat, canonical_lon)
            
            # Get coordinates of perturbed track at corresponding *real* time, if it exists
            for ptr in tdict[ctr]:

                # check if matching perturbed track exists at the specified time and is within the lat range
                ct_float = cftime.date2num(ct_date, 'days since 0001-01-01')
                if len(np.where(ptr.time == ct_float)[0]) != 0:
                    ip = np.where(ptr.time == ct_float)[0][0]
                else:
                    ip = None
                    continue
                plon, plat = ptr.lon[ip], ptr.lat[ip]
                # print(f"Tau, clat, plat = {(tau, clat, plat)}")
                if lat_range[0] <= np.abs(plat) <= lat_range[-1]:
                    pass
                else:
                    continue

                pda_window = extract_window(pert_da, plon, plat, time=ct_date, window_size=window_size)
                pda_window = _assign_canonical_coords(
                    pda_window, canonical_lat, canonical_lon)
                
                # get background error if we're using it
                if bkgd_da is not None:
                    # notice that we need to convert the time to "time since perturbation" since this is the 
                    # unit used for the time axis of bkgd_da
                    bda_window = extract_window(bkgd_da, plon, plat, time=ct_float-t0_mem, window_size=window_size)
                    bda_window = _assign_canonical_coords(bda_window, canonical_lat, canonical_lon)

                # Now, calculate error!

                if err_metric == 'mse':
                    err_window = (cda_window - pda_window)**2
                    if bkgd_da is not None:
                        err_window = err_window - bda_window**2
                elif err_metric == 'mae':
                    err_window = np.abs(cda_window - pda_window) 
                    if bkgd_da is not None:
                        err_window = err_window - bda_window
                else:
                    raise Exception("Error metric string not recognized")
                
                if clat < 0:
                    err_window = err_window.copy()
                    err_window.data = np.flip(err_window.data, axis=0)

                data = err_window.values
                if accum[tau]['data'] is None:
                    accum[tau]['data'] = data.copy()
                else:
                    accum[tau]['data'] += data
                accum[tau]['n'] += 1

    for da in (ctrl_da, pert_da):
        da.close()
    if bkgd_da is not None:
        bkgd_da.close()
#    pdb.set_trace()
    return accum

# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def _run_members(members, ctrl_path, bkgd_path, pert_basename, obs, time_offsets,
                 lat_range, use_anom_fields, window_size,
                 canonical_lat, canonical_lon, n_workers,
                 err_metric='mae'):
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
                    t0_mem, ctrl_path, bkgd_path, mem_dir, pert_basename, obs, tdict,
                    time_offsets, lat_range, use_anom_fields,
                    window_size, canonical_lat, canonical_lon,
                    err_metric = err_metric
                ): idx
                for idx, (mem_dir, tdict, t0_mem) in enumerate(members)
            }
            for future in as_completed(futures):
                raw[futures[future]] = future.result()
        return raw
    else:
        return [
            _accumulate_member_for_offsets(
                t0_mem, ctrl_path, bkgd_path, mem_dir, pert_basename, obs, tdict,
                time_offsets, lat_range, use_anom_fields,
                window_size, canonical_lat, canonical_lon)
            for mem_dir, tdict, t0_mem in members
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

def get_ensemble_multivar_composites(ens_set, md, obs_list,
                                      start_delta, time_offsets=None,
                                      lat_range=(30, 60),
                                      use_anom_fields=False,
                                      window_size=32, n_workers=1,
                                      err_metric='mae'):
    """
    Build feature-centred composites of error fields for multiple atmospheric 
    variables, pooled across members within each ensemble initialisation time (t0).

    Parameters
    ----------
    ens_set : EnsembleSet
        The ensemble set object.
    md : dict
        A dictionary whose keys are track objects representing control
        simulation tracks and whose values are lists of track objects
        representing perturbed simulation tracks which "match" the control.
    obs_list : list of dict
        Each entry specifies one variable:

        * ``'ctrl_file'`` - str, basename of the .nc file containing the control-run data
        * ``'pert_file'`` - str, basename of the .nc file containing the perturbed-run data
        * ``'bkgd_file'`` - str, basename of the .nc file containing the "background error" field (default to None)
        * ``'obs'``       - callable (module-level) that extracts a DataArray
        * ``'name'``      - str, variable name in the output Dataset

    start_delta : float
        The time difference between the start of each control track being
        composited and the initialization of the corresponding ensemble. 
    time_offsets : array-like of float or None, optional
        Offsets in days from peak intensity.  If None a single snapshot at
        ``time_offset=0.0`` is returned and the output has no
        ``'time_offset'`` dimension.
    lat_range : tuple of float, optional
        (min_abs_lat, max_abs_lat) filter on peak latitude.
    use_anom_fields : bool, optional
        If True, subtract the per-member time-mean before compositing.
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

    # Build the canonical grid from the first variable and first control track
    first_entry = obs_list[0]
    ds_ref = xr.open_dataset(join(ens_set.expdir, 'postprocessed', first_entry['ctrl_file']))
    da_ref = first_entry['obs'](ds_ref)
    ctd = {i: tr for i, tr in enumerate(md.keys())}
    # (we set allowed_starts to None because it's not relevant)
    canonical_grid = _make_canonical_grid(da_ref, ctd, None, lat_range, window_size)
    canonical_lat, canonical_lon = canonical_grid

    # Collect member tuples
    members = []
    for t0, ens in ens_set.ensemble_dict.items():
        # filter dictionary to contain only those tracks corresponding to a given ensemble
        t_subdict = {k: md[k] for k in md.keys() if k.time[0] == t0 + start_delta}
        for i_mem, mem_dir in ens.mem_dirs.items():
            # filter dictionary again!
            t_subsubdict = dict(t_subdict)
            for k, v in t_subdict.items():
                t_subsubdict[k] = [tr for tr in v if tr.mem_id == i_mem]
            members.append((mem_dir, t_subsubdict, t0))    

    data_vars = {}
    n_feats   = {}

    for entry in obs_list:
        varname       = entry['name']
        ctrl_basename = entry['ctrl_file']
        pert_basename = entry['pert_file']
        bkgd_basename = entry['bkgd_file']
        obs           = entry['obs']

        # load in control and background datasets
        ctrl_path = join(ens_set.expdir, 'postprocessed', ctrl_basename)
        bkgd_path = join(ens_set.expdir, 'postprocessed', bkgd_basename) if bkgd_basename is not None else None
        
        raw_results = _run_members(
            members, ctrl_path, bkgd_path, pert_basename, 
            obs, time_offsets,
            lat_range, use_anom_fields, window_size,
            canonical_lat, canonical_lon, n_workers,
            err_metric = err_metric)

        da, nf = _reduce_to_t0(
            raw_results, members, time_offsets,
            canonical_lat, canonical_lon, varname)

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

GFDL_STORAGE = os.environ['GFDL_STORAGE']

def test():
    expname     = sys.argv[1]
    feat_type   = sys.argv[2]
    n_workers   = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    expdir = join(GFDL_STORAGE, expname)
    t0_list = np.array([360.0, 370.0])
    es = EnsembleSet(directory=expdir, n_mems=10, t0_list=t0_list)

    obs_list = [
        {'ctrl_file': 'base_anoms.nc', 'pert_file': 'pert_anoms.nc', 'bkgd_file': 'ps_bkgderr_ts.pickle',
         'obs': of.surface_pressure, 'name': 'ps'},
        {'ctrl_file': 'base_ps_vor.nc', 'pert_file': 'pert_ps_vor.nc', 'bkgd_file': 'vor850_bkgderr_ts.pickle',
         'obs': of.vorticity_850, 'name': 'vor850'},
        {'ctrl_file': 'pv250.nc', 'pert_file': 'pv250.nc', 'bkgd_file': 'pv250_bkgderr_ts.pickle',
         'obs': of.pv, 'name': 'pv250'},
    ]
    start_delta = 9.0       # time difference between ensemble start and track start
    allowed_starts  = t0_list + start_delta
    time_offsets    = np.arange(-2.0, 1.25, 0.25)

    # construct match dictionary
    md = dict()
    for i, t_as in enumerate(allowed_starts):
        md.update(es.match_tracks(feat_type, (t_as, t_as), ensemble_start=t0_list[i]))     # only allow tracks to start at specified times

    # Error composites per t0
    ds_t0, n_feats = get_ensemble_multivar_composites(
        es, md, obs_list, start_delta,
        time_offsets=time_offsets, lat_range=(30, 60), n_workers=n_workers
    )
    print("Per-t0 dataset:\n", ds_t0)

    # Fully pooled
    ds_pooled, n_pooled = reduce_composites(ds_t0, n_feats)
    print("\nPooled dataset:\n", ds_pooled)
    pdb.set_trace()

def main():
    expname     = sys.argv[1]
    feat_types  = ["vor850_cycs", "vor850_acycs"]
    n_workers   = os.process_cpu_count()

    expdir = join(GFDL_STORAGE, expname)
    t0_list = np.arange(360.0, 720.0, 10.0)
    es = EnsembleSet(directory=expdir, n_mems=20, t0_list=t0_list)

    remove_background = True
    for feat_type in feat_types:
        obs_list = [
            {'ctrl_file': 'base_anoms.nc', 'pert_file': 'pert_anoms.nc', 'bkgd_file': 'ps_bkgderr_ts.pickle',
            'obs': of.surface_pressure, 'name': 'ps'},
            {'ctrl_file': 'base_ps_vor.nc', 'pert_file': 'pert_ps_vor.nc', 'bkgd_file': 'vor850_bkgderr_ts.pickle',
            'obs': of.vorticity_850, 'name': 'vor850'},
            {'ctrl_file': 'pv250.nc', 'pert_file': 'pv250.nc', 'bkgd_file': 'pv250_bkgderr_ts.pickle',
            'obs': of.pv, 'name': 'pv250'},
            {'ctrl_file': 'pv500.nc', 'pert_file': 'pv500.nc', 'bkgd_file': 'pv500_bkgderr_ts.pickle',
            'obs': of.pv, 'name': 'pv500'},
            {'ctrl_file': 'pv850.nc', 'pert_file': 'pv850.nc', 'bkgd_file': 'pv850_bkgderr_ts.pickle',
            'obs': of.pv, 'name': 'pv850'},
        ]

        if not remove_background:   # if we're not removing the background, then reset names of bkgd_file to none for each variable
            for item in obs_list:
                item['bkgd_file'] = None

        start_delta = 15.0       # time difference between ensemble start and track start
        allowed_starts  = t0_list + start_delta
        time_offsets    = np.arange(-4.0, 4.25, 0.25)

        # construct match dictionary
        md = dict()
        for i, t_as in enumerate(allowed_starts):
            md.update(es.match_tracks(feat_type, (t_as, t_as), ensemble_start=t0_list[i]))     # only allow tracks to start at specified times

        # Error composites per t0
        ds_t0, n_feats = get_ensemble_multivar_composites(
            es, md, obs_list, start_delta,
            time_offsets=time_offsets, lat_range=(30, 60), n_workers=n_workers
        )
        print("Per-t0 dataset:\n", ds_t0)

        # Fully pooled
        ds_pooled, n_pooled = reduce_composites(ds_t0, n_feats)
        print("\nPooled dataset:\n", ds_pooled)

        dump_filename = f'{feat_type}_bkgd_rel_err_composite.pickle' if remove_background else f'{feat_type}_err_composite.pickle'

        pickle.dump((ds_pooled, n_pooled),
                    open(join(expdir, 'postprocessed', dump_filename), 'wb'))

def main2():
    expname     = sys.argv[1]
    feat_types  = ["vor850_cycs", "vor850_acycs"]
    n_workers   = os.process_cpu_count()

    expdir = join(GFDL_STORAGE, expname)
    t0_list = np.arange(360.0, 1441.0, 10.0)
    es = EnsembleSet(directory=expdir, n_mems=10, t0_list=t0_list)

    remove_background = False
    for feat_type in feat_types:
        obs_list = [
            {'ctrl_file': 'dpv250_dt.nc', 'pert_file': 'dpv250_dt.nc', 'bkgd_file': 'pv250_bkgderr_ts.pickle',
            'obs': of.pv, 'name': 'pv250'},
            {'ctrl_file': 'dpv500_dt.nc', 'pert_file': 'dpv500_dt.nc', 'bkgd_file': 'pv500_bkgderr_ts.pickle',
            'obs': of.pv, 'name': 'pv500'},
            {'ctrl_file': 'dpv850_dt.nc', 'pert_file': 'dpv850_dt.nc', 'bkgd_file': 'pv850_bkgderr_ts.pickle',
            'obs': of.pv, 'name': 'pv850'},
        ]

        if not remove_background:   # if we're not removing the background, then reset names of bkgd_file to none for each variable
            for item in obs_list:
                item['bkgd_file'] = None

        start_delta = 9.0       # time difference between ensemble start and track start
        allowed_starts  = t0_list + start_delta
        time_offsets    = np.arange(-4.0, 4.25, 0.25)

        # construct match dictionary
        md = dict()
        for i, t_as in enumerate(allowed_starts):
            md.update(es.match_tracks(feat_type, (t_as, t_as), ensemble_start=t0_list[i]))     # only allow tracks to start at specified times

        # Error composites per t0
        ds_t0, n_feats = get_ensemble_multivar_composites(
            es, md, obs_list, start_delta,
            time_offsets=time_offsets, lat_range=(30, 60), n_workers=n_workers
        )
        print("Per-t0 dataset:\n", ds_t0)

        # Fully pooled
        ds_pooled, n_pooled = reduce_composites(ds_t0, n_feats)
        print("\nPooled dataset:\n", ds_pooled)

        dump_filename = f'{feat_type}_tend_err_composite.pickle'

        pickle.dump((ds_pooled, n_pooled),
                    open(join(expdir, 'postprocessed', dump_filename), 'wb'))

if __name__ == '__main__':
    main2()