# Methods for computing Lagrangian error statistics.

from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import os
from os.path import join
import pickle
import sys
import xarray as xr

import pdb

sys.path.append('..')
from experiment import PredictabilityExperiment
import observable_functions as of 
from ensemble import Ensemble, EnsembleSet
from tools import binned_average_timeseries, compute_geodesic_distance
from track import *

ErrorTimeSeries = namedtuple('ErrorTimeSeries', ['times', 'err', 'err_spread', 'sample_counts'])

def _process_track_pair(tr_c, tr_p, time_norm, int_metric='TRACK'):
    """
    Worker function: computes squared intensity and position errors
    for a single (control track, perturbed track) pair.
    """
    t0_c = tr_c.time[0]
    t0 = np.max((tr_p.time[0], t0_c))
    tf = np.min((tr_p.time[-1], tr_c.time[-1]))

    p_ids = np.where((tr_p.time >= t0) & (tr_p.time <= tf))
    c_ids = np.where((tr_c.time >= t0) & (tr_c.time <= tf))

    lats_c, lons_c, ints_c = tr_c.lat[c_ids], tr_c.lon[c_ids], tr_c.intensity[int_metric][c_ids]
    lats_p, lons_p, ints_p = tr_p.lat[p_ids], tr_p.lon[p_ids], tr_p.intensity[int_metric][p_ids]

    times = tr_c.time[c_ids]
    if time_norm == "peak_intensity":
        i_max = np.argmax(np.abs(tr_c.intensity[int_metric]))
        times = times - tr_c.time[i_max]
    elif time_norm == "genesis":
        times = times - t0_c

    int_err = np.abs(ints_c - ints_p)
    pos_err = compute_geodesic_distance([lons_c, lats_c], [lons_p, lats_p])
#    if (pos_err < 1e-3).any():
#        print(pos_err)
#        print(tr_c.time[0])

    return (times, int_err), (times, pos_err)

def match_tracks(cts, ets, delta_tau=9.0, T=60, k=2, S=3):
    """
    Matches a set of control tracks to a set of ensemble tracks using the
    criteria of Froude et al. (2007). 

    Arguments
    ---------
    cts : dict
        Dictionary whose values are Track objects representing feature tracks
        in a control run of a predictability experiment.
    ets : dict
        Dictionary whose keys are floats representing ensemble initialization
        times and whose values are lists of dictionaries, each element of which
        contains the Track objects for an individual ensemble member.
    delta_tau : float, optional
        The offset between ensemble initialization and track genesis time, in
        days, for control tracks subject to matching. Default is 9.0.
    T : float, optional
    k : int, optional
    S : float, optional
        Threshold parameters used in track matching. Default values are
        T = 60, k = 2, S = 3.
    """
    init_times = np.asarray(list(ets.keys()))
    t_gen = init_times + delta_tau          # when do tracks generate?
    md = dict()

    for ck, ct in cts.items():
        time_c, lon_c, lat_c = ct.time, ct.lon, ct.lat
        if time_c[0] in t_gen:              # does control track generate at a "correct" time?
            et_dict_list = ets[time_c[0] - delta_tau]       # grab appropriate ensemble track dictionary list
            for etd in et_dict_list:
                for ek, et in etd.items():
                    time_e, lon_e, lat_e = et.time, et.lon, et.lat

                    # determine how many points overlap in time
                    common_times, it_overl_c, it_overl_e = np.intersect1d(time_c, time_e, return_indices=True)
                    it_overl = list(zip(it_overl_c, it_overl_e))

                    # check if time overlap criterion is satisfied
                    if 100*(2*len(it_overl)/(len(time_c) + len(time_e))) > T:
                        s_array = compute_geodesic_distance(
                            (lon_c[it_overl_c], lat_c[it_overl_c]),
                            (lon_e[it_overl_e], lat_e[it_overl_e])
                        )
                        s_to_compare = np.array(s_array[:k])
                        # print(s_to_compare)
                        # check if "geodesic closeness" criterion is satisfied
                        if len(np.where(s_to_compare > S)[0]) > 0:
                            pass
                        else:
                            if md.get(ct) == None:      # add track as key to dictionary if it isn't there already
                                md[ct] = [et]
                                key_to_append = ek
                            else:
                                md[ct].append(et)
                # print(f"Track {hash(tc)} matched")
    return md

def compute_all_dx_dy(md, time_norm="genesis"):
    """
    Given a dictionary of matched tracks, returns a dictionary whose keys are
    (normalized) track time coordinates and whose values are 2-element lists.
    These elements are themselves lists of all the position errors for all
    matched tracks coexisting at the corresponding time.

    Arguments
    ---------
    md : dict
        The matched-track dictionary
    time_norm : str, optional
        Describes how the time axis is normalized. Default is "genesis", in
        which the genesis time of each track is subtracted from that track's
        time coordinates. "peak_intensity" normalizes time with respect to the
        time at which the feature's intensity is maximized (use with caution).
    """
    r_earth = 6.371e+03 # km

    dxdy_scatters = dict()    # each dictionary key is a time after genesis; each value gives the x and y errors at that time
    for ct, et_list in md.items():
        ctime, clat, clon = ct.time - ct.time[0], ct.lat, ct.lon
        for (t, lat, lon) in zip(ctime, clat, clon):
            if t not in dxdy_scatters:
                dxdy_scatters[t] = [[], []]
#               print(f"added time {t} to dictionary")
            for et in et_list:
                t_idx = np.where(et.time - ct.time[0] == t)[0]
                if t_idx.shape == (1,):    # that is, if there's exactly one match:
                    t_idx = t_idx[0]
                    elat, elon = et.lat[t_idx], et.lon[t_idx]
                    dlat = elat - lat; dlon = elon - lon
                    dlon = (dlon + 180) % 360 - 180    # transform to interval [-180, 180)
                    dx = r_earth * np.cos(np.deg2rad(lat)) * np.deg2rad(dlon)
                    dy = r_earth * np.deg2rad(dlat)
                    dxdy_scatters[t][0].append(dx); dxdy_scatters[t][1].append(dy)

    return dxdy_scatters

def compute_all_dx_dy_dint(md, time_norm="genesis", int_metric='nearby_maximum'):
    """
    Given a dictionary of matched tracks, returns a dictionary whose keys are
    (normalized) track time coordinates and whose values are 3-element lists.
    These elements are themselves lists of all the position and intensity
    (x, y, int) errors for all matched tracks coexisting at the corresponding
    time.

    Arguments
    ---------
    md : dict
        The matched-track dictionary
    time_norm : str, optional
        Describes how the time axis is normalized. Default is "genesis", in
        which the genesis time of each track is subtracted from that track's
        time coordinates. "peak_intensity" normalizes time with respect to the
        time at which the feature's intensity is maximized (use with caution).
    int_metric : str, optional
        Metric for measuring track intensity. Default is "nearby_maximum".
    """
    r_earth = 6.371e+03 # km

    dxdydint_scatters = dict()    # each dictionary key is a time after genesis; each value gives the x and y errors at that time
    for ct, et_list in md.items():
        ctime, clat, clon, cint = ct.time - ct.time[0], ct.lat, ct.lon, ct.intensity[int_metric]
        for (t, lat, lon, i_c) in zip(ctime, clat, clon, cint):
            if t not in dxdydint_scatters:
                dxdydint_scatters[t] = [[], [], []]
#               print(f"added time {t} to dictionary")
            for et in et_list:
                t_idx = np.where(et.time - ct.time[0] == t)[0]
                if t_idx.shape == (1,):    # that is, if there's exactly one match:
                    t_idx = t_idx[0]
                    elat, elon = et.lat[t_idx], et.lon[t_idx]
                    dlat = elat - lat; dlon = elon - lon
                    dlon = (dlon + 180) % 360 - 180    # transform to interval [-180, 180)
                    dx = r_earth * np.cos(np.deg2rad(lat)) * np.deg2rad(dlon)
                    dy = r_earth * np.deg2rad(dlat)
                    dint = et.intensity[int_metric][t_idx] - i_c
                    dxdydint_scatters[t][0].append(dx)
                    dxdydint_scatters[t][1].append(dy)
                    dxdydint_scatters[t][2].append(dint)

    return dxdydint_scatters


def get_lagrangian_err_spread(exp, var, feature_type, init_times=None, delta_tau=9.0,
                              time_norm="genesis", int_metric='nearby_maximum',
                              T=60, k=2, S=3):
    """
    Computes the mean (signed) Lagrangian error and spread as functions of time
    for feature trajectories of specified type.

    Arguments
    ---------
    exp : PredictabilityExperiment
        The experiment whose Lagrangian predictability statistics are to be
        computed.
    var, feature_type : str, str
        The tracking variable and feature type; e.g. 'MSLP' and 'cycs'.
    init_times : iterable of floats or None, optional
        Initialization times of ensembles from which statistics should be
        computed. Default is None, in which case all existing ensembles are
        used.
    delta_tau : float, optional
        The time offset between track genesis and ensemble perturbation time,
        in days. Default is 9.0.
    time_norm : str, optional
        Describes how the time axis is normalized. Default is "genesis", in
        which the genesis time of each track is subtracted from that track's
        time coordinates. "peak_intensity" normalizes time with respect to the
        time at which the feature's intensity is maximized (use with caution).
    int_metric : str, optional
        Metric for measuring track intensity. Default is "nearby_maximum".
    T, S : float, optional
    k : int, optional
        Track-matching parameters (see `match_tracks` docstring for a
        description). Default values are T = 60, k = 2, S = 3.
    """
    cts = exp.get_control_tracks(var, feature_type)
    ets = exp.get_ensemble_tracks(var, feature_type, init_times=init_times)

    md = match_tracks(cts, ets, delta_tau=delta_tau, T=T, k=k, S=S)
    dxdydint_scatters = compute_all_dx_dy_dint(md, time_norm=time_norm)

    # construct arrays containing the data
    times = np.asarray(list(dxdydint_scatters.keys()))
    pos_err = np.zeros((len(times), 2))
    pos_var = np.zeros((len(times), 2))
    int_err = np.zeros(times.shape)
    int_var = np.zeros(times.shape)
    counts = np.zeros(times.shape)

    for t_idx, (t, (dx_list, dy_list, dint_list)) in enumerate(dxdydint_scatters.items()):
        dx_arr = np.asarray(dx_list)
        dy_arr = np.asarray(dy_list)
        dint_arr = np.asarray(dint_list)
        counts[t_idx] = len(dx_arr)
        mean_dx = np.mean(dx_arr); mean_dy = np.mean(dy_arr); mean_dint = np.mean(dint_arr)
        var_dx = np.var(dx_arr); var_dy = np.var(dy_arr); var_dint = np.var(dint_arr)
        pos_err[t_idx,:] = mean_dx, mean_dy
        pos_var[t_idx,:] = var_dx, var_dy
        int_err[t_idx] = mean_dint
        int_var[t_idx] = var_dint

    pos_result = ErrorTimeSeries(times, pos_err, pos_var, counts)
    int_result = ErrorTimeSeries(times, int_err, int_var, counts)
    return pos_result, int_result

    delt = 9.0
    if tstarts is None:
        tstarts = np.asarray([t0ens + delt for t0ens in exp.start_times])
        print(tstarts)
    if save_name is None:
        save_name = feature_type + '_lagerr_stats.pickle'

    # Build match dictionary sequentially (likely stateful)
    md = dict()
    for t0 in tstarts:
        md.update(exp.match_tracks(feature_type, (t0, t0), ensemble_start=t0-delt, filtering=filtering, S=4))


    # Flatten all (tr_c, tr_p) pairs for parallel dispatch
    pairs = [
        (tr_c, tr_p)
        for tr_c, tr_p_list in md.items()
        for tr_p in tr_p_list
    ]

    int_err_dat = []
    pos_err_dat = []

    # create control flow for if we want to do the parallelization
    if (n_workers is None) or (n_workers > 1):
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_track_pair, tr_c, tr_p, time_norm, int_metric=int_metric): (tr_c, tr_p)
                for tr_c, tr_p in pairs
            }
            for future in as_completed(futures):
                try:
                    int_result, pos_result = future.result()
                    int_err_dat.append(int_result)
                    pos_err_dat.append(pos_result)
                    tr_c, tr_p = futures[future]
                    print(f"Track pair ({tr_c}, {tr_p.mem_id}) succeeded")
                except Exception as exc:
                    tr_c, tr_p = futures[future]
                    print(f"Track pair ({tr_c}, {tr_p.mem_id}) failed: {exc}")
    else:
        for tr_c, tr_p in pairs:
            try:
                int_result, pos_result = _process_track_pair(tr_c, tr_p, time_norm, int_metric=int_metric)
                int_err_dat.append(int_result)
                pos_err_dat.append(pos_result)
                print(f"Track pair ({tr_c}, {tr_p.mem_id}) succeeded")
            except Exception as exc:
                print(f"Track pair ({tr_c}, {tr_p.mem_id}) failed: {exc}")

    print(f"Number of error time series: {len(int_err_dat)}")

    int_res = binned_average_timeseries(int_err_dat, bin_width=0.25, center_metric=center_metric)
    int_ets = ErrorTimeSeries(int_res[0], int_res[1], int_res[2], int_res[3])

    pos_res = binned_average_timeseries(pos_err_dat, bin_width=0.25, center_metric=center_metric)
    pos_ets = ErrorTimeSeries(pos_res[0], pos_res[1], pos_res[2], pos_res[3])

    if save_out:
        pickle.dump((pos_ets, int_ets), open(join(exp.expdir, 'postprocessed', save_name), 'wb'))

    return pos_ets, int_ets

def _get_pair_egr(tr_c, tr_p, time_norm, int_metric='nearby_maximum'):
    """
    Worker function: computes the error growth rate in position and intensity
    errors for a single (control track, perturbed track) pair.
    """
    t0_c = tr_c.time[0]
    t0 = np.max((tr_p.time[0], t0_c))
    tf = np.min((tr_p.time[-1], tr_c.time[-1]))

    p_ids = np.where((tr_p.time >= t0) & (tr_p.time <= tf))
    c_ids = np.where((tr_c.time >= t0) & (tr_c.time <= tf))

    lats_c, lons_c, ints_c = tr_c.lat[c_ids], tr_c.lon[c_ids], tr_c.intensity[int_metric][c_ids]
    lats_p, lons_p, ints_p = tr_p.lat[p_ids], tr_p.lon[p_ids], tr_p.intensity[int_metric][p_ids]

    times = tr_c.time[c_ids]
    if time_norm == "peak_intensity":
        i_max = np.argmax(np.abs(tr_c.intensity[int_metric]))
        times = times - tr_c.time[i_max]
    elif time_norm == "genesis":
        times = times - t0_c

    int_err = np.abs(ints_c - ints_p)
    pos_err = compute_geodesic_distance([lons_c, lats_c], [lons_p, lats_p])
    if (pos_err < 1e-3).any():
#        pdb.set_trace()
        print(pos_err)
        print(tr_c.time[0])

    die_dt = np.gradient(int_err, times)
    dpe_dt = np.gradient(pos_err, times)
    return times, int_err, pos_err

def get_lagrangian_err_growth(exp, feature_type, tstarts=None, time_norm="genesis",
                                int_metric='nearby_maximum', center_metric='median', filtering='etc_only',
                                save_out=True, save_name=None, n_workers=None):
    delt = 9.0
    if tstarts is None:
        tstarts = np.asarray([t0ens + delt for t0ens in exp.start_times])
        print(tstarts)
    if save_name is None:
        save_name = feature_type + '_lagerr_stats.pickle'

    # Build match dictionary sequentially (likely stateful)
    md = dict()
    for t0 in tstarts:
        md.update(exp.match_tracks(feature_type, (t0, t0), ensemble_start=t0-delt, filtering=filtering, S=4))


    # Flatten all (tr_c, tr_p) pairs for parallel dispatch
    pairs = [
        (tr_c, tr_p)
        for tr_c, tr_p_list in md.items()
        for tr_p in tr_p_list
    ]

    int_err_dat = []
    pos_err_dat = []

    # create control flow for if we want to do the parallelization
    if (n_workers is None) or (n_workers > 1):
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_get_pair_egr, tr_c, tr_p, time_norm, int_metric=int_metric): (tr_c, tr_p)
                for tr_c, tr_p in pairs
            }
            for future in as_completed(futures):
                try:
                    t, die_dt, dpe_dt = future.result()
                    int_err_dat.append((t, die_dt))
                    pos_err_dat.append((t, dpe_dt))
                    tr_c, tr_p = futures[future]
                    print(f"Track pair ({tr_c}, {tr_p.mem_id}) succeeded")
                except Exception as exc:
                    tr_c, tr_p = futures[future]
                    print(f"Track pair ({tr_c}, {tr_p.mem_id}) failed: {exc}")
    else:
        for tr_c, tr_p in pairs:
            try:
                t, die_dt, dpe_dt = _get_pair_egr(tr_c, tr_p, time_norm, int_metric=int_metric)
                int_err_dat.append((t, die_dt))
                pos_err_dat.append((t, dpe_dt))
                print(f"Track pair ({tr_c}, {tr_p.mem_id}) succeeded")
            except Exception as exc:
                print(f"Track pair ({tr_c}, {tr_p.mem_id}) failed: {exc}")

    print(f"Number of error time series: {len(int_err_dat)}")

    int_res = binned_average_timeseries(int_err_dat, bin_width=0.25, center_metric=center_metric)
    int_ets = ErrorTimeSeries(int_res[0], int_res[1], int_res[2], int_res[3])

    pos_res = binned_average_timeseries(pos_err_dat, bin_width=0.25, center_metric=center_metric)
    pos_ets = ErrorTimeSeries(pos_res[0], pos_res[1], pos_res[2], pos_res[3])

    if save_out:
        pickle.dump((pos_ets, int_ets), open(join(exp.expdir, 'postprocessed', save_name), 'wb'))

    return pos_ets, int_ets
    

GFDL_STORAGE = os.environ['GFDL_STORAGE']

def test():
    expname = sys.argv[1]
    feature = sys.argv[2]

    directory = join(GFDL_STORAGE, expname)
    es = EnsembleSet(directory, n_mems=20, t0_list=np.arange(660.0, 690.0, 10.0))
    pos_ets, int_ets = get_lagrangian_err_statistics(es, feature, filtering='etc_only', n_workers=1)
    print(pos_ets)
    print("")
    print(int_ets)

def main():
    expname = sys.argv[1]
    features = ['vor850_cycs', 'vor850_acycs']

    directory = join(GFDL_STORAGE, expname)
    es = EnsembleSet(directory, n_mems=20, t0_list = np.arange(360.0, 1450.0, 10.0))
    for feature in features:
        pos_ets1, int_ets1 = get_lagrangian_err_statistics(es, feature, int_metric='nearby_maximum', n_workers=1,
                                                         center_metric='median', save_out=True, filtering='etc_only',
                                                         save_name=f"{feature}_lagerr_stats_qtiles.pickle")
        pos_ets2, int_ets2 = get_lagrangian_err_statistics(es, feature, int_metric='nearby_maximum', n_workers=1,
                                                         center_metric='mean', save_out=True, filtering='etc_only',
                                                         save_name=f"{feature}_lagerr_stats_mean+std.pickle")

def egr(expname=None):
    if expname == None:
        expname = sys.argv[1]
    features = ['vor850_cycs', 'vor850_acycs']

    directory = join(GFDL_STORAGE, expname)
    es = EnsembleSet(directory, n_mems=20, t0_list = np.arange(360.0, 1450.0, 10.0))
    for feature in features:
        pos_ets1, int_ets1 = get_lagrangian_err_growth(es, feature, int_metric='nearby_maximum', n_workers=1,
                                                         center_metric='median', save_out=True, filtering='etc_only',
                                                         save_name=f"{feature}_lagerr_growth_stats_qtiles.pickle")
        pos_ets2, int_ets2 = get_lagrangian_err_growth(es, feature, int_metric='nearby_maximum', n_workers=1,
                                                         center_metric='mean', save_out=True, filtering='etc_only',
                                                         save_name=f"{feature}_lagerr_growth_stats_mean+std.pickle")


# Testing
if __name__ == '__main__':
    my_exp = PredictabilityExperiment('default')
    var = 'vor850'
    feature_type = 'cycs'
    cts = my_exp.get_control_tracks(var, feature_type)
    ets = my_exp.get_ensemble_tracks(var, feature_type, init_times=[360.0])
    md = match_tracks(cts, ets)
    dxdydint_scatter = compute_all_dx_dy_dint(md)
    pdb.set_trace()