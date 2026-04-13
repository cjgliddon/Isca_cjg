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
import observable_functions as of 
from ensemble import Ensemble, EnsembleSet
from tools import binned_average_timeseries, compute_geodesic_distance
from track import *

ErrorTimeSeries = namedtuple('ErrorTimeSeries', ['times', 'err', 'err_spread', 'sample_counts'])

def _process_track_pair(tr_c, tr_p, time_norm):
    """
    Worker function: computes squared intensity and position errors
    for a single (control track, perturbed track) pair.
    """
    t0_c = tr_c.time[0]
    t0 = np.max((tr_p.time[0], t0_c))
    tf = np.min((tr_p.time[-1], tr_c.time[-1]))

    p_ids = np.where((tr_p.time >= t0) & (tr_p.time <= tf))
    c_ids = np.where((tr_c.time >= t0) & (tr_c.time <= tf))

    lats_c, lons_c, ints_c = tr_c.lat[c_ids], tr_c.lon[c_ids], tr_c.intensity[c_ids]
    lats_p, lons_p, ints_p = tr_p.lat[p_ids], tr_p.lon[p_ids], tr_p.intensity[p_ids]

    times = tr_c.time[c_ids]
    if time_norm == "peak_intensity":
        i_max = np.argmax(np.abs(tr_c.intensity))
        times = times - tr_c.time[i_max]
    elif time_norm == "genesis":
        times = times - t0_c

    int_sqerr = (ints_c - ints_p) ** 2
    pos_sqerr = compute_geodesic_distance([lons_c, lats_c], [lons_p, lats_p]) ** 2
    if (pos_sqerr < 1e-3).any():
#        pdb.set_trace()
        print(pos_sqerr)
        print(tr_c.time[0])
#    pdb.set_trace()
#    if np.isnan(pos_sqerr).any():
#        pdb.set_trace()

    return (times, int_sqerr), (times, pos_sqerr)


def get_lagrangian_err_statistics(exp, feature_type, tstarts=None, time_norm="genesis",
                                  center_metric='median', filtering='etc_only',
                                  save_out=True, save_name=None, n_workers=None):
    delt = 9.0
    if tstarts is None:
        tstarts = np.asarray([t0ens + delt for t0ens in exp.start_times])
        print(tstarts)
    if center_metric == 'rmse':
        center_metric = 'mean'
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
                executor.submit(_process_track_pair, tr_c, tr_p, time_norm): (tr_c, tr_p)
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
                int_result, pos_result = _process_track_pair(tr_c, tr_p, time_norm)
                int_err_dat.append(int_result)
                pos_err_dat.append(pos_result)
                print(f"Track pair ({tr_c}, {tr_p.mem_id}) succeeded")
            except Exception as exc:
                print(f"Track pair ({tr_c}, {tr_p.mem_id}) failed: {exc}")

    print(f"Number of error time series: {len(int_err_dat)}")

    int_res = binned_average_timeseries(int_err_dat, bin_width=0.25, center_metric=center_metric)
    int_ets = ErrorTimeSeries(int_res[0], np.sqrt(int_res[1]), np.sqrt(int_res[2]), int_res[3])

    pos_res = binned_average_timeseries(pos_err_dat, bin_width=0.25, center_metric=center_metric)
    pos_ets = ErrorTimeSeries(pos_res[0], np.sqrt(pos_res[1]), np.sqrt(pos_res[2]), pos_res[3])

    if save_out:
        pickle.dump((pos_ets, int_ets), open(join(exp.expdir, save_name), 'wb'))

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
    es = EnsembleSet(directory, n_mems=20, t0_list = np.arange(360.0, 720.0, 10.0))
    for feature in features:
        pos_ets, int_ets = get_lagrangian_err_statistics(es, feature, center_metric='median', 
                                                         save_out=True, filtering='etc_only',
                                                         save_name=f"{feature}_lagerr_stats_mean-std.pickle")

# Testing
if __name__ == '__main__':
    main()