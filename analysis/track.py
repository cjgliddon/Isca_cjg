import cftime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pickle
import xarray as xr
from collections import namedtuple

import pdb

from tools import binned_average_timeseries, compute_geodesic_distance, extract_window
import observable_functions as of

class Track:
    """
    An object representing the trajectory of a meteorological feature (e.g. a cyclone), containing
    information about its position and intensity as a function of time.
    """
    def __init__(self, arr, feature_type=None, var=None, unit=None, mem_id = None):
        """ Arguments:
            :param arr: a numpy array of shape (n, 4) which contains the track information. The first column of
                        the array gives time coordinates; the subsequent columns give the track's longitude, latitude,
                        and feature intensity.
            :param feature_type:    (optional) A string describing the feature being tracked.    
            :param var:             (optional) A string describing the variable used for the tracking.
            :param unit:            (optional) A string giving the units of the track intensity. 
        """
        self.time = arr[:,0]
        self.lon = arr[:,1]
        self.lat = arr[:,2]
        self.intensity = arr[:,3]
        self.length = len(self.time)

        if feature_type != None:
            self.ftype = feature_type
        if var != None:
            self.var = var
        if unit != None:
            self.int_unit = unit
        if mem_id != None:
            self.mem_id = mem_id

    def recompute_intensity(self, darr, method=None, return_dict=False):
        """
        Recomputes the intensity time series.
        """
        from intensity_validation.intensity import (methods as int_methods, get_along_track_intensity)

        if not hasattr(self, 'intensity_dict'):
            int_dict = dict()
        else:
            int_dict = self.intensity_dict

        if method is not None:
            ts = get_along_track_intensity(self, darr, method=method)
            int_dict.update({method: ts})
        else:
            int_dict.update(
                {method: get_along_track_intensity(self, darr, method=method) for method in int_methods}
            )
        self.intensity_dict = int_dict
        return int_dict if return_dict else None

    def get_peak_coords(self, convert_date=False, n_offset=0, metric=None):
        """
        Returns the time, longitude, and latitude of the track's peak intensity.

        Arguments
        ---------
        convert_date : boolean, optional
            If true, returns the time as a cftime object. If false (default), returns time as a
            float giving the number of days since 0001-01-01.
        n_offset : int, optional
            If nonzero, the function returns the coordinates offset from the peak by the given 
            number of timesteps. If the object does not exist at the given offset, the function
            returns a tuple of NaNs.

        """
        if metric is None:
            i_max = np.argmax(self.intensity) + n_offset
        else:
            pass

        if (i_max >= 0 and i_max < self.length):
            t = self.time[i_max]
            if convert_date == True:
                t = cftime.num2date(t, 'days since 0001-01-01', calendar='360_day')
            return t, self.lon[i_max], self.lat[i_max]
        else:
            return np.nan, np.nan, np.nan
    
    def get_peak_intensity(self, metric=None):
        """ Returns the peak intensity of the feature. """
        if metric is None:
            return np.max(self.intensity)
        else:
            int_ts
    
    def get_displacement(self):
        """ Returns the displacement in geodesic degrees of the feature. """
        disp = compute_geodesic_distance(
            (self.lon[0], self.lat[0]), (self.lon[-1], self.lat[-1])
        )
        return disp

# ------------- METHODS --------------

def load_trackdict_from_file(fn, mem_id = None, format="pickle"):
    """
    Returns a dictionary of Track objects from a given filename.
    
    :param str fn:          filename from which to load track data
    :param str format:      the format of the file corresponding to the parameter `fn`. Default
                            assumption is that `fn` is in `.pickle` format.
    """
    raw_dict = pickle.load(open(fn, "rb"))
    new_dict = dict()
    for key in raw_dict.keys():
        new_dict[key] = Track(raw_dict[key], mem_id=mem_id)
    return new_dict

def filter_tracks(td, filter_method="etc_only"):
    """
    Filters a track dictionary.
    """
    new_dict = dict(td)
    for k, tr in td.items():
        if filter_method == "etc_only":
            if (np.abs(tr.lat) < 20).any():
                del new_dict[k]
    return new_dict

def match_tracks(tdc, tdp, t0_range, T=60, k=2, S=3):
    """ 
    This function accepts two track dictionaries, one representing a "control" set and the other a
    "perturbed" set. It applies the matching method described in Froude et al. (Mon. Wea. Rev.,
    Feb. 2007) to match tracks in the control set with those in the perturbed set. It returns a new
    dictionary whose keys are the tracks in the control set and whose values are lists of tracks in the
    perturbed set which match the control-set track.

    :param dict tdc:    A dictionary of "control" tracks as loaded using the 
                        `load_trackdict_from_file` method.
    :param dict or array-like tdp:  
        May be either a single dictionary of "perturbed" tracks as loaded using the 
        `load_trackdict_from_file` method, or a list/array of such dictionaries.
    :param array-like t0_range:
        A 2-element list, tuple, or array giving the range of allowed *starting times* for tracks
        that are to be matched.
    :param int or float, optional T:

    :param int or float, optional k:

    :param int or float, optional S:
    """
    md = dict()     # initialize match dictionary

    # if we just have a single perturbed track dict, convert it into a 1-element list so the loop works
    if type(tdp) == dict:
        tdp = [tdp]

    for key_c in tdc.keys():
        tc = tdc[key_c]
        time_c, lon_c, lat_c = tc.time, tc.lon, tc.lat
        # check that track start is in the appropriate range
        if (time_c[0] >= t0_range[0]) and (time_c[0] <= t0_range[1]):
            # loop over perturbed track dictionaries
            for td in tdp:
                for key_p in td.keys():
                    tp = td[key_p]
                    time_p, lon_p, lat_p = tp.time, tp.lon, tp.lat
                    
                    # determine how many points overlap in time
                    common_times, it_overl_c, it_overl_p = np.intersect1d(time_c, time_p, return_indices=True)
                    it_overl = list(zip(it_overl_c, it_overl_p))

                    if 100*(2*len(it_overl)/(len(time_c) + len(time_p))) > T:
                        s_array = compute_geodesic_distance(
                            (lon_c[it_overl_c], lat_c[it_overl_c]),
                            (lon_p[it_overl_p], lat_p[it_overl_p])
                        )
                        s_to_compare = s_array[:k]
                        # check if "geodesic closeness" criterion is satisfied
                        if len(np.where(s_to_compare > S)[0]) > 0:
                            pass
                        else:
                            if md.get(tc) == None:      # add track as key to dictionary if it isn't there already
                                md[tc] = [tp]
                                key_to_append = key_p
                            else:
                                md[tc].append(tp)
            # print(f"Track {hash(tc)} matched")
    return md

def get_peak_composites(tdict, ds, obs, allowed_starts, lat_range=(30, 60), time_offset=0.0, use_anom_fields=False, invert_SH=False):
    """
    Generates data of composites of cyclones from a specified track dictionary at the coordinates
    of their peak intensity (or at a specified time offset from their peak intensity).

    Parameters
    ----------
    tdict : dict
        A track dictionary whose values must be Track objects.
    ds : xarray.Dataset
        A dataset containing the model output.
    obs : callable
        A function which extracts from the dataset an xarray.DataArray containing only the
        variable/level of interest. The `observable_functions` module contains such functions for
        many relevant variables.
    allowed_starts : iterable
        An iterable of times at which the tracks are allowed to initialize.
    invert_SH: bool, optional
        If True, takes the negative of the field before adding to the composite if the feature is
        in the Southern Hemisphere; this is necessary for fields which have opposite sign in the
        SH such as vorticity.
    """
    da = obs(ds)
    # extract fields to plot
    if use_anom_fields:
        da = da - da.mean(dim='time')

    n_offset = int(4*time_offset)

    da_comp = 0; n_feat = 0
    for key, tr in tdict.items():
        t0 = tr.time[0]
        t, lon, lat = tr.get_peak_coords(convert_date=False, n_offset=n_offset)
        if np.isnan(t) == False:
            t_date = cftime.num2date(t, "days since 0001-01-01", calendar='360_day')
            if np.abs(lat) <= lat_range[-1] and np.abs(lat) >= lat_range[0]:
                if t0 in allowed_starts:
                    da_window = extract_window(da, lon, lat, time=t_date)
                    if lat < 0:     # invert axes if in SH due to reflective symmetry about equator
                        da_window.data = np.flip(da_window.data, axis=0)
                        if invert_SH:   
                            da_window = -da_window    # flip the sign of the vorticity anomalies
                    if type(da_comp) != int:
                        da_comp.data = da_comp.data + da_window.data
                    else:  
                        da_comp = da_comp + da_window
                    n_feat += 1

    da_comp = da_comp/n_feat
    return da_comp, n_feat

CompTimeSeries = namedtuple("CompTimeSeries", ['time', 'data', 'n'])

def get_composite_time_series(tdict, ds, obs, allowed_starts, time_range, lat_range=(30, 60), use_anom_fields=False, invert_SH=False):
    """ """
    times = time_range
    data = np.empty(time_range.shape, dtype='object')
    n = np.empty(time_range.shape, dtype='int')
    for jt, t in enumerate(times):
        da, nt = get_peak_composites(tdict, ds, obs, allowed_starts, lat_range=lat_range, time_offset=t, use_anom_fields=use_anom_fields, invert_SH=invert_SH)
        data[jt] = da
        n[jt] = nt
    result = CompTimeSeries(times, data, n)
    return result

def get_err_bkgd_time_series(tdict, bkgd, allowed_starts, time_conversion_rule, time_range, lat_range=(30, 60)):
    """ """
    ds_key = list(bkgd.keys())[0]
    bkgd = bkgd[ds_key]

    times = time_range
    bkgd_times = bkgd.time.data
    data = np.empty(time_range.shape, dtype='object')
    n = np.empty(time_range.shape, dtype='int')
    for jt, t in enumerate(times):
        n_offset = int(4*t)
        
        da_comp = 0; n_feat = 0
        for key, tr in tdict.items():
            t0 = tr.time[0]
            t_peak, lon, lat = tr.get_peak_coords(convert_date=False, n_offset=n_offset)
            if np.isnan(t_peak) == False and t0 in allowed_starts:
                t_peak_conv = time_conversion_rule(t_peak, t0)
                if t_peak_conv <= bkgd_times[-1] and np.abs(lat) <= lat_range[-1] and np.abs(lat) >= lat_range[0]:
                    da_window = extract_window(bkgd, lon, lat, time=t_peak_conv)
                    if lat < 0:     # invert axes if in SH due to reflective symmetry about equator
                        da_window.data = np.flip(da_window.data, axis=0)
                    if type(da_comp) != int:
                        da_comp.data = da_comp.data + da_window.data
                    else:  
                        da_comp = da_comp + da_window
                    n_feat += 1
        data[jt] = da_comp/n_feat
        n[jt] = n_feat
    result = CompTimeSeries(times, data, n)
    return result

ErrorTimeSeries = namedtuple('ErrorTimeSeries', ['times', 'err', 'err_spread', 'sample_counts'])

if __name__ == "__main__":
    print("Got into Main")

    from os.path import join
    import pdb
    import pickle

    GFDL_STORAGE = "/orcd/data/talia_tb/001/aqua_gcm_runs"
    # exproot = join(GFDL_STORAGE, "frierson_moist", "es0_1.25")
    exproot = join(GFDL_STORAGE, "rotation", "omegax0.5")

    # load dataset and tracks
#    ds = xr.open_dataset(join(exproot, "postprocessed", "base_anoms.nc"))
#    ds2 = xr.open_dataset(join(exproot, "postprocessed", "base_thermo_ll.nc"))
    ds = xr.open_dataset(join(exproot, 'postprocessed', 'pv250.nc'))
    td = load_trackdict_from_file(join(exproot, "tracks", "vor850", "vor850_cycs.pickle"))
    ofunc = lambda ds: ds['pv']
#
    allowed_starts = np.arange(365.5, 1080.0, 10.0)
    comp = get_composite_time_series(td, ds, ofunc, allowed_starts, time_range=np.arange(-2.0, 1.25, 0.25))
    pickle.dump(comp, open("data/ps_composite_pv_default_va.pickle", "wb"))
    print("PV composite saved")
# 
#    td = load_trackdict_from_file(join(exproot, "tracks", "vor850", "vor850_acycs.pickle"))
#    comp = get_composite_time_series(td, ds, of.surface_pressure, allowed_starts, time_range=np.arange(-2.0, 1.25, 0.25))
#    pickle.dump(comp, open("data/ps_composite_ts_default_va.pickle", "wb"))
#    print("MSLP composite saved")
#
#    comp = get_composite_time_series(td, ds, of.vorticity_850, allowed_starts, time_range=np.arange(-2.0, 1.25, 0.25), invert_SH=True)
#    pickle.dump(comp, open("data/vor850_composite_ts_default_va.pickle", "wb"))
#    print("Vorticity composite saved")
#
#    comp = get_composite_time_series(td, ds2, of.temperature_850, allowed_starts, time_range=np.arange(-2.0, 1.25, 0.25))
#    pickle.dump(comp, open("data/T850_composite_ts_default_va.pickle", "wb"))
#    print("Temperature composite saved")
#
#    comp = get_composite_time_series(td, ds2, of.q_surf, allowed_starts, time_range=np.arange(-2.0, 1.25, 0.25))
#    pickle.dump(comp, open("data/qsurf_composite_ts_default_va.pickle", "wb"))
#    print("Sphum composite saved")
#
#    bkgd1 = xr.open_dataset("ad_hoc_scripts/ps_abs_error_time_series.nc")
#    bkgd2 = xr.open_dataset("ad_hoc_scripts/vor_abs_error_time_series.nc")
#
#    tcr = lambda t,t0: (t - t0 + 9)
#    comp = get_err_bkgd_time_series(td, bkgd1, allowed_starts, tcr, time_range=np.arange(-2.0, 1.25, 0.25))
#    pickle.dump(comp, open("data/ps_abs_error_composite_ts_va.pickle", "wb"))
#    print("ps background composite saved")
#
#    comp = get_err_bkgd_time_series(td, bkgd2, allowed_starts, tcr, time_range=np.arange(-2.0, 1.25, 0.25))
#    pickle.dump(comp, open("data/vor_abs_error_composite_ts_va.pickle", "wb"))
#    print("vor background composite saved")
#
#    # script for calculating Lagrangian error statistics
#    tracked_var = 'vor850'
#    feature_type = 'cycs'
#    track_basepath = f"{tracked_var}_{feature_type}.pickle"
#    
#    control_tracks = load_trackdict_from_file(join(exproot, "tracks", tracked_var, track_basepath))
#
#    start_times = np.arange(365.5, 1440.0, 10.0)
#    nmems = 10
#    md_full = dict()
#    for t in start_times:
#        perturbed_tracks = []
#        for n in range(1, nmems+1):
#            perturbed_tracks.append(
#                load_trackdict_from_file(join(exproot, "ensembles", str(t), "spec_mag_0.02", f"b{n:02d}", tracked_var, track_basepath))
#            )
#        md = match_tracks(control_tracks, perturbed_tracks, t0_range=(t+9.0, t+9.0))
#        md_full.update(md)
#
#    lagrangian_errors = get_lagrangian_err_statistics(md_full, center_metric="rmse")
#    pickle.dump(lagrangian_errors, open("data/vor850_cycs_lagrangian_errors_rmse.pickle", "wb"))
#
#    tracked_var = 'vor850'
#    feature_type = 'acycs'
#    track_basepath = f"{tracked_var}_{feature_type}.pickle"
#    
#    control_tracks = load_trackdict_from_file(join(exproot, "tracks", tracked_var, track_basepath))
#
#    start_times = np.arange(365.5, 1440.0, 10.0)
#    nmems = 10
#    md_full = dict()
#    for t in start_times:
#        perturbed_tracks = []
#        for n in range(1, nmems+1):
#            perturbed_tracks.append(
#                load_trackdict_from_file(join(exproot, "ensembles", str(t), "spec_mag_0.02", f"b{n:02d}", tracked_var, track_basepath))
#            )
#        md = match_tracks(control_tracks, perturbed_tracks, t0_range=(t+9.0, t+9.0))
#        md_full.update(md)
#
#    lagrangian_errors = get_lagrangian_err_statistics(md_full, center_metric="rmse")
#    pickle.dump(lagrangian_errors, open("data/vor850_acycs_lagrangian_errors_rmse.pickle", "wb"))