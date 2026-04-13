import os
from os.path import join
import sys

sys.path.append('..')
sys.path.append('/home/cgliddon/pvtend/src')
import observable_functions as of
from ensemble import Ensemble, EnsembleSet
from tools import extract_window
from track import *
# import pvtend
from pvtend import compute_orthogonal_basis, project_field
from pvtend.decomposition.basis import (
    compute_quadrupole_basis, gram_schmidt_orthogonalize,
    PRENORM_PHI1, PRENORM_PHI2, PRENORM_PHI3, PRENORM_PHI4,
)
from pvtend.decomposition.smoothing import gaussian_smooth_nan

import pdb

GFDL_STORAGE = os.environ['GFDL_STORAGE'] 

R_EARTH = 6371000   # m 
prenorms = [PRENORM_PHI1, PRENORM_PHI2, PRENORM_PHI3, PRENORM_PHI4]

latmin = 25; latmax = 60
window_hw = 25
'''
# =============================================================================
Key routines of this module:

get_decomp_coefficients_single: calculates beta, ax, ay, and gamma for a single track as a function of time
get_decomp_coefficients_family: as above, except for all matching control-perturbed tracks of a set
accum_dcoeff_errors: helper function for...
calc_ensmean_dcoeff_errors:
'''

# extract patch of pv anom from track
# t_rel = 0.0
# ofunc         = of.pv
# ds_climo_path = join(exproot, 'postprocessed', 'pv250_mean.nc')
# ds_path       = join(exproot, 'postprocessed', 'pv250.nc')
# ds_tdv_path    = join(exproot, 'postprocessed', 'dpv250_dt.nc')

coeff_keys = ["beta", "ax", "ay", "gamma"]

def get_decomp_coefficients_single(ds_path, ds_climo_path, ds_tdv_path, tr, feat_type,
                                   ofunc=of.pv, latmin=25, latmax=60, window_hw=25,
                                   return_full_results=False):
    """
    Helper function for computing pv decomposition statistics for a single track.

    Arguments
    ---------
    ds_path : str
        Path to the pv dataset
    ds_climo_path : str
        Path to the pv climatology dataset
    ds_tdv_path : str
        Path to the Eulerian pv time tendency dataset
    tr : Track
        The track along which we want to calculate the coefficients
    feat_type : str
        The type of feature we're tracking. Either "cyc" or "acyc". 
    ofunc : callable
        A function for extracting the pv DataArray from the raw dataset. 
        Default is the function `pv` from the `observable_functions` module.
    latmin, latmax : float
        The minimum and maximum allowable latitudes (in both hemispheres) 
        for the feature center. Default is (25, 60).
    window_hw : float
        The half-width of the analysis window, in degrees.
    """
    # load datasets
    da = ofunc(xr.open_dataset(ds_path))
    da_mean = ofunc(xr.open_dataset(ds_climo_path))
    # pdb.set_trace()
    # floating point weirdness is going on with the latitudes, so we have to be careful
    da_anom = da.values - da_mean.isel(time=0).values
    # da_anom = da - da_mean.isel(time=0)
    da_tdv = ofunc(xr.open_dataset(ds_tdv_path))

    # instantiate dictionary containing decomposition results -- zeros for now
    coefficients_dict = {k: np.full(tr.time.shape, 0.0) for k in coeff_keys}
    if return_full_results:
        full_results = dict()

    for jt, t_num in enumerate(tr.time):
        clon, clat = tr.lon[jt], tr.lat[jt]
        if latmin <= np.abs(clat) <= latmax:
            t = cftime.num2date(t_num, 'days since 0001-01-01', calendar='360_day')
            da_loc = da.sel(time=t)
            da_tdv_loc = da_tdv.sel(time=t)
            da_anom_loc = da_anom[np.where(da.time.values == t)[0][0]]
            lat_vals = da_loc.lat.values
            lon_vals = da_loc.lon.values

            # Latitude mask (simple)
            lat_mask = np.abs(lat_vals - clat) <= window_hw
            lat_sel = lat_vals[lat_mask]

            # Longitude mask with wrapping
            lon_offset = ((lon_vals - clon + 180.0) % 360.0) - 180.0
            lon_mask = np.abs(lon_offset) <= window_hw

            # Extract and sort by relative longitude
            pv_patch  = da_loc.values[np.ix_(lat_mask, lon_mask)]
            ppv_patch = da_anom_loc[np.ix_(lat_mask, lon_mask)]
            x_rel_raw = lon_offset[lon_mask]
            sort_idx = np.argsort(x_rel_raw)
            x_rel = x_rel_raw[sort_idx]
            y_rel = lat_sel - clat
            pv_patch  = pv_patch[:, sort_idx]
            ppv_patch = ppv_patch[:, sort_idx]
            
            # --- Compute spatial gradients in SI (PVU/m) ---
            dlat_deg = abs(float(lat_vals[1] - lat_vals[0]))
            dlon_deg = abs(float(lon_vals[1] - lon_vals[0]))
            dy_m = dlat_deg * np.pi * R_EARTH / 180.0
            dx_m_per_lat = dlon_deg * np.pi * R_EARTH * np.cos(np.deg2rad(lat_sel)) / 180.0

            pv_dy = np.gradient(ppv_patch, dy_m, axis=0)
            pv_dx = np.zeros_like(ppv_patch)
            for j in range(len(lat_sel)):
                pv_dx[j] = np.gradient(ppv_patch[j], dx_m_per_lat[j])
            raw_phi4 = compute_quadrupole_basis(pv_dx, y_rel)

            # collect
            raw_fields = [ppv_patch, pv_dx, pv_dy, raw_phi4]
            
            # normalize and smooth
            GRID_SPACING = float(np.mean(np.abs(np.diff(x_rel))))
            SMOOTH_DEG   = 6.0
            normed_smoothed = [
                gaussian_smooth_nan(field * pn, smoothing_deg=SMOOTH_DEG, grid_spacing=GRID_SPACING)
                for field, pn in zip(raw_fields, prenorms)
            ]

            # check whether to do a positive or negative mask
            if (clat > 0 and feat_type == 'cyc') or (clat < 0 and feat_type == 'acyc'):
                mask_pos = True
            else:
                mask_pos = False
            
            print(f"Mask: {mask_pos}")

            mask = np.ones(pv_patch.shape, dtype=bool)
            for f in normed_smoothed:
                mask &= np.isfinite(f)
            if mask_pos:
                mask &= normed_smoothed[0] >= 0
            else:
                mask &= normed_smoothed[0] <= 0
            weights = np.where(mask, 1.0, 0.0)
            ortho, norms = gram_schmidt_orthogonalize(normed_smoothed, weights, mask)
            
            pv_dt_patch = da_tdv_loc.values[np.ix_(lat_mask, lon_mask)][:, sort_idx]
            basis = compute_orthogonal_basis(
                pv_anom=ppv_patch,
                pv_dx=pv_dx,
                pv_dy=pv_dy,
                x_rel=x_rel,
                y_rel=y_rel,
                mask_negative=False,
                mask_threshold=None,
                apply_smoothing=True,
                smoothing_deg=SMOOTH_DEG,
                grid_spacing=GRID_SPACING,
            )
            # --- Project tendency onto basis ---
            result = project_field(pv_dt_patch, basis)
            # add coefficients to dictionary
            for k, v in coefficients_dict.items():
                v[jt] = result[k]
            if return_full_results:
                full_results[t_num] = result

            del da_loc
            del da_anom_loc
        else:
            print("Track failed latitude test")

    coefficients_dict['time'] = tr.time

    del da
    del da_anom
    del da_mean
    del da_tdv

    return coefficients_dict if not return_full_results else full_results

def get_decomp_coefficients_family(ctrl_ds_path, ctrl_climo_path, ctrl_tdv_path, ensdir, pert_ds_fn, pert_tdv_fn, 
                                   ct, pts, t0_ens, feat_type, ofunc=of.pv, latmin=25, latmax=60, window_hw=25):
    """
    """
    ctrl_coeffs = get_decomp_coefficients_single(ctrl_ds_path, ctrl_climo_path, ctrl_tdv_path, ct, feat_type, 
                                                 ofunc=ofunc, latmin=latmin, latmax=latmax, window_hw=window_hw)
    pert_coeffs_all = dict()
    for pt in pts:
        imem = pt.mem_id
        memdir = join(ensdir, f"b{imem:02d}")
        pert_coeffs = get_decomp_coefficients_single(join(memdir, pert_ds_fn),
                                                     ctrl_climo_path,
                                                     join(memdir, pert_tdv_fn),
                                                     pt, feat_type,
                                                     ofunc=ofunc, latmin=latmin, latmax=latmax, window_hw=window_hw)
        pert_coeffs_all[imem] = pert_coeffs

    # finally, normalize all dictionaries by peak time of control track
    t_peak = ct.get_peak_coords()[0]
    for cdict in [ctrl_coeffs] + list(pert_coeffs_all.values()):
        cdict["time"] = cdict["time"] - t_peak

    print(t0_ens)
    print(ctrl_coeffs)
#    pdb.set_trace()
    return ctrl_coeffs, pert_coeffs_all
    
def _accum_dcoeff_errors(tot_err_dict, c_coeffs, p_coeffs_all):
    """
    """
    nt = tot_err_dict['norm_times']
    ct = c_coeffs['time']
    for imem, pc in p_coeffs_all.items():
        pt = pc['time']
        t_overl, ic, ip = np.intersect1d(ct, pt, return_indices=True)
        nt_mask = np.isin(nt, t_overl)
        at_mask = np.isin(t_overl, nt)
        for ck in coeff_keys:
            abs_err = np.abs(c_coeffs[ck][ic] - pc[ck][ip])
            try:
                tot_err_dict[ck][nt_mask] = tot_err_dict[ck][nt_mask] + abs_err[at_mask]
            except:
                pdb.set_trace()
        tot_err_dict['counts'][nt_mask] += 1

    return tot_err_dict

def calc_ensmean_dcoeff_errors(eset, feat_prefix, obs_name, obs_func=of.pv):
    """
    """
    # instantiate total error dictionary
    exproot = eset.expdir

    norm_times = np.arange(-4.0, 4.25, 0.25)
    tot_err_dict = dict({
        'norm_times': norm_times,
        'counts': np.zeros(norm_times.shape),
    })
    tot_err_dict.update({k: np.zeros(norm_times.shape) for k in coeff_keys})
    
    if feat_prefix in ['MSLP_cycs', 'vor850_cycs']:
        feat_type = 'cyc'
    elif feat_prefix in ['MSLP_acycs', 'vor850_acycs']:
        feat_type = 'acyc'
    else:
        raise Exception("Unrecognized feature-type prefix")

    # get names
    ds_fn     = f"{obs_name}.nc"
    ds_tdv_fn = f"d{obs_name}_dt.nc"
    ctrl_ds       = join(exproot, 'postprocessed', ds_fn)
    ctrl_ds_climo = join(exproot, 'postprocessed', f"{obs_name}_mean.nc")
    ctrl_ds_tdv   = join(exproot, 'postprocessed', ds_tdv_fn)

    # get match dictionary
    md = dict()
    delta_t = 9.0
    for t0 in eset.start_times:
        md = eset.match_tracks(feat_prefix, (t0+delta_t, t0+delta_t), t0)
        for ct, pts in md.items():
            ccoeffs, pcoeffs = get_decomp_coefficients_family(ctrl_ds, ctrl_ds_climo, ctrl_ds_tdv, 
                                                              eset.ensemble_dict[t0].path_to,
                                                              ds_fn, ds_tdv_fn, ct, pts, 
                                                              t0, feat_type, ofunc=obs_func)
            _accum_dcoeff_errors(tot_err_dict, ccoeffs, pcoeffs)

    for k in ['beta', 'ax', 'ay', 'gamma']:
        tot_err_dict[k] = tot_err_dict[k]/tot_err_dict['counts']

    return tot_err_dict


def control_test():

    expname = sys.argv[1]
    exproot = join(GFDL_STORAGE, expname)
    
    control_tracks = load_trackdict_from_file(join(exproot, 'tracks', 'vor850', 'vor850_cycs.pickle'))
    control_pv    = join(exproot, 'postprocessed', 'pv250.nc')
    control_climo = join(exproot, 'postprocessed', 'pv250_mean.nc')
    control_tdv   = join(exproot, 'postprocessed', 'dpv250_dt.nc')
    feat_type = 'cyc'

    coeff_dict = dict()
    for k, tr in control_tracks.items():
        coeff_dict[k] = get_decomp_coefficients_single(
            control_pv, control_climo, control_tdv, tr, feat_type=feat_type
        )
        print(f"Calculated decomposition statistics for track {k}")
        print(coeff_dict[k])
    print("Calculation complete")
    print(coeff_dict)

def err_test():

    expname = sys.argv[1]
    exproot = join(GFDL_STORAGE, expname)

    t0_list = np.arange(360.0, 430.0, 10.0)
    eset = EnsembleSet(exproot, n_mems=20, t0_list=t0_list)
    md = dict()
    feature_type = 'vor850_cycs'
    # loop over start times and add matches to dictionary
    for t0 in t0_list:
        md.update(eset.match_tracks(feature_type, (t0+9.0, t0+9.0), ensemble_start=t0))
    
    # do the error calculation
    err_dict = calc_ensmean_dcoeff_errors(eset, feature_type, 'pv500')
    print(err_dict)

    return

def main():

    expname = sys.argv[1]
    exproot = join(GFDL_STORAGE, expname)

    t0_list = np.arange(360.0, 430.0, 10.0)
    eset = EnsembleSet(exproot, n_mems=20, t0_list=t0_list)
    md = dict()
    feature_type = 'vor850_cycs'
    # loop over start times and add matches to dictionary
    for t0 in t0_list:
        md.update(eset.match_tracks(feature_type, (t0+9.0, t0+9.0), ensemble_start=t0))
    
    # do the error calculation
    err_dict = calc_ensmean_dcoeff_errors(eset, feature_type, 'pv500')
    print(err_dict)

    return

if __name__ == '__main__':
    err_test()
