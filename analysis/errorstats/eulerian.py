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
from tools import get_lat_indices

def lat_weighted_spatial_mean(ds):
    """ Calculates the latitude-weighted spatial mean of a dataset. """
    weights = np.cos(np.deg2rad(ds.lat))
    ds_weighted = ds.weighted(weights)
    ds_mean = ds_weighted.mean(dim=('lon', 'lat'))
    return ds_mean

def mae(ds1, ds2):
    """ Calculates the spatially-averaged mean absolute error between datasets. """
    error = np.abs(ds2 - ds1)
    err_mean = lat_weighted_spatial_mean(error).data
    return err_mean

def mse(ds1, ds2):
    """ Calculates the spatially-averaged mean square error between datasets. """
    sqerror    = (ds2 - ds1)**2
    sqerr_mean = lat_weighted_spatial_mean(sqerror).data
    return sqerror


#def calc_eulerian_error(exp, obsfunc,
#                        control_file='base_ps_vor.nc', region='ml', errfunc=mae, save_out=True, 
#                        save_name="eulerian_error_ts.pickle"):
#    """
#    Arguments
#    ---------
#    exp : EnsembleSet
#
#    obsfunc : callable
#
#    control_file : str, optional
#
#    save_out : bool, optional
#    """
#    expdir = exp.expdir 
#    control_da = obsfunc(xr.open_dataset(join(expdir, 'postprocessed', control_file)))  # load control data
#    control_da = control_da.isel(lat=get_lat_indices(region))
#
#    # initialize list of error time series
#    err_list = []
#
#    # loop over ensemble members
#    for t0, e in exp.ensemble_dict.items():
#        for i_mem, filename in e.data.items():
#            pert_da = obsfunc(xr.open_dataset(filename))
#            pert_da = pert_da.isel(lat=get_lat_indices(region))
#            err = errfunc(control_da, pert_da)
#            err_list.append(err)
#            print(f"Error for {t0}, member {i_mem} appended to list")
#    
#    # truncate all error time series to length of shortest time series
#    min_len = min(len(s) for s in err_list)
#    print(f"minimum time series length: {min_len}")
#    err_list = np.array([s[:min_len] for s in err_list])
#    mean_err = np.mean(err_list, axis=0)
#
#    if save_out:
#        pickle.dump((err_list, mean_err), open(
#            join(expdir, 'postprocessed', save_name), 'wb'
#        ))
#
#    return err_list, mean_err

def _process_member(filename, control_path, obsfunc, errfunc, region, control_file_is_path=True):
    """Worker function for parallel processing of a single ensemble member."""
    import xarray as xr
    from tools import get_lat_indices

    # Each worker reloads control (cheaply, since it's one file) to avoid pickling xarray objects
    control_da = obsfunc(xr.open_dataset(control_path))
    control_da = control_da.isel(lat=get_lat_indices(region))

    pert_da = obsfunc(xr.open_dataset(filename))
    pert_da = pert_da.isel(lat=get_lat_indices(region))
    return errfunc(control_da, pert_da)


def calc_eulerian_error(exp, obsfunc,
                        control_file='base_ps_vor.nc', region='ml', errfunc=mae,
                        save_out=True, save_name="eulerian_error_ts.pickle",
                        n_workers=None):
    expdir = exp.expdir
    control_path = join(expdir, 'postprocessed', control_file)

    # Collect all member filenames upfront
    if control_file[:2] != 'pv':
        member_files = [
            filename
            for e in exp.ensemble_dict.values()
            for filename in e.data.values()
            if filename is not None
        ]
    else:   # if we're calculating PV errors, need to select different files in ensemble folders
        member_files = [
            join(direc, control_file)
            for e in exp.ensemble_dict.values()
            for direc in e.mem_dirs.values()
        ]

    err_list = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_member, f, control_path, obsfunc, errfunc, region): f
            for f in member_files
        }
        for future in as_completed(futures):
            try:
                err_list.append(future.result())
                print(f"Processed: {futures[future]}")
            except Exception as exc:
                print(f"Member {futures[future]} failed: {exc}")

    min_len = min(len(s) for s in err_list)
    err_list = np.array([s[:min_len] for s in err_list])
    if errfunc == mae:
        mean_err = np.mean(err_list, axis=0)
    elif errfunc == mse:
        mean_err = np.sqrt(np.sum(err_list, axis=0))

    if save_out:
        pickle.dump((err_list, mean_err), open(
            join(expdir, 'postprocessed', save_name), 'wb'
        ))

    return err_list, mean_err

def _errbkgd_member(filename, control_path, obsfunc, err_metric='mae'):
    """Worker function for parallel processing of a single ensemble member."""
    import xarray as xr

    # Each worker reloads control (cheaply, since it's one file) to avoid pickling xarray objects
    control_da = obsfunc(xr.open_dataset(control_path))
    pert_da = obsfunc(xr.open_dataset(filename))

    if err_metric == 'mae':
        err = np.abs(control_da - pert_da)
    elif err_metric == 'rmse':
        err = (control_da - pert_da)**2
    
    # renormalize time coordinates
    time_since_pert = np.array(list(range(1, len(err.time) + 1))) * 0.25
    err = err.assign_coords(time=time_since_pert)
    return err

def calc_background_error(exp, obsfunc, control_file='base_ps_vor.nc', err_metric='mae',
                          save_out=True, save_name='eulerian_error_background.pickle',
                          n_workers=None):
    expdir = exp.expdir
    control_path = join(expdir, 'postprocessed', control_file)

    # Collect all member filenames upfront
    if control_file[:2] != 'pv':
        member_files = [
            filename
            for e in exp.ensemble_dict.values()
            for filename in e.data.values()
            if filename is not None
        ]
    else:   # if we're calculating PV errors, need to select different files in ensemble folders
        member_files = [
            join(direc, control_file)
            for e in exp.ensemble_dict.values()
            for direc in e.mem_dirs.values()
        ]

    err_sum = 0; n_errs = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_errbkgd_member, f, control_path, obsfunc, err_metric): f
            for f in member_files
        }
        for future in as_completed(futures):
            try:
                err_sum = err_sum + future.result()
                n_errs += 1
                print(f"Processed: {futures[future]}")
            except Exception as exc:
                print(f"Member {futures[future]} failed: {exc}")
    if err_metric == 'mae':
        mean_err = err_sum / n_errs
    elif err_metric == 'rmse':
        mean_err = np.sqrt(err_sum / n_errs)
    if save_out:
        pickle.dump(mean_err, open(
            join(expdir, 'postprocessed', save_name), 'wb'
        ))
    return mean_err, n_errs

GFDL_STORAGE = os.environ['GFDL_STORAGE']

def test():
    expname = 'frierson_default'
    directory = join(GFDL_STORAGE, expname)
    es = EnsembleSet(directory, n_mems=10, t0_list=np.arange(365.5, 385.5, 10.0))
    err_list, mean_err = calc_eulerian_error(es, of.vorticity_850, save_name='vor850_eulerr_ts.pickle')
    print(mean_err)
    print(len(err_list))

    mean_err, n_errs = calc_background_error(es, of.vorticity_850, save_out=False)
    print((mean_err, n_errs))


def main():
    expname = sys.argv[1]
    varia   = sys.argv[2]

    directory = join(GFDL_STORAGE, expname)
    if varia == 'vor850':
        ofunc = of.vorticity_850
        cfile = 'base_ps_vor.nc'
    elif varia == 'ps':
        ofunc = of.surface_pressure
        cfile = 'base_ps_vor.nc'
    elif (varia == 'pv250' or varia == 'pv500' or varia == 'pv850'):
        ofunc = of.pv
        cfile = f'{varia}.nc'


    es = EnsembleSet(directory, n_mems=20, t0_list=np.arange(360.0, 720.0, 10.0))
    err_list, mean_err = calc_eulerian_error(es, ofunc, control_file=cfile, errfunc=mae, save_name=f'{varia}_eulerr_ts.pickle', n_workers=1)
    mean_err, n_errs = calc_background_error(es, ofunc, control_file=cfile, err_metric='mae', save_name=f'{varia}_bkgderr_ts.pickle')
    print(f"num_errs for {(expname, varia)}: {n_errs}")


# Testing
if __name__ == '__main__':
    main()
