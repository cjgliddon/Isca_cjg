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
from experiment import PredictabilityExperiment
from tools import get_lat_indices
import observable_functions as of
from tools import var_dict
GFDL_DATA = os.environ['GFDL_DATA']

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
    return sqerr_mean


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
    err = errfunc(control_da, pert_da)
    return err


def calc_eulerian_error(exp, var,
                        metric='rmse', region='ml',
                        init_times=None, mems=None,
                        savename=None, n_workers=None):
    """
    Calculates timeseries of Eulerian-averaged error growth in a given
    atmospheric variable across ensembles.

    Arguments
    ---------
    exp : PredictabilityExperiment
        The experiment for which we wish to calculate the error timeseries.        
    var : str
        The variable whose errors we wish to calculate. 
    metric : str, optional
        Error metric to use. Options include:
        - "mae" (mean absolute error)
        - "rmse" (root mean square error)
        - "acc" (anomaly correlation coefficient) [#TODO]
        Default is "rmse".
    region : str, optional
        Region over which to calculate error. Default is "ml" (mid-latitudes).
        [#TODO: describe options more thoroughly]
    init_times : iterable of floats or None, optional
        If not None (default), the ensemble averaging is restricted to
        ensembles initialized at the given times. When None, all ensembles
        present are used. 
    mems : iterable of ints or None, optional
        If not None (default), the ensemble averaging is restricted to the
        listed ensemble members. When None, all ensemble members present are
        used.
    savename : str or None, optional
        Filename to which error data should be saved. If None, a default name
        is constructed based on the parameters passed to the function.
    n_workers : int or None, optional
        Number of simultaneous workers to use, if function is run as a parallel
        process.

    Returns
    -------
    err_list, mean_err
        An array of ensemble member error time series and their mean.
    """
    # path to control data
    control_path = join(exp.postproc, var_dict[var]['pp_file'])
    # paths to ensembles
    e_dirs = exp.get_ensemble_dirs(init_times=init_times, mems=mems)

    # determine "error function"
    if metric == 'rmse':
        errfunc = mse
    elif metric == 'mae':
        errfunc = mae
    elif metric == 'acc':
        raise Exception("Warning: ACC calculation not yet implemented. Please try another method.")

    # Collect all member filenames upfront
    if var[:2] != 'pv':
        member_files = [join(ed, 'atmos_6_hourly.nc') for ed in e_dirs]
    else:   # if we're calculating PV errors, need to select different files in ensemble folders
        member_files = [join(ed, f"{var}.nc") for ed in e_dirs]

    err_list = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_member, f, control_path, var_dict[var]['func'], errfunc, region): f
            for f in member_files
        }
        for future in as_completed(futures):
            try:
                err_list.append(future.result())
                print(f"Processed: {futures[future]}")
            except Exception as exc:
                print(f"Member {futures[future]} failed: {exc}")

    min_len = min(len(s) for s in err_list)
    err_list = np.array([s[-min_len:] for s in err_list])
    if metric == 'mae':
        mean_err = np.mean(err_list, axis=0)
    elif metric == 'rmse':
        mean_err = np.sqrt(np.mean(err_list, axis=0))

    savedir = join(GFDL_DATA, exp.name, 'analysis')
    if savename == None:
        savename = f"{var}_{metric}_timeseries.pickle"
    
    pickle.dump((err_list, mean_err), open(
        join(savedir, savename), 'wb'
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
    my_exp = PredictabilityExperiment('default')
    err_list, mean_err = calc_eulerian_error(my_exp, 'Z250', init_times=[360.0], mems=range(1, 11))
    print(mean_err)
    print(len(err_list))

def main():
    expname = sys.argv[1]
    varia   = sys.argv[2]
    my_exp = PredictabilityExperiment(expname)

    err_list, mean_err = calc_eulerian_error(my_exp, varia)
#    mean_err, n_errs = calc_background_error(es, ofunc, control_file=cfile, err_metric='mae', save_name=f'{varia}_bkgderr_ts.pickle')


# Testing
if __name__ == '__main__':
    test()
