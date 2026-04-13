import numpy as np
import xarray as xr
from os.path import join
import sys
import pdb
import cftime

sys.path.append("..")
import observable_functions as of

GFDL_STORAGE="/orcd/data/talia_tb/001/aqua_gcm_runs"
exproot = join(GFDL_STORAGE, "frierson_moist", "es0_0.75")

fn1 = join(exproot, "control", "postprocessed", "base_ps_vor.nc")
da_ctrl = xr.open_dataset(fn1)['ps']

t0_ens = float(sys.argv[1])


# instantiate an array
err_da = 0
n = 0

ensemble_times = np.arange(t0_ens, t0_ens+10.0, 10.0)
nmems = 10
for t in ensemble_times:
    for i in range(1, nmems+1):
        subdir = join(exproot, "ensembles", str(t), "spec_mag_0.02", f"b{i:02d}")
        da_pert = xr.open_dataset(join(subdir, "atmos_6_hourly.nc"))['ps']
        err = np.abs(da_pert - da_ctrl)
        err = err.assign_coords(time=(cftime.date2num(da_pert.time, "days since 0001-01-01") - t))
        err = err.sel(time=slice(0, 30))
        err_da = err_da + err
        n += 1
        print(f"Error for t = {t}, n = {i} added")

err_da = err_da/n
err_da = err_da.assign_coords(time=(cftime.num2date(err_da.time + t0_ens, "days since 0001-01-01", calendar='360_day')))
err_da.to_netcdf(f'ps_abs_error_time_series_ens{t0_ens}.nc')


da_ctrl = xr.open_dataset(fn1)['vor']
# instantiate an array
err_da = 0
n = 0

ensemble_times = np.arange(t0_ens, t0_ens+10.0, 10.0)
nmems = 10
for t in ensemble_times:
    for i in range(1, nmems+1):
        subdir = join(exproot, "ensembles", str(t), "spec_mag_0.02", f"b{i:02d}")
        da_pert = xr.open_dataset(join(subdir, "atmos_6_hourly.nc"))['vor']
        err = np.abs(da_pert - da_ctrl)
        err = err.assign_coords(time=(cftime.date2num(da_pert.time, "days since 0001-01-01") - t))
        err = err.sel(time=slice(0, 30))
        err_da = err_da + err
        n += 1
        print(f"Error for t = {t}, n = {i} added")

err_da = err_da/n
err_da = err_da.assign_coords(time=(cftime.num2date(err_da.time + t0_ens, "days since 0001-01-01", calendar='360_day')))
err_da.to_netcdf(f'vor_abs_error_time_series_ens{t0_ens}.nc')
        