import numpy as np
import xarray as xr
from os.path import join
import sys
import pdb
import cftime

sys.path.append("..")
import observable_functions as of

GFDL_STORAGE="/orcd/data/talia_tb/001/aqua_gcm_runs"
expname = sys.argv[1]
exproot = join(GFDL_STORAGE, expname)
# exproot = join(GFDL_STORAGE, "frierson_default")

fn1 = join(exproot, "postprocessed", "base_ps_vor.nc")
fn2 = join(exproot, "postprocessed", "base_thermo_ll.nc")
# da_ctrl = xr.open_dataset(fn1)['ps']
# 
# 
# # instantiate an array
# err_da = 0
# n = 0
# 
# ensemble_times = np.arange(365.5, 720.0, 10.0)
# nmems = 10
# for t in ensemble_times:
#     for i in range(1, nmems+1):
#         subdir = join(exproot, "ensembles", str(t), "spec_mag_0.02", f"b{i:02d}")
#         try:
#             da_pert = xr.open_dataset(join(subdir, "atmos_6_hourly.nc"))['ps']
#             err = np.abs(da_pert - da_ctrl)
#             err = err.assign_coords(time=(cftime.date2num(da_pert.time, "days since 0001-01-01") - t))
#             err = err.sel(time=slice(0, 30))
#             err_da = err_da + err
#             n += 1
#             print(f"Error for t = {t}, n = {i} added")
#         except:
#             print(f"Error in adding t = {t}, n = {i}")
# 
# err_da = err_da/n
# err_da.to_netcdf(join(exproot, "other_data", 'ps_abs_error_time_series.nc'), mode="w")
# 
# 
# da_ctrl = xr.open_dataset(fn1)['vor']
# 
# 
# # instantiate an array
# err_da = 0
# n = 0
# 
# ensemble_times = np.arange(365.5, 1440.0, 10.0)
# nmems = 10
# for t in ensemble_times:
#     for i in range(1, nmems+1):
#         subdir = join(exproot, "ensembles", str(t), "spec_mag_0.02", f"b{i:02d}")
#         try:
#             da_pert = xr.open_dataset(join(subdir, "atmos_6_hourly.nc"))['vor']
#             err = np.abs(da_pert - da_ctrl)
#             err = err.assign_coords(time=(cftime.date2num(da_pert.time, "days since 0001-01-01") - t))
#             err = err.sel(time=slice(0, 30))
#             err_da = err_da + err
#             n += 1
#             print(f"Error for t = {t}, n = {i} added")
#         except:
#             print(f"Error in adding t = {t}, n = {i}")
# 
# 
# err_da = err_da/n
# err_da.to_netcdf(join(exproot, "other_data", 'vor_abs_error_time_series.nc'), mode="w")
        

da_ctrl = xr.open_dataset(fn2)['temp']
err_da = 0
n = 0

ensemble_times = np.arange(365.5, 1440.0, 10.0)
nmems = 10
for t in ensemble_times:
    for i in range(1, nmems+1):
        subdir = join(exproot, "ensembles", str(t), "spec_mag_0.02", f"b{i:02d}")
        try:
            da_pert = xr.open_dataset(join(subdir, "atmos_6_hourly.nc"))['temp']
            err = np.abs(da_pert - da_ctrl)
            err = err.assign_coords(time=(cftime.date2num(da_pert.time, "days since 0001-01-01") - t))
            err = err.sel(time=slice(0, 30))
            err_da = err_da + err
            n += 1
            print(f"Error for t = {t}, n = {i} added")
        except:
            print(f"Error in adding t = {t}, n = {i}")

err_da = err_da/n
err_da.to_netcdf(join(exproot, "other_data", 'T850_abs_error_time_series.nc'), mode="w")
        