import numpy as np
import xarray as xr
from os.path import join
import sys
import pdb

sys.path.append("..")
import observable_functions as of

GFDL_STORAGE="/orcd/data/talia_tb/001/aqua_gcm_runs"
exproot = join(GFDL_STORAGE, "frierson_default")

print("Converting control data")
fn1 = join(exproot, "control", "run0013", "atmos_6_hourly.nc")
ds = xr.open_dataset(fn1)
theta = of.potential_temperature(ds)
theta.to_netcdf("theta.ds")
pv = of.PV_bc(ds)
pv.to_netcdf("pv.ds")
