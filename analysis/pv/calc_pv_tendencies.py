import numpy as np
import os
from os.path import join
import sys
import xarray as xr

sys.path.append('..')
# my modules
from time_derivative import time_derivative

GFDL_STORAGE = os.environ['GFDL_STORAGE'] 

def test():
    expname = sys.argv[1]
    exproot = join(GFDL_STORAGE, expname)

    pv_vars = ["pv250", "pv500", "pv850"]

    do_control   = True
    do_ensembles = False

    # control run time tendencies
    if do_control:
        postproc = join(exproot, 'postprocessed')
        for varname in pv_vars:
            ds = xr.open_dataset(join(postproc, f"{varname}.nc"))
            dds_dt = time_derivative(ds, method='forward')
            dds_dt.to_netcdf(join(postproc, f"d{varname}_dt.nc"))
            print(f"Control {varname} tendency saved")

    # ensemble run time tendencies
    if do_ensembles:
        nmems = 20
        start_times = [360.0]
        data_dirs = [join(exproot, 'ensembles', str(t0), 'spec_mag_0.02', f'b{imem:02d}')
                    for t0 in start_times
                    for imem in range(1, nmems+1)]
        for ddir in data_dirs:
            for varname in pv_vars:
                try:
                    ds = xr.open_dataset(join(ddir, f"{varname}.nc"))
                    dds_dt = time_derivative(ds, method='forward')
                    dds_dt.to_netcdf(join(ddir, f"d{varname}_dt.nc"))
                    ds.close(); dds_dt.close()
                    del ds
                    del dds_dt
                except:
                    print(f"Error in {ddir}, {varname}")
            print(f"All tendencies saved for ensemble directory {ddir}")

def main():
    expname = sys.argv[1]
    exproot = join(GFDL_STORAGE, expname)

    pv_vars = ("pv250", "pv500", "pv850")

    do_control   = True
    do_ensembles = False

    # control run time tendencies
    if do_control:
        postproc = join(exproot, 'postprocessed')
        for varname in pv_vars:
            ds = xr.open_dataset(join(postproc, f"{varname}.nc"))
            dds_dt = time_derivative(ds, method='centered')
            dds_dt.to_netcdf(join(postproc, f"d{varname}_dt.nc"))
            print(f"Control {varname} tendency saved")

    # ensemble run time tendencies
    if do_ensembles:
        nmems = 20
        start_times = np.arange(1440.0, 1445.0, 10.0)
        data_dirs = [join(exproot, 'ensembles', str(t0), 'spec_mag_0.02', f'b{imem:02d}')
                    for t0 in start_times
                    for imem in range(1, nmems+1)]
        for ddir in data_dirs:
            for varname in pv_vars:
                try:
                    ds = xr.open_dataset(join(ddir, f"{varname}.nc"))
                    dds_dt = time_derivative(ds, method='centered')
                    dds_dt.to_netcdf(join(ddir, f"d{varname}_dt.nc"))
                    ds.close(); dds_dt.close()
                    del ds
                    del dds_dt
                except:
                    print(f"Error in {ddir}, {varname}")
            print(f"All tendencies saved for ensemble directory {ddir}")
    

if __name__ == '__main__':
    main()