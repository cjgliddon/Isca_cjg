from concurrent.futures import ProcessPoolExecutor, as_completed

import glob
import numpy as np
import os
import subprocess
import sys
import xarray as xr

# my modules
sys.path.append('..')
from observable_functions import PV_isobaric

import pdb

# ------ helper functions for parallelization ------

def _calculate_pv_emem_single(direc, omega_frac):
    ds = xr.open_dataset(os.path.join(direc, 'atmos_6_hourly.nc'))
    
    pv = PV_isobaric(ds, omega_frac=omega_frac)
    ds.close()          # explicitly release the raw dataset
    del ds

    for level, label in [(850.0, 'pv850'), (500.0, 'pv500'), (250.0, 'pv250')]:
        pv_lev = pv.sel(pfull=level, method='nearest')
        pv_lev.to_netcdf(os.path.join(direc, f"{label}.nc"))
        del pv_lev      # free each level after writing

    del pv
    return

# ----------------- BEGIN SCRIPT -------------------

GFDL_STORAGE = os.environ['GFDL_STORAGE']
expname = sys.argv[1]
omega_frac = float(sys.argv[2])
do_control  = bool(int(sys.argv[3]))
do_ensemble = bool(int(sys.argv[4]))

exproot = os.path.join(GFDL_STORAGE, expname)

if __name__ == "__main__":
    print("Got into Main")

    n_workers = min(4, os.cpu_count())   # tune this to your node's RAM
    # n_workers = 2

    # calculate PV fields for control run
    if do_control: 
        print("Generating control PV files...")
        rundirs     = [os.path.join(exproot, f"run{ich:04d}") for ich in range(13, 51)]
        postprocdir = os.path.join(exproot, "postprocessed")
        for i, rd in enumerate(rundirs):
            ds = xr.open_dataset(os.path.join(rd, "atmos_6_hourly.nc"))
            pv = PV_isobaric(ds, omega_frac=omega_frac)
            pv850 = pv.sel(pfull=850.0, method='nearest')
            pv500 = pv.sel(pfull=500.0, method='nearest')
            pv250 = pv.sel(pfull=250.0, method='nearest')

            pv850.to_netcdf(os.path.join(postprocdir, f"pv_850_{i}.nc"))
            pv500.to_netcdf(os.path.join(postprocdir, f"pv_500_{i}.nc"))
            pv250.to_netcdf(os.path.join(postprocdir, f"pv_250_{i}.nc"))

            print(f"Saved PV fields for directory {rd}")

        subprocess.run(["/home/cgliddon/Isca/analysis/pv/merge_control_pvs.sh", exproot])

    if do_ensemble:
        ensemble_starts = np.arange(360.0, 720.0, 10.0)
        n_mems = 10
        rundirs = [os.path.join(exproot, 'ensembles', str(t0), 'spec_mag_0.02', f"b{n:02d}") 
                    for t0 in ensemble_starts 
                    for n in range(1, n_mems+1)]
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_calculate_pv_emem_single, rd, omega_frac): rd for rd in rundirs
            }
            for future in as_completed(futures):
                rd = futures[future]
                try:
                    res = future.result()
                    print(f"Saved PV fields for directory {rd}")
                except Exception as exc:
                    print(f"Failed save for directory {rd}: {exc}")
#        for rd in rundirs:
#            ds = xr.open_dataset(os.path.join(rd, 'atmos_6_hourly.nc'))
#            pv = PV_isobaric(ds, omega_frac=omega_frac)
#
#            pv850 = pv.sel(pfull=850.0, method='nearest')
#            pv500 = pv.sel(pfull=500.0, method='nearest')
#            pv250 = pv.sel(pfull=250.0, method='nearest')
#
#            pv850.to_netcdf(os.path.join(rd, f"pv850.nc"), mode='w')
#            pv500.to_netcdf(os.path.join(rd, f"pv500.nc"), mode='w')
#            pv250.to_netcdf(os.path.join(rd, f"pv250.nc"), mode='w')
#            print(f"Saved PV fields for directory {rd}")
    
    print(f"Completed PV field calculation for experiment {expname}")
    # ds = xr.open_dataset(os.path.join(rundirs[0], 'pv250.nc'))