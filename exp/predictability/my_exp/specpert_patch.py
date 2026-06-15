import numpy as np
import os
from os.path import join
import subprocess
import sys
from shared_vars import expname, GFDL_STORAGE

if __name__ == "__main__":
    print("Got into Main")
    t0, tf, dt = (float(arg) for arg in sys.argv[1:4])
    mem0, memf=(int(arg) for arg in sys.argv[4:6])
    print(f"t0, tf, dt for patching: {t0, tf, dt}")
    print("")
    n_mems = int(sys.argv[4])
    print(f"Number of ensemble members: {n_mems}")

    patch_list = []     # list of ensemble members to be "patched"
    datadir="ensembles"
    for time in np.arange(t0, tf+dt, dt):
        for i_mem in range(mem0, memf+1):
            ensemble_folder = join(GFDL_STORAGE, expname, datadir, str(time), f"mem{i_mem:02d}")
            out_nc = join(ensemble_folder, "atmos_6_hourly.nc")
            if not os.path.isfile(out_nc):
                patch_list.append((time, i_mem))
    
    for time, i_mem in patch_list:
        print(f"Patching ensemble member {(time, i_mem)}...")
        subprocess.run(['python', 'specpert.py', str(time), str(i_mem), "1", "0"])
        print("Patching successful.")
    print("All needed patching complete.")