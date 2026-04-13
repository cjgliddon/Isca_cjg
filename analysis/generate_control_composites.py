import os
from os.path import join
import pickle
import sys
import xarray as xr

from track import *
import observable_functions as of

# This is a script which generates

GFDL_STORAGE = "/orcd/data/talia_tb/001/aqua_gcm_runs"
expname = sys.argv[1]
t_ini, t_fin = float(sys.argv[2]), float(sys.argv[3])
if len(sys.argv) > 4:
    dt = float(sys.argv[4])
else:
    dt = 0.25
exproot = join(GFDL_STORAGE, expname)

ds = xr.open_dataset(join(exproot, "postprocessed", "base_anoms.nc"))
ds2 = xr.open_dataset(join(exproot, "postprocessed", "base_thermo_ll.nc"))
ds3 = xr.open_dataset(join(exproot, "postprocessed", "base_z500.nc"))

track_types = ["vor850/vor850_cycs.pickle", "vor850/vor850_acycs.pickle"]
abbrs = ["vc", "va"]
obsf = [of.surface_pressure, of.vorticity_850, of.temperature_850, of.q_surf, of.height_500]
obsf_str = ['ps', 'vor850', 'T850', 'qsurf', 'Z500']
ds_list = [ds, ds, ds2, ds2, ds3]
invert_bool = [False, True, False, False, False]
anom_bool = [False, False, False, False, True]

assert len(track_types) == len(abbrs)
assert len(obsf) == len(obsf_str)

for i in range(len(track_types)):
    td = load_trackdict_from_file(join(exproot, "tracks", track_types[i]))
    for j in range(len(obsf)):
        composite = get_peak_composites(td, ds_list[j], obsf[j], np.arange(t_ini, t_fin+dt, dt), use_anom_fields=anom_bool[j], invert_SH=invert_bool[j])
        if not os.path.isdir(join(exproot, 'composites')):
            os.mkdir(join(exproot, 'composites'))
        out_file = join(exproot, 'composites', f"{obsf_str[j]}_control_{abbrs[i]}.pickle")
        pickle.dump(composite, open(out_file, "wb"))
