import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from os.path import join

# An animation of one ensemble
import sys
sys.path.append("..")
from visualizations.animations import animate_tracks_and_synop
from track import load_trackdict_from_file

ds1 = xr.open_dataset("ps_abs_error_time_series_ens375.5.nc")
ds2 = xr.open_dataset("vor_abs_error_time_series_ens375.5.nc")
ds = xr.merge([ds1, ds2])

GFDL_STORAGE = "/orcd/data/talia_tb/001/aqua_gcm_runs"
exproot = join(GFDL_STORAGE, "frierson_default", "control")
td_cyc_fn = join(exproot, "tracks", "vor850", "vor850_cycs.pickle")
td_cyc = load_trackdict_from_file(td_cyc_fn) 
td_acyc_fn = join(exproot, "tracks", "vor850", "vor850_acycs.pickle")
td_acyc = load_trackdict_from_file(td_acyc_fn)

animate_tracks_and_synop([td_cyc, td_acyc], ds, save_prefix="ens375.ps_err") 
animate_tracks_and_synop([td_cyc, td_acyc], ds, save_prefix="ens375.vor_err") 