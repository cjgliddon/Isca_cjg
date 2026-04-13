import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import os
from os.path import join
import sys
import xarray as xr

import pdb

# my modules
sys.path.append('..')
sys.path.append('../..')
from ensemble import EnsembleSet
from pv_decomposition import get_decomp_coefficients_family
from track import load_trackdict_from_file
from visualizations.animations import animate_matched_tracks, animate_matched_tracks_pv

# load the data
GFDL_STORAGE=os.environ["GFDL_STORAGE"]
expname="default"
exproot=join(GFDL_STORAGE, expname)

t0 = 670.0
eset = EnsembleSet(exproot, n_mems=20, t0_list=[t0])
md = eset.match_tracks("vor850_cycs", (t0+9.0, t0+9.0), ensemble_start=t0)

# create a new dictionary so we can reference the keys of the match-dict
tracks = md.keys()
track_index = {ind: tr for ind, tr in enumerate(md.keys())}

# do the error calculation
var = "pv850"
ctrl_ds       = join(exproot, "postprocessed", f"{var}.nc")
ctrl_ds_climo = join(exproot, "postprocessed", f"{var}_mean.nc")
ctrl_ds_tdv   = join(exproot, "postprocessed", f"d{var}_dt.nc")
pds_fn        = f"{var}.nc"
pds_tdv_fn    = f"d{var}_dt.nc"
ensemble_path = eset.ensemble_dict[t0].path_to

results_dict = dict()
for ind, tr in track_index.items():
    results_dict[ind] = get_decomp_coefficients_family(
        ctrl_ds, ctrl_ds_climo, ctrl_ds_tdv, ensemble_path,
        pds_fn, pds_tdv_fn, tr, md[tr], t0, "cyc"
    )

print(results_dict)

anim_bkgd = xr.open_dataset(join(exproot, "postprocessed", "base_anoms.nc"))
anim_bkgd = xr.open_dataset(join(exproot, 'postprocessed', 'pv850.nc'))
# pdb.set_trace()
animate_matched_tracks_pv(md, -anim_bkgd, "sample_track_matches", pv_scale=1)

# fig, axes = plt.subplots(2, 2, figsize=(8,8))
# ctrl_res, pert_res = results_dict[0]
# params = ['beta', 'ax', 'ay', 'gamma']
# for i, p in enumerate(params):
#     ax = axes[np.unravel_index(i, (2, 2))]
#     ax.plot(ctrl_res['time'], ctrl_res[p], color='k', linewidth=2, label='control')
#     for j_mem, data in pert_res.items():
#         mask = np.where(data[p] != 0.0)
#         ax.plot(data['time'][mask], data[p][mask], color='0.5', alpha=0.7, linewidth=1)
#     ax.set_title(p)
# 
# plt.savefig('pv_decomp_stats_0.png', format='png')
# plt.close()
# 
# fig, axes = plt.subplots(2, 2, figsize=(8,8))
# ctrl_res, pert_res = results_dict[3]
# params = ['beta', 'ax', 'ay', 'gamma']
# for i, p in enumerate(params):
#     ax = axes[np.unravel_index(i, (2, 2))]
#     ax.plot(ctrl_res['time'], ctrl_res[p], color='k', linewidth=2, label='control')
#     for j_mem, data in pert_res.items():
#         mask = np.where(data[p] != 0.0)
#         ax.plot(data['time'][mask], data[p][mask], color='0.5', alpha=0.7, linewidth=1)
#     ax.set_title(p)
# 
# plt.savefig('pv_decomp_stats_3.png', format='png')