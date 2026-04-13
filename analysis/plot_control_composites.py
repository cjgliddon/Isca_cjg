import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from os.path import join
import pickle
import sys

obs = ['ps', 'vor850', 'Z500', 'qsurf', 'T850']
feat = ['vc', 'va']

GFDL_STORAGE = "/orcd/data/talia_tb/001/aqua_gcm_runs"
expname = sys.argv[1]
exproot = join(GFDL_STORAGE, expname)

# use lambda functions to streamline loading data
load_comp = lambda o, f: pickle.load(open(join(exproot, f"composites/{o}_control_{f}.pickle"), 'rb'))

ps_vc, nc = load_comp(obs[0],feat[0])
vor_vc, nc = load_comp(obs[1],feat[0])
z_vc, nc = load_comp(obs[2], feat[0])
q_vc, nc = load_comp(obs[3],feat[0])
temp_vc, nc = load_comp(obs[4],feat[0])

ps_va, na = load_comp(obs[0],feat[1])
vor_va, na = load_comp(obs[1],feat[1])
z_va, na = load_comp(obs[2], feat[1])
q_va, na = load_comp(obs[3],feat[1])
temp_va, na = load_comp(obs[4],feat[1])

# Show composites of cyclones and anticyclones
fig, axes = plt.subplots(1, 2, figsize=(10, 6), layout='compressed')
fig.suptitle(f'850-hPa vorticity, temperature, 500-hPa GPH')

# cyclone low-level variables
ax = axes[0]
cm = ax.pcolormesh(vor_vc.lon, vor_vc.lat, vor_vc*1e5)
ct = ax.contour(temp_vc.lon, temp_vc.lat, temp_vc, levels=8, colors='b', linewidths=1)
ct2 = ax.contour(z_vc.lon, z_vc.lat, z_vc, levels=8, colors='k', linewidths=1)
ax.clabel(ct)
ax.clabel(ct2)
fig.colorbar(cm, ax=ax, orientation='horizontal', label='$ζ$ ($10^{-5}$ s$^{-1}$)')
ax.set_xlabel('lon')
ax.set_ylabel('lat')
ax.set_title(f'cyclones (n = {nc})')

# cyclone low-level variables
ax = axes[1]
cm = ax.pcolormesh(vor_va.lon, vor_va.lat, vor_va*1e5)
ct = ax.contour(temp_va.lon, temp_va.lat, temp_va, levels=8, colors='b', linewidths=1)
ct2 = ax.contour(z_va.lon, z_va.lat, z_va, levels=8, colors='k', linewidths=1)
ax.clabel(ct)
ax.clabel(ct2)
fig.colorbar(cm, ax=ax, orientation='horizontal', label='$ζ$ ($10^{-5}$ s$^{-1}$)')
ax.set_xlabel('lon')
ax.set_ylabel('lat')
ax.set_title(f'anticyclones (n = {na})')

plt.savefig(join(exproot, "composites", f"composite_control.png"), format="png")