import cftime
import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import sys
import pickle
sys.path.append("..")
from tools import get_lat_indices
import observable_functions as of

GFDL_STORAGE='/orcd/data/talia_tb/001/aqua_gcm_runs'
expname='frierson_moist/es0_0.75'

mean_error_ds_fn = os.path.join(GFDL_STORAGE, expname, "other_data", "ps_abs_error_time_series.nc")
ctrl_ds_fn = os.path.join(GFDL_STORAGE, expname, "postprocessed", "base_ps_vor.nc")
obs_func = of.surface_pressure

# add ensemble filenames to list
times = np.arange(365.5, 1440.0, 10.0)
nmems = 10
ensemble_ds_fn_list = []
for t in times:
    for i in range(1, nmems+1):
        ensemble_ds_fn_list.append((
            t,
            os.path.join(GFDL_STORAGE, expname, 'ensembles', str(t), 'spec_mag_0.02', f"b{i:02d}", "atmos_6_hourly.nc")
        ))
# load control and mean datasets
mean_err_ds = xr.open_dataset(mean_error_ds_fn)

# calculate mean error time series
indices_to_avg = get_lat_indices("ml")
mean_err_ts = mean_err_ds.isel(lat=indices_to_avg).mean(dim=('lat', 'lon'))
pickle.dump(mean_err_ts, open("mean_err_ts.pickle", "wb"))

# extract lats from control ds
ctrl_ds = obs_func(xr.open_dataset(ctrl_ds_fn)).isel(lat=indices_to_avg)

# calculate 10 random ensemble member error time series
len_ensemble_list = len(ensemble_ds_fn_list)
entries = np.random.choice(len_ensemble_list, size=10)
ensemble_ds_subset = [(ensemble_ds_fn_list[i][0], xr.open_dataset(ensemble_ds_fn_list[i][1])) for i in entries]

print("All data loaded")

# get error curves
err_ts = []
for t0_ens, eds in ensemble_ds_subset:
    t0_ens_date = cftime.num2date(t0_ens, "days since 0001-01-01", calendar="360_day")
    eds_times = eds['time'].data
    eds = obs_func(eds).sel(time=slice(t0_ens_date, eds_times[-1])).isel(lat=indices_to_avg)

    # calculate weighted error
    err = np.abs(eds - ctrl_ds.sel(time=slice(t0_ens_date, eds_times[-1])))
    weights = np.cos(np.deg2rad(err.lat))
    err_weighted = err.weighted(weights=weights)
    err_mean = err_weighted.mean(dim=('lat', 'lon'))
    err_ts.append(err_mean)
    print("Error added to list")

pickle.dump(err_ts, open("err_ts_list.pickle", "wb"))

# generate figure ----------------->

fig = plt.figure(); ax = plt.axes()
times_from_ts = lambda da: np.arange(0, 0.25*len(da.time), 0.25)
ax.plot(times_from_ts(mean_err_ts), obs_func(mean_err_ts), color='k', linewidth=1.5, label='MAE')
for ts in err_ts:
    ax.plot(times_from_ts(ts), ts, color='b', alpha=0.5, linewidth=1)
plt.show()
