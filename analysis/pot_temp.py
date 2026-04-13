import numpy as np
import xarray as xr
from metpy import units
from metpy.calc import calc

def potential_temperature(ds):
    """ Given a dataset ds which includes temperature data as a function of pressure, 
        calculates and returns potential temperature as a DataArray."""
    temp = ds['temp'].data
    pres = ds['pfull'].data
    theta = np.zeros(temp.shape)
    nt, num_p, ny, nx = theta.shape
    for i_r in range(nt*num_p*ny*nx):
        i,j,k,l = np.unravel_index(i_r, (nt, num_p, ny, nx))
        tloc = temp[i,j,k,l] * units.K
        ploc = pres[j] * units.hPa
        theta[i,j,k,l] = calc.potential_temperature(ploc, tloc)._magnitude
    theta_da = xr.DataArray(data=theta,
                            dims=["time", "pfull", "lat", "lon"],
                            coords=dict(
                                time=("time", ds.time.data),
                                pfull=("pfull", ds.pfull.data),
                                lat=("lat", ds.lat.data),
                                lon=("lon", ds.lon.data),
                                ),
                            attrs=dict(
                                 description="potential temperature as a function of pressure",
                                 units="K"
                                 ),
                            )
    return theta