# freely adapted/condensed from Justin Finkel's "observable_functions.py" from the TEAMS repo
# TODO: trim

import numpy as np
import xarray as xr
from metpy import calc
from metpy.units import units

import pdb

Omega = 7.2921150e-5

def get_pressure_coord_name(ds):
    return list(ds.coords)[-1]

def surface_pressure(ds):
    return ds["ps"]

def temperature_850(ds):
    return ds["temp"].sel(pfull=850.0, method="nearest")

def q_surf(ds):
    pcoord = get_pressure_coord_name(ds)
    return ds["sphum"].isel({pcoord: -1})

def vorticity_850(ds):
    return ds["vor"].sel(pfull=850.0, method="nearest")

def height_500(ds):
    return ds["height"].sel(pfull=500.0, method="nearest")

def pressure(ds):
    # Return pressure with "pfull" as a vertical coordinate. 
    p_edge = ds["bk"]*ds["ps"] # Pascals
    p_cent = 0.5*(p_edge + p_edge.shift(phalf=-1)).isel(phalf=slice(None,-1)).rename({"phalf": "pfull"}).assign_coords({"pfull": ds["pfull"]})
    dp_dpfull = (p_edge.shift(phalf=-1) - p_edge)/(ds["phalf"].shift(phalf=-1) - ds["phalf"])
    dp_dpfull = dp_dpfull.isel(phalf=slice(None,-1)).rename({"phalf": "pfull"}).assign_coords({"pfull": ds["pfull"]})
    return p_cent, dp_dpfull

def gph(ds):
    # return geopotential with "phalf" as a vertical coordinate, calculated from temp.
    # TODO: correct for effects of moisture? Also clean up to be more idiomatic probably
    R = 287.04		# gas constant for dry air
    g = 9.80665		# gravitational constant
    p_edge = ds["bk"]*ds["ps"]	# pressures at the edges of the sigma levels, in Pa
    delta_p = (p_edge.shift(phalf=-1) - p_edge)		# thickness of layer beneath each phalf level. Note that the bottom layer has
                                # 'nan' as the value
    temp = ds['temp']							# temperatures in each level
    pres = pressure(ds)[0]							# pressures in each level
    dp_pc = delta_p.isel(phalf=slice(None,-1)).rename({"phalf": "pfull"}).assign_coords({"pfull": ds["pfull"]})
    delta_z = (R*temp)/(g*pres)*dp_pc
    geopots = []
    geopot_shape = None
    p_num = len(delta_z['pfull'])
    for i in range(0,p_num):	
        geopot = delta_z.isel(pfull=slice(i,p_num)).sum(dim='pfull').values	# calculates the gph at each level boundary...
        geopots += [geopot]					# and adds it to the list
        geopot_shape = geopot.shape
    # finally, need to add the gph for the bottom layer...
    geopots += [np.zeros(geopot_shape)]
    # and let's organize as a DataArray
    geopots = np.asarray(geopots) 
    gph_array = xr.DataArray(
        data = geopots,
        dims = ["phalf","time","lat","lon"],
        coords = dict(
            phalf=p_edge["phalf"].values,
            time=p_edge["time"].values,
            lat=p_edge["lat"].values,
            lon=p_edge["lon"].values,
            ),
        name='height',
        attrs=dict(
            description="Geopotential height as function of pressure",
            units="m",
            )
        )
    return gph_array

def potential_temperature(ds):
    """ Given a dataset ds which includes temperature data as a function of pressure,
        calculates and returns potential temperature as a DataArray."""
    temp = ds['temp'].data * units.K
    # pres has shape (num_p,) — broadcast over time, lat, lon automatically
    pres = ds['pfull'].data * units.hPa

    theta = calc.potential_temperature(pres[:, None, None], temp).magnitude

    theta_da = xr.DataArray(
        data=theta,
        dims=["time", "pfull", "lat", "lon"],
        coords=dict(
            time=("time", ds.time.data),
            pfull=("pfull", ds.pfull.data),
            lat=("lat", ds.lat.data),
            lon=("lon", ds.lon.data),
        ),
        attrs=dict(
            description="potential temperature as a function of pressure",
            units="K",
        ),
    )
    return theta_da

def PV_isobaric(ds, omega=Omega, omega_frac=1.0, x_dim=-1, y_dim=-2, vertical_dim=-3,
                dx=None, dy=None, parallel_scale=None, meridional_scale=None):
    
    # Keep only needed variables — drop anything else upfront
    ds = ds[['ucomp', 'vcomp', 'vor', 'temp', 'pfull']].copy()
    
    theta = potential_temperature(ds) * units.K
    pres  = ds['pfull'] * units.hPa
    u     = ds['ucomp'] * units.meter / units.second
    v     = ds['vcomp'] * units.meter / units.second
    vor   = ds['vor']   / units.second
    
    f    = 2 * omega * omega_frac * np.sin(np.deg2rad(ds.lat)) / units.second
    avor = vor + f;  del vor, f          # free intermediates as soon as possible

    dthetadp = calc.first_derivative(theta, x=pres, axis=vertical_dim)
    
    if (np.shape(theta)[y_dim] == 1) and (np.shape(theta)[x_dim] == 1):
        dthetady = units.Quantity(0, 'K/m')
        dthetadx = units.Quantity(0, 'K/m')
    else:
        dthetadx, dthetady = calc.geospatial_gradient(
            theta, dx=dx, dy=dy,
            x_dim=x_dim, y_dim=y_dim,
            parallel_scale=parallel_scale,
            meridional_scale=meridional_scale,
        )
    del theta                            # no longer needed

    dudp = calc.first_derivative(u, x=pres, axis=vertical_dim);  del u
    dvdp = calc.first_derivative(v, x=pres, axis=vertical_dim);  del v

    g = 9.80665 * units.meter / units.second**2
    pv_da = - g * (dudp * dthetady - dvdp * dthetadx + avor * dthetadp)
    del dudp, dvdp, dthetadx, dthetady, avor, dthetadp

    pv_da = pv_da.rename("pv")
    pv_da.data = pv_da.data.to('K * m**2 / (s * kg)').magnitude
    pv_da.attrs.update(dict(
        description="isobaric baroclinic potential vorticity",
        units="K*m^2/(kg*s)",
    ))
    return pv_da

def pv(ds, flip_sign=False):
    """ Returns the potential vorticity at all model levels present. """
    return ds['pv'] if flip_sign == False else -ds['pv']

def pv_250(ds):
    """ Returns the potential vorticity at 250 hPa. """
    return ds['pv'].sel(pfull=250.0, method='nearest')

def pv_500(ds):
    """ Returns the potential vorticity at 500 hPa. """
    return ds['pv'].sel(pfull=500.0, method='nearest')

def pv_850(ds):
    """ Returns the potential vorticity at 850 hPa. """
    return ds['pv'].sel(pfull=850.0, method='nearest')

def omega_from_div(ds, div_var="div", plev_dim="pfull") -> xr.DataArray:
    """
    Compute the vertical velocity omega (Pa/s) from horizontal wind divergence
    using the continuity equation:

        div + ∂(omega)/∂p = 0  =>  omega(p) = -∫[p_top -> p] div dp'

    Integration proceeds downward from the top of the atmosphere (where omega = 0)
    to each pressure level using the trapezoidal rule.

    Parameters
    ----------
    div_dataset : xr.Dataset
        Dataset containing the horizontal divergence field (s⁻¹) on pressure levels.
    div_var : str, optional
        Name of the divergence variable within the dataset. Default: "div".
    plev_dim : str, optional
        Name of the pressure-level dimension (Pa). Default: "pfull".

    Returns
    -------
    xr.DataArray
        DataArray of omega values (Pa/s) on the same grid as the input divergence,
        with the same coordinates and dimensions.
    """
    div = ds[div_var]

    # Pressure coordinate (Pa). Sort ascending so we integrate top -> surface.
    p = div[plev_dim]
    if p.values[0] > p.values[-1]:
        # Levels are stored surface -> top; flip so index 0 = top of atmosphere
        div = div.isel({plev_dim: slice(None, None, -1)})
        p   = div[plev_dim]

    p_vals = p.values          # shape (n_lev,)
    n_lev  = len(p_vals)

    # Allocate output with the same shape/coords as div (in sorted order)
    omega = xr.zeros_like(div)

    # Trapezoidal integration from p_top downward.
    # omega(p_k) = -∫[p_top -> p_k] div dp
    #            ≈ -Σ_{j=0}^{k-1}  0.5*(div_j + div_{j+1}) * (p_{j+1} - p_j)
    #
    # omega at the topmost level is 0 (boundary condition).
    for k in range(1, n_lev):
        dp = p_vals[k] - p_vals[k - 1]          # positive (pressure increases downward)
        div_mean = 0.5 * (
            div.isel({plev_dim: k - 1}) + div.isel({plev_dim: k})
        )
        omega[{plev_dim: k}] = omega[{plev_dim: k - 1}] - div_mean * dp

    # Restore the original level ordering if it was flipped
    omega = omega.sortby(plev_dim, ascending=(ds[div_var][plev_dim].values[0]
                                               < ds[div_var][plev_dim].values[-1]))

    omega.attrs.update({
        "long_name": "Lagrangian pressure tendency (omega)",
        "units":     "Pa s-1",
        "derived_from": f"continuity equation integration of {div_var}",
    })
    omega.name = "omega"
    return omega


# ----------------------------- CLAUDE -------------------------------

def compute_wind_from_vordiv(vor, div, vor_var='vor', div_var='div', 
                              lat_dim='lat', lon_dim='lon'):
    """
    Compute zonal (u) and meridional (v) wind components from vorticity and divergence.
    
    This uses spectral methods to solve the Helmholtz decomposition:
    - ζ = ∂v/∂x - ∂u/∂y (vorticity)
    - δ = ∂u/∂x + ∂v/∂y (divergence)
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing vorticity and divergence fields
    vor_var : str, default='vor'
        Name of vorticity variable in dataset
    div_var : str, default='div'
        Name of divergence variable in dataset
    lat_dim : str, default='lat'
        Name of latitude dimension
    lon_dim : str, default='lon'
        Name of longitude dimension
    
    Returns
    -------
    xarray.Dataset
        Dataset containing 'u' and 'v' wind components
    """
    
    # Get coordinate arrays
    lat = vor[lat_dim].values
    lon = vor[lon_dim].values
    
    # Earth radius
    a = 6.371e6  # meters
    
    # Convert to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    # Grid spacing
    dlat = np.deg2rad(np.abs(lat[1] - lat[0]))
    dlon = np.deg2rad(np.abs(lon[1] - lon[0]))
    
    # Create meshgrid for latitude (needed for scale factors)
    lat_2d, _ = np.meshgrid(lat_rad, lon_rad, indexing='ij')
    
    # Wavenumbers for FFT
    nlon = len(lon)
    nlat = len(lat)
    k = np.fft.fftfreq(nlon, dlon / (2 * np.pi))
    
    # Initialize output arrays with same shape as input
    u = xr.zeros_like(vor)
    v = xr.zeros_like(vor)
    
    # Get all dimension names except lat/lon for iteration
    other_dims = [d for d in vor.dims if d not in [lat_dim, lon_dim]]
    
    # Process each level/time separately
    if other_dims:
        # Stack all non-spatial dimensions
        vor_stack = vor.stack(z=other_dims)
        div_stack = div.stack(z=other_dims)
        u_stack = u.stack(z=other_dims)
        v_stack = v.stack(z=other_dims)
        
        for i in range(vor_stack.sizes['z']):
            vor_2d = vor_stack.isel(z=i).values
            div_2d = div_stack.isel(z=i).values
            
            u_2d, v_2d = _solve_vordiv_2d(vor_2d, div_2d, lat_2d, a, k, dlat)
            
            u_stack[{lat_dim: slice(None), lon_dim: slice(None), 'z': i}] = u_2d
            v_stack[{lat_dim: slice(None), lon_dim: slice(None), 'z': i}] = v_2d
        
        u = u_stack.unstack('z')
        v = v_stack.unstack('z')
    else:
        u_2d, v_2d = _solve_vordiv_2d(vor.values, div.values, lat_2d, a, k, dlat)
        u.values = u_2d
        v.values = v_2d
    
    # Create output dataset
    ds_out = xr.Dataset({
        'u': u.assign_attrs(long_name='Zonal wind component', units='m/s'),
        'v': v.assign_attrs(long_name='Meridional wind component', units='m/s')
    })
    
    return ds_out


def _solve_vordiv_2d(vor, div, lat_2d, a, k, dlat):
    """Solve for u,v from 2D vorticity and divergence fields."""
    
    nlat, nlon = vor.shape
    
    # Apply FFT in longitude direction
    vor_fft = np.fft.fft(vor, axis=1)
    div_fft = np.fft.fft(div, axis=1)
    
    # Initialize Fourier coefficients
    u_fft = np.zeros_like(vor_fft, dtype=complex)
    v_fft = np.zeros_like(vor_fft, dtype=complex)
    
    # Solve for each zonal wavenumber
    for ik, kval in enumerate(k):
        if kval == 0:
            # For k=0, solve using finite differences in latitude
            # This handles the rotational and divergent parts separately
            chi_k = _solve_poisson_meridional(div_fft[:, ik], lat_2d[:, 0], a, dlat)
            psi_k = _solve_poisson_meridional(vor_fft[:, ik], lat_2d[:, 0], a, dlat)
            
            # Velocity components from streamfunction and velocity potential
            v_fft[:, ik] = -_meridional_derivative(psi_k, a, dlat)
            u_fft[:, ik] = _meridional_derivative(chi_k, a, dlat)
        else:
            # For k≠0, use spectral solution
            cos_lat = np.cos(lat_2d[:, 0])
            ikval = 1j * kval / (a * cos_lat)
            
            # Solve for streamfunction (ψ) and velocity potential (χ)
            laplacian_k = -(kval**2) / (a**2 * cos_lat**2)
            
            # Use centered differences for second derivative in latitude
            laplacian_lat = _laplacian_lat_component(np.ones(nlat), lat_2d[:, 0], a, dlat)
            total_laplacian = laplacian_k + laplacian_lat
            
            # Avoid division by zero
            total_laplacian[np.abs(total_laplacian) < 1e-20] = 1e-20
            
            psi_k = vor_fft[:, ik] / total_laplacian
            chi_k = div_fft[:, ik] / total_laplacian
            
            # Compute velocities
            u_rot = -_meridional_derivative(psi_k, a, dlat)
            v_rot = ikval * psi_k
            
            u_div = ikval * chi_k
            v_div = _meridional_derivative(chi_k, a, dlat)
            
            u_fft[:, ik] = u_rot + u_div
            v_fft[:, ik] = v_rot + v_div
    
    # Inverse FFT
    u = np.real(np.fft.ifft(u_fft, axis=1))
    v = np.real(np.fft.ifft(v_fft, axis=1))
    
    return u, v


def _meridional_derivative(field, a, dlat):
    """Compute meridional derivative using centered differences."""
    deriv = np.zeros_like(field)
    deriv[1:-1] = (field[2:] - field[:-2]) / (2 * a * dlat)
    deriv[0] = (field[1] - field[0]) / (a * dlat)
    deriv[-1] = (field[-1] - field[-2]) / (a * dlat)
    return deriv


def _laplacian_lat_component(field, lat, a, dlat):
    """Compute the latitudinal component of the Laplacian operator."""
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    
    # Second derivative approximation
    d2f = np.zeros_like(field)
    d2f[1:-1] = (field[2:] - 2*field[1:-1] + field[:-2]) / dlat**2
    
    # First derivative for metric term
    df = np.zeros_like(field)
    df[1:-1] = (field[2:] - field[:-2]) / (2 * dlat)
    
    result = (d2f - np.tan(lat) * df) / a**2
    return result


def _solve_poisson_meridional(source, lat, a, dlat):
    """Simple solver for Poisson equation in meridional direction."""
    # This is a simplified approach - for production use, consider more sophisticated methods
    nlat = len(lat)
    result = np.zeros(nlat, dtype=source.dtype)
    
    # Integrate twice (simplified approach)
    # This works for the k=0 mode
    integral1 = np.cumsum(source) * dlat * a
    result = np.cumsum(integral1) * dlat * a
    
    # Remove mean to ensure unique solution
    result -= np.mean(result)
    
    return result

if __name__ == "__main__":

    from os.path import join
    import matplotlib.pyplot as plt

    GFDL_STORAGE = "/orcd/data/talia_tb/001/aqua_gcm_runs"
    ds_path = join(GFDL_STORAGE, "default", "run0020", "atmos_6_hourly.nc")
    ds_sample = xr.open_dataset(ds_path).isel(time=slice(0, 4))
    print("Sample dataset loaded")

    theta_sample = potential_temperature(ds_sample)
    print(theta_sample)

    pv_sample = PV_isobaric(ds_sample)
    print(pv_sample)
    fig, ax = plt.subplots()
    cm = ax.pcolormesh(pv_sample.lon, pv_sample.lat, pv_sample.isel(time=-1).sel(pfull=250.0, method='nearest'))
    fig.colorbar(cm)
    plt.savefig('pv_test.png', format='png')
    plt.close()

    ds_path2 = join(GFDL_STORAGE, "default", "postprocessed", "pv250.nc")
    ds_sample = xr.open_dataset(ds_path2).isel(time=slice(0, 4))
    pv = pv(ds_sample, flip_sign=True)
    fig, ax = plt.subplots()
    cm = ax.pcolormesh(pv.lon, pv.lat, pv.isel(time=-1))
    fig.colorbar(cm)
    plt.savefig('pv_test2.png', format='png')
    plt.close()