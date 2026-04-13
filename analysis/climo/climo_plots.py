import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_surface_means(ds, figsize=(14, 5)):
    """
    Plot time-mean surface temperature and specific humidity.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing 'temp' and 'sphum' variables with coordinates
        (time, lat, lon, pfull)
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, layout='constrained')
    
    # Time-mean surface temperature (lowest pressure level)
    temp_surface = ds['temp'].isel(pfull=-1).mean(dim='time')
    
    # Time-mean surface specific humidity
    sphum_surface = ds['sphum'].isel(pfull=-1).mean(dim='time')
    
    # Plot temperature
    ax = axes[0]
    im1 = ax.contourf(ds['lon'], ds['lat'], temp_surface, levels=15, cmap='RdYlBu_r')
    ax.set_title('surface temperature')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.grid(True, color='gray', alpha=0.5, linestyle='--')
    plt.colorbar(im1, ax=ax, orientation='horizontal', label='K')
    
    # Plot specific humidity
    ax = axes[1]
    im2 = ax.contourf(ds['lon'], ds['lat'], sphum_surface * 1000,  # Convert to g/kg
                      levels=15, cmap='YlGnBu')
    ax.set_title('surface specific humidity')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.grid(True, color='gray', alpha=0.5, linestyle='--')
    plt.colorbar(im2, ax=ax, orientation='horizontal', pad=0.05, label='g/kg')
    fig.suptitle('Time mean surface quantities')

    return fig, axes


def plot_zonal_mean_cross_sections(ds, figsize=(10, 10)):
    """
    Plot lat-vs-pressure cross-sections of time- and zonal-mean winds.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing 'ucomp' and 'vcomp' variables
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, layout='constrained')
    
    # Time and zonal mean fields
    ucomp_mean = ds['ucomp'].mean(dim=['time', 'lon'])
    vcomp_mean = ds['vcomp'].mean(dim=['time', 'lon'])
    
    # Plot zonal wind
    ax = axes[0]
    levels_u = np.arange(-40, 41, 4)
    im1 = ax.contourf(ds['lat'], ds['pfull'], ucomp_mean, 
                      levels=levels_u, cmap='RdBu_r', extend='both')
    ax.contour(ds['lat'], ds['pfull'], ucomp_mean, 
               levels=[0], colors='black', linewidths=2)
    ax.invert_yaxis()
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title('$u$')
    plt.colorbar(im1, ax=ax, label='m/s')
    
    # Plot meridional wind
    ax = axes[1]
    levels_v = np.arange(-4, 4.1, 0.4)
    im2 = ax.contourf(ds['lat'], ds['pfull'], vcomp_mean, 
                      levels=levels_v, cmap='RdBu_r', extend='both')
    ax.contour(ds['lat'], ds['pfull'], vcomp_mean, 
               levels=[0], colors='black', linewidths=2)
    ax.invert_yaxis()
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title('$v$')
    plt.colorbar(im2, ax=ax, label='m/s')

    fig.suptitle('Time-zonal mean winds')
    
    return fig, axes


def plot_temporal_variances(ds, figsize=(17, 5)):
    """
    Plot temporal variance of surface pressure, 850-hPa vorticity, 
    and 500-hPa geopotential height.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing 'ps', 'vor', and 'height' variables
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, layout='constrained')
    
    # Variance of surface pressure
    ps_var = ds['ps'].var(dim='time')
    
    # Variance of 850-hPa vorticity
    vor_var = ds['vor'].sel(pfull=850.0, method='nearest').var(dim='time')
    
    # Variance of 500-hPa geopotential height
    height_var = ds['height'].sel(pfull=500.0, method='nearest').var(dim='time')

    
    # Plot surface pressure variance
    ax = axes[0]
    im1 = ax.contourf(ds['lon'], ds['lat'], ps_var / 100,  # Convert to hPa^2
                      levels=15, cmap='viridis')
    ax.set_title('surface pressure')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.grid(True, color='gray', alpha=0.5, linestyle='--')
    plt.colorbar(im1, ax=ax, orientation='horizontal', label='hPa²')
    
    # Plot 850-hPa vorticity variance
    ax = axes[1]
    im2 = ax.contourf(ds['lon'], ds['lat'], vor_var * 1e10,  # Scale for visibility
                      levels=15, cmap='plasma')
    ax.set_title('850-hPa vorticity')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.grid(True, color='gray', alpha=0.5, linestyle='--')
    plt.colorbar(im2, ax=ax, orientation='horizontal', label='×10⁻¹⁰ s⁻²')
    
    # Plot 500-hPa geopotential height variance
    ax = axes[2]
    im3 = ax.contourf(ds['lon'], ds['lat'], height_var,
                      levels=15, cmap='inferno')
    ax.set_title('$Z_{500}$')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.grid(True, color='gray', alpha=0.5, linestyle='--')
    plt.colorbar(im3, ax=ax, orientation='horizontal', label='m²')
    
    fig.suptitle("Temporal variances")
    return fig, axes


def plot_all_climatology(ds, save_prefix=None):
    """
    Generate all climatology plots.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing all required variables
    save_prefix : str, optional
        If provided, save figures with this prefix
    
    Returns:
    --------
    dict : Dictionary containing all figure objects
    """
    figures = {}
    
    # Surface means
    fig1, _ = plot_surface_means(ds)
    figures['surface_means'] = fig1
    if save_prefix:
        fig1.savefig(f'{save_prefix}_surface_means.png', dpi=300, bbox_inches='tight')
    
    # Zonal mean cross-sections
    fig2, _ = plot_zonal_mean_cross_sections(ds)
    figures['zonal_cross_sections'] = fig2
    if save_prefix:
        fig2.savefig(f'{save_prefix}_zonal_cross_sections.png', dpi=300, bbox_inches='tight')
    
    # Temporal variances
    fig3, _ = plot_temporal_variances(ds)
    figures['temporal_variances'] = fig3
    if save_prefix:
        fig3.savefig(f'{save_prefix}_temporal_variances.png', dpi=300, bbox_inches='tight')
    
    return figures


# Example usage:
# figures = plot_all_climatology(ds, save_prefix='gcm_climatology')
# plt.show()
if __name__ == '__main__':

    from os.path import join
    import sys
    import xarray as xr 

    exp_dir = sys.argv[1]
    out_prefix = sys.argv[2]
    # files = [join(exp_dir, f'run{i_mem:04d}', 'atmos_6_hourly.nc') for i_mem in (25, 26, 27, 28)]
    # ds = xr.concat([xr.open_dataset(fn) for fn in files], dim='time')
    file_n = join(exp_dir, "run0025", "atmos_6_hourly.nc")
    ds = xr.open_dataset(file_n)
    climo_figs = plot_all_climatology(ds, save_prefix=out_prefix)