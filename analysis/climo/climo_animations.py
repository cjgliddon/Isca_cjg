import cftime
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.animation import FuncAnimation
# from IPython.display import HTML

def animate_zonal_mean_winds(ds, interval=250, save_path=None):
    """
    Animate lat-vs-pressure cross sections of zonal-mean zonal and meridional winds.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing 'ucomp' and 'vcomp' variables
    interval : int
        Delay between frames in milliseconds (default: 100)
    save_path : str, optional
        If provided, save animation to this path (e.g., 'animation.mp4' or 'animation.gif')
    
    Returns:
    --------
    anim : matplotlib.animation.FuncAnimation object
    """
    # Compute zonal means
    ucomp_zm = ds['ucomp'].mean(dim='lon')
    vcomp_zm = ds['vcomp'].mean(dim='lon')
    
    # Set up the figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), layout='constrained')
    
    # Determine contour levels based on full data range
    u_min, u_max = float(ucomp_zm.min()), float(ucomp_zm.max())
    v_min, v_max = float(vcomp_zm.min()), float(vcomp_zm.max())
    u_max = np.max((np.abs(u_min), np.abs(u_max)))
    v_max = np.max((np.abs(v_min), np.abs(v_max)))
    levels_u = np.linspace(-u_max, u_max, 25)
    levels_v = np.linspace(-v_max, v_max, 25)
    
    # Initialize plots
    ax_u = axes[0]
    ax_v = axes[1]
    
    # First frame
    im_u = ax_u.contourf(ds['lat'], ds['pfull'], ucomp_zm.isel(time=0),
                         levels=levels_u, cmap='RdBu_r', extend='both')
    cont_u = ax_u.contour(ds['lat'], ds['pfull'], ucomp_zm.isel(time=0),
                          levels=[0], colors='black', linewidths=2)
    ax_u.invert_yaxis()
    ax_u.set_ylabel('Pressure (hPa)')
    title_u = ax_u.set_title('')
    cbar_u = plt.colorbar(im_u, ax=ax_u, label='m/s')
    
    im_v = ax_v.contourf(ds['lat'], ds['pfull'], vcomp_zm.isel(time=0),
                         levels=levels_v, cmap='RdBu_r', extend='both')
    cont_v = ax_v.contour(ds['lat'], ds['pfull'], vcomp_zm.isel(time=0),
                          levels=[0], colors='black', linewidths=2)
    ax_v.invert_yaxis()
    ax_v.set_xlabel('Latitude')
    ax_v.set_ylabel('Pressure (hPa)')
    title_v = ax_v.set_title('')
    cbar_v = plt.colorbar(im_v, ax=ax_v, label='m/s')
        
    def update(frame):
        # Clear previous contours
        for coll in ax_u.collections:
            coll.remove()
        for coll in ax_v.collections:
            coll.remove()
        
        # Update zonal wind
        ax_u.contourf(ds['lat'], ds['pfull'], ucomp_zm.isel(time=frame),
                     levels=levels_u, cmap='RdBu_r', extend='both')
        ax_u.contour(ds['lat'], ds['pfull'], ucomp_zm.isel(time=frame),
                    levels=[0], colors='black', linewidths=2)
        title_u.set_text(f'Zonal-Mean Zonal Wind - Time step {frame}')
        
        # Update meridional wind
        ax_v.contourf(ds['lat'], ds['pfull'], vcomp_zm.isel(time=frame),
                     levels=levels_v, cmap='RdBu_r', extend='both')
        ax_v.contour(ds['lat'], ds['pfull'], vcomp_zm.isel(time=frame),
                    levels=[0], colors='black', linewidths=2)
        title_v.set_text(f'Zonal-Mean Meridional Wind - Time step {frame}')
        
        return []
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(ds['time']),
                        interval=interval, blit=False)
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000/interval)
        else:
            anim.save(save_path, writer='ffmpeg', fps=1000/interval)
        print(f"Animation saved to {save_path}")
    
    return anim


def animate_synoptic_fields(ds, interval=100, save_path=None, tracks=None, 
                           intensity_scale=50):
    """
    Animate 850-hPa vorticity (filled contours), surface pressure (dashed black contours),
    and 500-hPa geopotential height (solid blue contours), with optional feature tracks.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing 'vor', 'ps', and 'height' variables
    interval : int
        Delay between frames in milliseconds (default: 100)
    save_path : str, optional
        If provided, save animation to this path (e.g., 'animation.mp4' or 'animation.gif')
    tracks : dict, optional
        Dictionary of tracked features. Each value is a numpy array of shape (n, 4)
        with columns: [time (days), longitude, latitude, intensity]
    intensity_scale : float
        Scaling factor for marker sizes based on intensity (default: 50)
    
    Returns:
    --------
    anim : matplotlib.animation.FuncAnimation object
    """
    # Extract variables
    vor_850 = ds['vor'].sel(pfull=850.0, method='nearest')
    ps = ds['ps']
    height_500 = ds['height'].sel(pfull=500.0, method='nearest')
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6), layout='constrained',
                          subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Determine contour levels based on full data range
    vor_min, vor_max = float(vor_850.min()), float(vor_850.max())
    levels_vor = np.linspace(vor_min, vor_max, 20)
    
    ps_min, ps_max = float(ps.min()/100), float(ps.max()/100)  # Convert to hPa
    levels_ps = np.linspace(ps_min, ps_max, 15)
    
    height_min, height_max = float(height_500.min()), float(height_500.max())
    levels_height = np.linspace(height_min, height_max, 15)
    
    # Initialize plot
    title = ax.set_title('')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    # First frame
    im_vor = ax.contourf(ds['lon'], ds['lat'], vor_850.isel(time=0),
                        levels=levels_vor, cmap='RdBu_r', extend='both',
                        transform=ccrs.PlateCarree())
    cont_ps = ax.contour(ds['lon'], ds['lat'], ps.isel(time=0)/100,
                        levels=levels_ps, colors='black', linewidths=1,
                        linestyles='dashed', transform=ccrs.PlateCarree())
    cont_height = ax.contour(ds['lon'], ds['lat'], height_500.isel(time=0),
                            levels=levels_height, colors='blue', linewidths=1.5,
                            transform=ccrs.PlateCarree())
    
    # Add colorbar for vorticity
    cbar = plt.colorbar(im_vor, ax=ax, orientation='horizontal', pad=0.05,
                       label='850-hPa Vorticity (s⁻¹)', shrink=0.8)
        
    # Convert cftime to days since start for matching with track times
    time_axis = ds['time']
    # Calculate days since first time step
    if hasattr(time_axis.values[0], 'dayofyr'):
        # For cftime objects, calculate days from start
        start_time = time_axis.values[0]
        days_since_start = np.array([
            (t.year - start_time.year) * 360 + (t.dayofyr - start_time.dayofyr)
            for t in time_axis.values
        ])
    else:
        # Fallback if time is already numeric
        days_since_start = np.arange(len(time_axis))
    
    def update(frame):
        # Clear previous contours
        for coll in ax.collections:
            coll.remove()
        
        # Update vorticity (filled)
        ax.contourf(ds['lon'], ds['lat'], vor_850.isel(time=frame),
                   levels=levels_vor, cmap='RdBu_r', extend='both',
                   transform=ccrs.PlateCarree())
        
        # Update surface pressure (dashed black)
        ax.contour(ds['lon'], ds['lat'], ps.isel(time=frame)/100,
                  levels=levels_ps, colors='black', linewidths=1,
                  linestyles='dashed', transform=ccrs.PlateCarree())
        
        # Update 500-hPa height (solid blue)
        ax.contour(ds['lon'], ds['lat'], height_500.isel(time=frame),
                  levels=levels_height, colors='blue', linewidths=1.5,
                  transform=ccrs.PlateCarree())
        
        # Add tracked features if provided
        if tracks is not None:
            current_day = days_since_start[frame]
            
            for track_id, track_data in tracks.items():
                # Extract track information
                track_times = track_data[:, 0]
                track_lons = track_data[:, 1]
                track_lats = track_data[:, 2]
                track_intensities = track_data[:, 3]
                
                # Find points that match current time (within a small tolerance)
                time_tolerance = 0.25  # days
                mask = np.abs(track_times - current_day) < time_tolerance
                
                if np.any(mask):
                    lons_now = track_lons[mask]
                    lats_now = track_lats[mask]
                    intensities_now = track_intensities[mask]
                    
                    # Plot features with size proportional to intensity
                    sizes = np.abs(intensities_now) * intensity_scale
                    ax.scatter(lons_now, lats_now, s=sizes, c='black', 
                             edgecolors='white', linewidths=0.5, alpha=0.8,
                             transform=ccrs.PlateCarree(), zorder=5)
        
        title.set_text(f'850-hPa Vorticity, SLP (dashed), 500-hPa Height (blue) - Time step {frame}')
        
        return []
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(ds['time']),
                        interval=interval, blit=False)
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000/interval)
        else:
            anim.save(save_path, writer='ffmpeg', fps=1000/interval)
        print(f"Animation saved to {save_path}")
    
    return anim

if __name__ == '__main__':
    
    import os
    from os.path import join
    import pickle
    import sys
    import xarray as xr

    GFDL_STORAGE=os.environ['GFDL_STORAGE']
    expname = sys.argv[1]
    expname_save = expname.replace('/', '_')
    exproot = join(GFDL_STORAGE, expname)

    ds = xr.open_dataset(join(exproot, 'ensembles', '690.0', 'spec_mag_0.02', 'b11', 'atmos_6_hourly.nc'))
#    tracks = pickle.load(open(join(GFDL_STORAGE)))
    save_out_dir = join('figures', 'climo_plots')
    anim1 = animate_zonal_mean_winds(ds, save_path=join(save_out_dir, f"{expname_save}_winds.gif"))

#    anim2 = animate_synoptic_fields(ds, save_path='figures/frierson_moist_es0_0.75_synoptic.gif', tracks=tracks)
