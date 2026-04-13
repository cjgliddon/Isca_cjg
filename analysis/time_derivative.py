import numpy as np
import xarray as xr
from typing import Literal

def _cftime_to_seconds(time_coord) -> np.ndarray:
    """Convert a cftime coordinate array to seconds since the first time step."""
    times = time_coord.values
    t0 = times[0]
    return np.array([(t - t0).total_seconds() for t in times])


def _compute_time_derivative(
    data: np.ndarray,
    t_seconds: np.ndarray,
    method: Literal["forward", "backward", "centered"],
) -> np.ndarray:
    """
    Compute partial time derivatives along the first axis of `data`
    using the specified finite differencing method.

    Parameters
    ----------
    data : np.ndarray
        Data array with time along axis 0.
    t_seconds : np.ndarray
        1-D array of time values in seconds (length == data.shape[0]).
    method : str
        One of 'forward', 'backward', or 'centered'.

    Returns
    -------
    deriv : np.ndarray
        Array of the same shape as `data`; boundary points that cannot be
        computed with the chosen scheme are filled with NaN.
    """
    n = len(t_seconds)
    deriv = np.full_like(data, np.nan, dtype=float)

    if method == "forward":
        # Use points i and i+1; undefined at the last time step
        for i in range(n - 1):
            dt = t_seconds[i + 1] - t_seconds[i]
            deriv[i] = (data[i + 1] - data[i]) / dt

    elif method == "backward":
        # Use points i-1 and i; undefined at the first time step
        for i in range(1, n):
            dt = t_seconds[i] - t_seconds[i - 1]
            deriv[i] = (data[i] - data[i - 1]) / dt

    elif method == "centered":
        # Use points i-1 and i+1; undefined at both endpoints
        for i in range(1, n - 1):
            dt = t_seconds[i + 1] - t_seconds[i - 1]
            deriv[i] = (data[i + 1] - data[i - 1]) / dt

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'forward', 'backward', or 'centered'."
        )

    return deriv


def time_derivative(
    ds: xr.Dataset | xr.DataArray,
    method: Literal["forward", "backward", "centered"] = "forward",
) -> xr.Dataset | xr.DataArray:
    """
    Compute numerical partial time derivatives of all fields in an xarray
    Dataset or DataArray.

    The derivative is computed with respect to the `time` coordinate using
    finite differencing. Boundary points where the chosen scheme is undefined
    are set to NaN.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Input data containing a `time` coordinate composed of cftime objects.
    method : {'forward', 'backward', 'centered'}, optional
        Finite differencing scheme to use. Default is 'forward'.
        - 'forward'  : (f[i+1] - f[i])   / (t[i+1] - t[i])   — NaN at last step
        - 'backward' : (f[i]   - f[i-1]) / (t[i]   - t[i-1]) — NaN at first step
        - 'centered' : (f[i+1] - f[i-1]) / (t[i+1] - t[i-1]) — NaN at both ends

    Returns
    -------
    xr.Dataset or xr.DataArray
        New object of the same type and shape as `ds`. Each variable's units
        attribute (if present) is updated to '[original unit] s**-1'.
    """
    t_seconds = _cftime_to_seconds(ds["time"])

    def _differentiate_dataarray(da: xr.DataArray) -> xr.DataArray:
        # Move time to axis 0 for uniform indexing, compute, then transpose back
        dim_order = da.dims
        da_t_first = da.transpose("time", ...)

        deriv_values = _compute_time_derivative(
            data=da_t_first.values,
            t_seconds=t_seconds,
            method=method,
        )

        # Build the output DataArray with the same coordinates / dims
        da_deriv = xr.DataArray(
            data=deriv_values,
            coords=da_t_first.coords,
            dims=da_t_first.dims,
            name=da.name,
            attrs=da.attrs.copy(),
        ).transpose(*dim_order)   # restore original dimension order

        # Update units attribute
        original_units = da.attrs.get("units", "?")
        da_deriv.attrs["units"] = f"{original_units} s**-1"
        da_deriv.attrs["derivative_method"] = method

        return da_deriv

    # ---- DataArray branch ------------------------------------------------
    if isinstance(ds, xr.DataArray):
        return _differentiate_dataarray(ds)

    # ---- Dataset branch --------------------------------------------------
    deriv_vars = {}
    for var in ds.data_vars:
        if "time" in ds[var].dims:
            deriv_vars[var] = _differentiate_dataarray(ds[var])
        else:
            # Variables without a time dimension are passed through unchanged
            deriv_vars[var] = ds[var]

    ds_deriv = xr.Dataset(deriv_vars, coords=ds.coords, attrs=ds.attrs.copy())
    ds_deriv.attrs["derivative_method"] = method
    return ds_deriv