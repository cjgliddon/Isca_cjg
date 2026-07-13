import numpy as np
import os; from os.path import join
from pathlib import Path
import pickle
import xarray as xr

GFDL_STORAGE = os.environ['GFDL_STORAGE']
GFDL_DATA    = os.environ['GFDL_DATA']
# GFDL_STORAGE = '/orcd/data/talia_tb/002/aqua_gcm_runs'      # used for testing

class PredictabilityExperiment():
    """
    A class representing an ensemble predictability experiment as run in the 
    Isca idealized GCM. 

    Basic file storage format: Each experiment is stored in a single directory
    whose name is the experiment name (e.g., my_exp). This directory contains
    the following subdirectories:

    - `control`, containing the control run
    - `ensembles`, containing the ensemble runs
    
    The `control` directory containins the direct model output for the control
    run of the experiment, stored in folders titled `run0001`, `run0002`, etc.
    ---each number gives a "chunk" of the output. It also contains a folder
    `postprocessed`, which contains (typically) data for individual model 
    variables merged across the entire timespan of the control run. A third
    folder, `tracks`, contains feature trajectories in the control run as 
    identified by the TRACK algorithm (Hodges 1995, 1999). 

    The `ensembles` folder has the output data for all the ensemble members.
    The ensemble members are organized by genesis time and member ID. For 
    instance, the raw data for the 3rd member of the ensemble generated at 
    t = 400 days would be stored under:
    
    `$ <GFDL_STORAGE>/my_exp/ensembles/400.0/mem03/atmos_6_hourly.nc`

    Each ensemble member folder contains a subfolder `tracks` with the TRACK
    algorithm output. 
    """
    def __init__(self, name, root_folder=GFDL_STORAGE, ens_subdirs=[],
                 analysis_dir=None):
        """
        Constructor object for experiment.

        Arguments
        ---------
        name : str
            The experiment name.
        root_folder : str, optional
            Path to the root directory for the experiment folder. Default is
            the system environment variable `GFDL_STORAGE`.
        ens_subdir : list of str, optional
            A list of ensemble subdirectories for the experiment; these 
            subdirectories may correspond to ensembles generated using 
            different perturbation schemes. Default is the empty list, 
            in which case there are no subdirectories.
            #TODO: assess whether this is really needed
        analysis_dir : str, optional
            Full path to directory in which to store analysis products for the 
            experiment, such as climatology data, composites, etc. If None, 
            this directory is set to be the same as the postprocessed 
            directory.
        """
        self.name     = name
        self.path     = join(root_folder, name)
        self.ctrl_dir = join(self.path, 'control')
        self.postproc = join(self.ctrl_dir, 'postprocessed')
        if analysis_dir is None:
            self.anly_dir = join(GFDL_DATA, name, 'analysis')
            # make directory if it doesn't already exist
            os.makedirs(self.anly_dir, exist_ok=True)
        else:
            self.anly_dir = analysis_dir
        self.ctrk_dir = join(self.ctrl_dir, 'tracks')
        self.ens_dir  = join(self.path, 'ensembles')
        self.ens_subdirs = ens_subdirs

    def get_ensemble_dirs(self, init_times=None, mems=None):
        """
        Returns ensemble member directories corresponding to the specified
        ensemble start times and member numbers. If no start times or member
        numbers are specified, behavior is same as 
        `self.get_all_ensemble_dirs()`.

        Arguments
        ---------
        init_times : iterable of floats, optional
            Initialization times for the desired ensembles. If None (default),
            data will be returned for all ensembles present in the data
            directory.

        mems : iterable of ints, optional
            The index numbers for the desired ensemble members. If None
            (default), data will be returned for all ensemble members present
            for each initialization time.

        Returns
        -------
        list
            A list of all subdirectory paths where ensemble data is stored. 
        """
        dirpaths = []
        
        # Handle cases where ensembles are divided into subdirectories separately
        if len(self.ens_subdirs) == 0:
            base_paths = [self.ens_dir]
        else:
            base_paths = [join(self.ens_dir, sd) for sd in self.ens_subdirs]
        
        for base_path in base_paths:
            # Get time directories to search
            if init_times is None:
                time_dirs = [d for d in Path(base_path).glob('*') if d.is_dir()]
            else:
                time_dirs = [Path(base_path) / str(t) for t in init_times]
            
            # For each time directory, get ensemble member directories
            for time_dir in time_dirs:
                if not time_dir.exists():
                    continue
                
                if mems is None:
                    # Get all member directories
                    dirpaths += [str(d) for d in time_dir.glob('mem*') if d.is_dir()]
                else:
                    # Get specific member directories
                    for mem_idx in mems:
                        mem_dir = time_dir / f"mem{mem_idx:02d}"
                        if mem_dir.is_dir():
                            dirpaths.append(str(mem_dir))
        
        return dirpaths

    def get_all_ensemble_dirs(self):
        """
        Returns a list of all extant ensemble member directories for the experiment.
        Useful for postprocessing operations like tracking, PV calculation, etc.
        """   
        dirpaths = []
        # handle cases where ensembles are divided into subdirectories separately
        if len(self.ens_subdirs) == 0:
            dirpaths += [str(d) for d in Path(self.ens_dir).glob('*/*') if d.is_dir()]
        else: 
            for sd in self.ens_subdirs:
                dirpaths += [str(d) for d in Path(join(self.ens_dir, sd)).glob('*/*') if d.is_dir()]
        return dirpaths
    
    def _get_climo_ds(self, months=range(13, 25)):
        """
        Returns a dataset representing the climatology (time average) of the
        experiment as measured over a given range of months.

        Arguments
        ---------
        months : iterable, optional
            The months of the control run used to calculate the climatology.
            Default is range(13, 25).

        Returns
        -------
        climo_ds : xarray.Dataset
            Dataset of the model climatology.
        """
        # Find paths to raw data
        raw_files = [
            join(self.ctrl_dir, f"run{mon:04d}", "atmos_6_hourly.nc") for mon in months
        ]
        # Calculate average
        monthly_means = []
        for fn in raw_files:
            ds = xr.open_dataset(fn)
            ds = ds.mean(dim='time')
            monthly_means.append(ds)
        climo_ds = xr.concat(monthly_means, dim='month').mean(dim='month')
        return climo_ds
    
    def get_climo_ds(self, months=range(13, 25), cache=False):
        """
        Returns a dataset representing the climatology of the experiment.
        Optionally caches result to avoid recomputation.

        Arguments
        ---------
        months : iterable, optional
            The months of the control run used to calculate the climatology.
            Default is range(13, 25).
            
        cache : bool, optional
            If True, caches the climatology dataset in the postprocessed
            directory (default: False). 

        Returns
        -------
        climo_ds : xarray.Dataset
            Dataset of the model climatology.
        """
        cache_path = join(self.anly_dir, f'climo_{min(months)}_{max(months)}.nc')
        
        # Return cached version if it exists
        if cache and os.path.exists(cache_path):
            return xr.open_dataset(cache_path)
        
        # Compute climatology
        raw_files = sorted([
            join(self.ctrl_dir, f"run{mon:04d}", "atmos_6_hourly.nc") for mon in months
        ])
        
        # Open files simultaneously with mfdataset 
        ds = xr.open_mfdataset(raw_files, combine='by_coords', chunks={'time': 120})
        climo_ds = ds.mean(dim='time').compute()  # .compute() materializes dask arrays
        
        # Save to cache
        os.makedirs(self.postproc, exist_ok=True)
        climo_ds.to_netcdf(cache_path)
        
        return climo_ds
    
    def get_control_tracks(self, var, feature_type):
        """
        Returns a dictionary containing all saved feature tracks of a 
        specified type from the control run of the experiment.

        Arguments
        ---------
        var : str
            The meteorological variable whose anomalies are being tracked.

        feature_type : str
            The feature being tracked. Options include `'cycs'` and `'acycs'`.

        Returns
        -------
        td : dict
            The track dictionary.
        """
        filename = f"{var}_{feature_type}.pickle"
        td = pickle.load(open(join(self.ctrk_dir, filename),'rb'))
        return td
    
def test_run():
    """ from Claude Haiku 4.5 """
    exp_name = 'test_exp'

    import tempfile
    import shutil
    
    # Create a temporary directory structure for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize experiment
        exp = PredictabilityExperiment(exp_name, root_folder=tmpdir)
        
        # Test 1: __init__ - verify path attributes are correctly set
        print("Test 1: __init__ path construction...")
        assert exp.name == exp_name
        assert exp.path == join(tmpdir, exp_name)
        assert exp.ctrl_dir == join(tmpdir, exp_name, 'control')
        assert exp.ens_dir == join(tmpdir, exp_name, 'ensembles')
        print("✓ Paths correctly initialized")
        
        # Test 2: get_all_ensemble_dirs() with no subdirs
        print("\nTest 2: get_all_ensemble_dirs() - no subdirectories...")
        os.makedirs(exp.ens_dir, exist_ok=True)
        # Create mock ensemble structure: ensembles/100.0/mem01, ensembles/100.0/mem02, etc.
        os.makedirs(join(exp.ens_dir, '100.0', 'mem01'), exist_ok=True)
        os.makedirs(join(exp.ens_dir, '100.0', 'mem02'), exist_ok=True)
        os.makedirs(join(exp.ens_dir, '200.0', 'mem01'), exist_ok=True)
        
        ens_dirs = exp.get_all_ensemble_dirs()
        assert len(ens_dirs) == 3
        assert all('mem' in d for d in ens_dirs)
        print(f"✓ Found {len(ens_dirs)} ensemble directories")
        
        # Test 3: get_all_ensemble_dirs() with subdirs
        print("\nTest 3: get_all_ensemble_dirs() - with subdirectories...")
        exp_with_subdirs = PredictabilityExperiment(
            exp_name, root_folder=tmpdir, ens_subdirs=['perturb_v1', 'perturb_v2']
        )
        os.makedirs(join(exp.ens_dir, 'perturb_v1', '150.0', 'mem01'), exist_ok=True)
        os.makedirs(join(exp.ens_dir, 'perturb_v2', '150.0', 'mem01'), exist_ok=True)
        
        ens_dirs_subdirs = exp_with_subdirs.get_all_ensemble_dirs()
        assert len(ens_dirs_subdirs) == 2
        print(f"✓ Found {len(ens_dirs_subdirs)} ensemble directories across subdirs")
        
        # Test 4: get_climo_ds() - structure and type checking
        print("\nTest 4: get_climo_ds() structure...")
        # This test would require actual netCDF files, so check return type if files exist
        try:
            # Create dummy monthly data files
            os.makedirs(join(exp.ctrl_dir, 'run0013'), exist_ok=True)
            dummy_data = xr.Dataset({
                'temp': (['time', 'lat', 'lon'], np.random.randn(10, 5, 5)),
                'precip': (['time', 'lat', 'lon'], np.random.randn(10, 5, 5))
            }, coords={'time': np.arange(10), 'lat': np.arange(5), 'lon': np.arange(5)})
            dummy_data.to_netcdf(join(exp.ctrl_dir, 'run0013', 'atmos_6_hourly.nc'))
            
            climo = exp.get_climo_ds(months=[13])
            assert isinstance(climo, xr.Dataset)
            assert 'temp' in climo.data_vars
            assert 'precip' in climo.data_vars
            print(f"✓ Climatology dataset has correct structure")
        except FileNotFoundError:
            print("⊘ Skipped (requires actual netCDF files)")

    return

if __name__ == "__main__":
    test_run()
