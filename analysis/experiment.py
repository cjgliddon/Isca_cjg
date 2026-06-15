import os; from os.path import join
from pathlib import Path

#GFDL_STORAGE = os.environ['GFDL_STORAGE']
GFDL_STORAGE = '/orcd/data/talia_tb/002/aqua_gcm_runs'      # used for testing

def PredictabilityExperiment():
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
    def __init__(self, name, root_folder=GFDL_STORAGE, ens_subdirs=[]):
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
        """
        self.name     = name
        self.path     = join(root_folder, name)
        self.ctrl_dir = join(self.path, 'control')
        self.postproc = join(self.ctrl_dir, 'postprocessed')
        self.ctrk_dir = join(self.ctrl_dir, 'tracks')
        self.ens_dir  = join(self.path, 'ensembles')
        self.ens_subdirs = ens_subdirs

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
