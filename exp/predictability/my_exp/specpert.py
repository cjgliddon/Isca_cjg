# This script is used to run the ensemble breeding for the frierson_default experiment.

import glob
import os
import shutil
import sys
import subprocess

import numpy as np

from isca import IscaCodeBase, DiagTable, Experiment, Namelist, GFDL_BASE, GFDL_DATA
from shared_vars import expname, my_diag, my_namelist, GFDL_STORAGE

NCORES = 16
base_dir = os.path.dirname(os.path.realpath(__file__))
# a CodeBase can be a directory on the computer,
# useful for iterative development
cb = IscaCodeBase.from_directory(GFDL_BASE)

# or it can point to a specific git repo and commit id.
# This method should ensure future, independent, reproducibility of results.
# cb = DryCodeBase.from_repo(repo='https://github.com/isca/isca', commit='isca1.1')

# compilation depends on computer specific settings.  The $GFDL_ENV
# environment variable is used to determine which `$GFDL_BASE/src/extra/env` file
# is used to load the correct compilers.  The env file is always loaded from
# $GFDL_BASE and not the checked out git repo.

# create an Experiment object to handle the configuration of model parameters
# and output diagnostics
#exp = Experiment('frierson_pert_test', codebase=cb)
exp = Experiment(expname, codebase=cb)

#Tell model how to write diagnostics
exp.diag_table = my_diag

#Empty the run directory ready to run
exp.clear_rundir()

# Define values for the 'core' namelist
# include perturbation parameters
exp.namelist = my_namelist
exp.namelist['spectral_dynamics_nml'].update({
    'do_perturbation': True,
    'pert_method': 'rand',
    'grid_type': 'spectral',
    'num_perturbations_actual': 1,
})
exp.set_resolution('T85', 30)


#Lets do a run!
if __name__=="__main__":
    # cb.compile()  # compile the source code to working directory $GFDL_WORK/codebase

    t0_mem = float(sys.argv[1])         # perturbation time
    i_mem = int(sys.argv[2])            # ensemble member (number affects RNG seed)
    # these lines make sure the working directory is different for each member, allowing the ensembles to run in parallel
    exp.workdir = os.path.join(exp.workdir, "mem" + str(i_mem))
    print(f'New working directory: {exp.workdir}')
    exp.rundir = os.path.join(exp.workdir, 'run')          # temporary area an individual run will be performed


    # specify where restart file is saved
    nrestart = int(t0_mem // 30)
    restartf = os.path.join(GFDL_DATA, expname, 'restarts', ('res%04d.tar.gz' % (nrestart)))

    dt = exp.namelist['main_nml']['dt_atmos']
    run_len = int(20 + np.ceil(t0_mem % 30))    # days; we want to make sure we get a full 30 days *after* each perturbation
    exp.namelist['main_nml']['days'] = run_len

    pfrac = 2.0E-2      # "perturbation fraction"

    datadir = 'ensembles'
    subdir = f'spec_mag_{pfrac}'
    exp.runfmt = os.path.join(datadir, str(t0_mem), subdir, 'b%02d')

    # update namelist dictionary with parameters relevant to breeding
    exp.namelist['spectral_dynamics_nml'].update({
        'seconds_to_perturb': int(t0_mem*86400 + dt),
        'seed_values': [i_mem],
        'perturbation_fraction': [pfrac],
    })
    # run the experiment!
    exp.run(i_mem, restart_file=restartf, num_cores=NCORES, overwrite_data=True, save_restarts=False)

    do_tracking = bool(int(sys.argv[3]))
    if do_tracking: 
        climo_file = os.path.join(GFDL_DATA, expname, 'postprocessed', 'base_t_mean.nc')
        print("Climatology file stored in:")
        print(climo_file)       
        
        ident = np.random.randint(100000, 999999)
        print(f"Ensemble member {i_mem} initialized at {t0_mem} has random ident {ident}")
        num_tracking_chunks = int(np.ceil(run_len*4 / 80)) 
        print(f"Number of chunks used in tracking: {num_tracking_chunks}")
        subprocess.run([os.path.join(base_dir, "tracking_single.bash"), climo_file, 
                        os.path.join(GFDL_DATA, expname, exp.runfmt % i_mem), str(ident), str(num_tracking_chunks)])
    do_cleanup = bool(int(sys.argv[4]))
    print(f"Cleanup? {do_cleanup}")
    if do_cleanup:
        files_to_remove = glob.glob(os.path.join(GFDL_DATA, expname, exp.runfmt % i_mem, "*.nc"))
        for fn in files_to_remove:
            os.remove(fn)
    destination = os.path.join(GFDL_STORAGE, expname, exp.runfmt % i_mem)
    if os.path.exists(destination):
        shutil.rmtree(destination)
    shutil.move(os.path.join(GFDL_DATA, expname, exp.runfmt % i_mem), destination)