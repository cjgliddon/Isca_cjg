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


# Let's do a run!
if __name__=="__main__":
    # cb.compile()  # compile the source code to working directory $GFDL_WORK/codebase

    t0_mem = float(sys.argv[1])         # perturbation time
    mem_range = range(                  # which ensemble members to run
        int(sys.argv[2], int(sys.argv[3])+1)
        )
    # these lines make sure the working directory is different for each member, allowing the ensembles to run in parallel
    exp.workdir = os.path.join(exp.workdir, "t0_" + str(t0_mem))
    print(f'New working directory: {exp.workdir}')
    exp.rundir = os.path.join(exp.workdir, 'run')          # temporary area an individual run will be performed


    # specify where restart file is saved

    dt = exp.namelist['main_nml']['dt_atmos']
    dt_save_days = 0.25
    ctrl_chunk_len = 30
    n_days_after_pert = 30
    run_len = n_days_after_pert + (t0_mem - dt_save_days) % ctrl_chunk_len + dt_save_days    # days; we want to make sure we get a full 30 days *after* each perturbation. Once again, in the edge case where t0_mem % 30 = 0 we want a 60-day run instead
    exp.namelist['main_nml']['days']  = int(run_len)
    exp.namelist['main_nml']['hours'] = int(24*(run_len % 1))   # fractional part x 24 h
    nrestart = int((t0_mem - dt_save_days ) // ctrl_chunk_len)      # edge case: if t0_mem % 30 = 0, then we restart on the previous step to get last timestep
    restartf = os.path.join(GFDL_DATA, expname, 'restarts', ('res%04d.tar.gz' % (nrestart)))


    pfrac = 2.0E-2      # "perturbation fraction"

    datadir = 'ensembles'
    exp.runfmt = os.path.join(datadir, str(t0_mem), 'mem%02d')

    # run the experiment!
    for i_mem in mem_range:
        exp.namelist['spectral_dynamics_nml'].update({
            'seconds_to_perturb': int(t0_mem*86400 + dt),
            'seed_values': [i_mem],
            'perturbation_fraction': [pfrac],
        })
        exp.run(i_mem, restart_file=restartf, num_cores=NCORES, overwrite_data=True, save_restarts=False)

        print(f"Ensemble member {i_mem} for start time {t0_mem} run successfully completed")

        # This "trims" off the portion of the ensemble run that doesn't 
        trim_data = True
        if trim_data:
            n_timesteps = int(run_len/dt_save_days)
            it0, itf = n_timesteps - int(n_days_after_pert/dt_save_days), n_timesteps
            fn_dir = os.path.join(GFDL_DATA, expname, exp.runfmt % i_mem)
            subprocess.run([os.path.join(GFDL_BASE, "postprocessing", "trim_output_times.sh"),
                            fn_dir, str(it0), str(itf)])

        print(f"Ensemble member {i_mem} for start time {t0_mem} data successfully trimmed")