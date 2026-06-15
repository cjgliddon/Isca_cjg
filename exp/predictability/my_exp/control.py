import os
import sys
import subprocess
import shutil

import numpy as np

from isca import IscaCodeBase, DiagTable, Experiment, Namelist, GFDL_DATA, GFDL_BASE
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
exp = Experiment(expname, codebase=cb)

#Tell model how to write diagnostics
#my_diag.add_field('dynamics', 'ucomp', time_avg=False)
#my_diag.add_field('dynamics', 'vcomp', time_avg=False)
exp.diag_table = my_diag

#Empty the run directory ready to run
exp.clear_rundir()

exp.namelist = my_namelist
exp.namelist['main_nml'].update({
     'days'   : 30,
     'hours'  : 0,
     'minutes': 0,
     'seconds': 0,
     'current_date' : [1,1,1,0,0,0],
     'calendar' : 'thirty_day'
    })
exp.set_resolution('T85', 30)


# Lets do a run!
if __name__=="__main__":
    print(f"Current directory: {base_dir}")

    # recompile code, but only if we want (turned off for now because of bugs in the codebase)
    recompile = False
    if recompile:
        cb.compile()  # compile the source code to working directory $GFDL_WORK/codebase

    # get the initial and final chunk numbers of the run
    run_i = int(sys.argv[1])
    if len(sys.argv) > 2:
        run_f = int(sys.argv[2]) + 1
    else:
        run_f = run_i + 1

    for j in range(run_i, run_f):
        use_restart = False if j == 1 else True     # don't use restart file on first run of spinup
        exp.run(j, num_cores=NCORES, overwrite_data=True, use_restart=use_restart)
        if len(sys.argv) > 3:
            move_to_storage = bool(int(sys.argv[3]))
            if move_to_storage:
                print(f"Moving run {j} to storage...")
                destination = os.path.join(GFDL_STORAGE, expname, 'control', exp.runfmt % j)
                if os.path.exists(destination):
                    shutil.rmtree(destination)
                shutil.move(os.path.join(GFDL_DATA, expname, 'control', exp.runfmt % j), destination)