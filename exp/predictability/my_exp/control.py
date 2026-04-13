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
my_diag.add_field('dynamics', 'ucomp', time_avg=False)
my_diag.add_field('dynamics', 'vcomp', time_avg=False)
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


#Lets do a run!
if __name__=="__main__":
    print(f"Current directory: {base_dir}")
    cb.compile()  # compile the source code to working directory $GFDL_WORK/codebase

    # passing no command line options causes the model to perform spinup
    # first, check if the restart files for first 2 years exist
    # TODO: make this even more elegant so model automatically restarts where it left off
    res_path = os.path.join(exp.restartdir, 'res0024.tar.gz')
    if not os.path.exists(res_path):
        exp.run(1, use_restart=False, num_cores=NCORES, overwrite_data=True)
        for i in range(2,25):
            exp.run(i, num_cores=NCORES, overwrite_data=True)
    else:
        for i in range(25, 50):
            exp.run(i, num_cores=NCORES, overwrite_data=True)
        # only do tracking after we've run the full ensemble
        if len(sys.argv) > 1:
            do_tracking = bool(int(sys.argv[1]))
            move_to_storage = bool(int(sys.argv[2]))
            if do_tracking:
                subprocess.run([os.path.join(base_dir, "control-tracking.sh"), expname])
            if move_to_storage:
                for i in range(1, 50):
                    destination = os.path.join(GFDL_STORAGE, expname, exp.runfmt % i)
                    if os.path.exists(destination):
                        shutil.rmtree(destination)
                    shutil.move(os.path.join(GFDL_DATA, expname, exp.runfmt % i), destination)
                if do_tracking:
                    shutil.move(os.path.join(GFDL_DATA, expname, 'postprocessed'), os.path.join(GFDL_DATA, expname, 'postprocessed'))
                    shutil.move(os.path.join(GFDL_DATA, expname, 'tracks'), os.path.join(GFDL_DATA, expname, 'tracks'))
