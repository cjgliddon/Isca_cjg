import numpy as np
import os; from os.path import join
import subprocess
import sys

sys.path.append('../analysis')
from experiment import PredictabilityExperiment

import pdb

GFDL_STORAGE = os.environ['GFDL_STORAGE']

tracking_argdict=dict({
    'MSLP': dict(
        ctrl_climo_fn  = 'base_t_mean.nc',
        ctrl_rawdat_fn = 'base_anoms.nc',
        pert_rawdat_fn = 'anomaly_ps.nc',
        specfilt_fn = 'specfilt_MSLP_T42_allk.in',
        trackin_fn  = 'RUNDATIN.6hr_MSLP',
        vartup='ps:surf',
    ),
    'vor850': dict(
        ctrl_climo_fn  = 'base_t_mean.nc',
        ctrl_rawdat_fn = 'base_anoms.nc',
        pert_rawdat_fn = 'anomaly_vor850.nc',
        specfilt_fn = 'specfilt_vor_T42_allk.in',
        trackin_fn  = 'RUNDATIN.6hr_VOR_T42',
        vartup = 'vor:850',
    ),
    'Z500': dict(
        ctrl_climo_fn  = 'tmean_height500.nc',
        ctrl_rawdat_fn = 'anomaly_height500.nc',
        pert_rawdat_fn = 'anomaly_height500.nc',
        specfilt_fn = 'specfilt_Z500_T42_allk.in',
        trackin_fn  = 'RUNDATIN.6hr_Z500',
        vartup = 'height:500',
    ),
    'Z250': dict(
        ctrl_climo_fn  = 'tmean_height250.nc',
        ctrl_rawdat_fn = 'anomaly_height250.nc',
        pert_rawdat_fn = 'anomaly_height250.nc',
        specfilt_fn = 'specfilt_Z250_T42_allk.in',
        trackin_fn  = 'RUNDATIN.6hr_Z250',
        vartup = 'height:250',
    ),
})

if __name__ == '__main__':
    print("Got into Main")

    expname = sys.argv[1]
    my_exp = PredictabilityExperiment(expname)

    root = join(GFDL_STORAGE, expname)

    do_merging=False
    compute_ensemble_anoms=True
    control_tracking=False
    ensemble_tracking=True
    parse_tracks=True
#    vars_to_track=['MSLP', 'vor850', 'Z500', 'Z250']
    vars_to_track=['Z250', 'Z500']

    cd = os.getcwd()

    if do_merging:
        os.chdir(join(cd, "merging"))
        subprocess.run(["python", "cdo_postprocess_base.py", "--config", "config_default.yaml"])

    if compute_ensemble_anoms:
        ens_dirs = my_exp.get_all_ensemble_dirs()
        os.chdir(join(cd, "merging"))
        for _dir in ens_dirs:
            for var in vars_to_track:
                pdb.set_trace()
                subprocess.run(["python", "cdo_compute_anomalies.py",
                               "--dataset", join(_dir, "atmos_6_hourly.nc"),
                               "--clim", join(my_exp.postproc, tracking_argdict[var]['ctrl_rawdat_fn']),
                               "--vars", tracking_argdict[var]['vartup']])
                print(f"Saved {var} anomaly field in directory {_dir}")
    
    trk_chunk_len = 80         # tracking chunk length, timesteps

#    if control_tracking:
    if False:
        total_ctrl_length = 120*12*9       # control run length, timesteps (TODO: make more flexible)
        n_chunks = int(np.ceil(total_ctrl_length/trk_chunk_len))
        ident = np.random.randint(99999999)

        # directory where the raw files are stored
        datadir = join(root, "postprocessed")
        # create track directory if it doesn't already exist
        trackdir = join(root, "tracks");  os.makedirs(trackdir, exist_ok=True)

        os.chdir(join(cd, "agcm_tracking_tools"))
        for var in vars_to_track:
            # arguments of tracking_single.sh
            # TODO: change to sbatch?
            pdb.set_trace()
            subprocess.run(["bash", "track_single.sh", datadir, 
                            tracking_argdict[var]['ctrl_rawdat_fn'], f"{ident:08d}", var,
                            tracking_argdict[var]['specfilt_fn'], tracking_argdict[var]['trackin_fn'],
                            str(n_chunks), trackdir])

    if ensemble_tracking:
        n_chunks = 2 # always
        mem_folders  = my_exp.get_all_ensemble_dirs()
        postproc_dir = join(my_exp.ctr_dir, 'postprocessed')
        for var in vars_to_track:
            for dir in mem_folders:
                # calculate the anomaly fields if they aren't already present 
                # in the ensemble member's data folder
                if not os.path.exists(join(dir, tracking_argdict[var]['pert_rawdat_fn'])):
                    subprocess.run(["python", "compute_anomalies.py", 
                                    "--dataset", join(dir, "atmos_6_hourly.nc"),
                                    "--clim", join(postproc_dir, tracking_argdict[var]['ctrl_climo_fn']),
                                    "--vars", tracking_argdict[var]['vartup']])
                
                # do the tracking on the anomaly fields
                os.chdir(join(cd, "agcm_tracking_tools"))
                ident = np.random.randint(99999999)
                subprocess.run(["bash", "track_single.sh", dir, 
                                tracking_argdict[var]['pert_rawdat_fn'], f"{ident:08d}", var,
                                tracking_argdict[var]['specfilt_fn'], tracking_argdict[var]['trackin_fn'],
                                str(n_chunks), dir])
    
    if parse_tracks:
        os.chdir(join(cd, "agcm_tracking_tools"))
        for var in vars_to_track:
            subprocess.run(["sbatch", "parse_tracks.sh", expname, 
                            str(int(control_tracking)), str(int(ensemble_tracking)),
                            var])
            # TODO: make track parsing more flexible & easily reconfigurable
