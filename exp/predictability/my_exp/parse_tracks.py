import numpy as np
from os.path import basename, join
import pickle
import subprocess
import sys

from isca import GFDL_DATA
from shared_vars import expname, GFDL_STORAGE

sys.path.append('/home/cgliddon/Isca/postprocessing/agcm_tracking_tools')
from parse_tracks_kh import *

if __name__ == '__main__':
    parse_control  = bool(int(sys.argv[1]))
    parse_ensemble = bool(int(sys.argv[2]))

    if parse_control:
        print("Parsing control tracks...")
        t0_mem = 360.25
        control_dir = join(GFDL_STORAGE, expname, 'tracks')
        for var in ['MSLP', 'vor850']:
            fps = [join(control_dir, var, bn) for bn in ['NH_TRACKS/ff_trs_neg', 'NH_TRACKS/ff_trs_pos', 
                                                    'SH_TRACKS/ff_trs_neg', 'SH_TRACKS/ff_trs_pos']]
            tds = [get_trackdict_from_file(fp) for fp in fps]
            tds_new = [convert_time_to_phys(td, t0_mem) for td in tds]
            if var == 'MSLP':
                cyc_tracks = merge_tracks([tds_new[0], tds_new[2]])
                acyc_tracks = merge_tracks([tds_new[1], tds_new[3]])
            elif var == 'vor850':
                cyc_tracks = merge_tracks([tds_new[1], tds_new[2]])
                acyc_tracks = merge_tracks([tds_new[0], tds_new[3]])
            out_name_c = join(control_dir, var, f'{var}_cycs.pickle')
            out_name_a = join(control_dir, var, f'{var}_acycs.pickle')
            print(f"Dumping data in {control_dir} :")
            pickle.dump(cyc_tracks, open(out_name_c, 'wb'))
            pickle.dump(acyc_tracks, open(out_name_a, 'wb'))

    if parse_ensemble:
        print("Parsing ensemble tracks...")
        complete = False
        while complete == False:
            patch_list = []     # a list of start time/ensemble member pairs with unsuccessful tracking
            complete = True

            for time in np.arange(360.0, 1440.0+10.0, 10.0):     # must be consistent with exp_workflow.sh
                # t0_mem = ((time - 0.25) // 30)*30 + 0.25       # uncomment this line if not using trim_data in specpert
                t0_mem = time                                    # comment this line if not using trim_data in specpert
                # dataroot = f'/orcd/data/talia_tb/001/aqua_gcm_runs/frierson_moist/ensembles/{time}/spec_mag_0.02'
                pert_str = 'spec_mag_0.02'
                for mem_id in list(range(1, 11)):
                    memdir = join(GFDL_STORAGE, expname, 'ensembles', str(time), pert_str, f'b{mem_id:02}')
                    for var in ['MSLP', 'vor850']:
                        try:
                            fps = [join(memdir, var, bn) for bn in ['NH_TRACKS/ff_trs_neg', 'NH_TRACKS/ff_trs_pos', 
                                                                    'SH_TRACKS/ff_trs_neg', 'SH_TRACKS/ff_trs_pos']]
                            tds = [get_trackdict_from_file(fp) for fp in fps]
                            tds_new = [convert_time_to_phys(td, t0_mem) for td in tds]
                            if var == 'MSLP':
                                cyc_tracks = merge_tracks([tds_new[0], tds_new[2]])
                                acyc_tracks = merge_tracks([tds_new[1], tds_new[3]])
                            elif var == 'vor850':
                                cyc_tracks = merge_tracks([tds_new[1], tds_new[2]])
                                acyc_tracks = merge_tracks([tds_new[0], tds_new[3]])
                            out_name_c = join(memdir, var, f'{var}_cycs.pickle')
                            out_name_a = join(memdir, var, f'{var}_acycs.pickle')
                            print(f"Dumping data in {memdir} :")
                            pickle.dump(cyc_tracks, open(out_name_c, 'wb'))
                            pickle.dump(acyc_tracks, open(out_name_a, 'wb'))
                        except:
                            print(f"Error in parsing tracks in directory {join(memdir, var)}")
                            patch_list.append((time, mem_id))
                            complete = False

            for (t_start, mem_id) in patch_list:

                memdir = join(GFDL_STORAGE, expname, 'ensembles', str(t_start), pert_str, f'b{mem_id:02}')
                # run_len = int(20 + np.ceil(t_start % 30))    # must agree with the run length in specpert.py
                run_len = 30.25
                num_tracking_chunks = int(np.ceil(run_len*4 / 80)) 
                subprocess.run(["/home/cgliddon/Isca/postprocessing/agcm_tracking_tools/patch_tracking.bash", 
                                join(GFDL_STORAGE, expname, memdir), str(num_tracking_chunks), 
                                join(GFDL_STORAGE, expname, "postprocessed", "base_t_mean.nc")])        # mean file
                print(f"Patched tracks for {(t_start, mem_id)}")

            print("Redoing loop...")