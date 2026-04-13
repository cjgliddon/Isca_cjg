import glob
import numpy as np
import os
from os.path import join

from track import load_trackdict_from_file, filter_tracks
from tools import compute_geodesic_distance

import pdb

class Ensemble:
    def __init__(self, directory, n_mems, t0, add_pv=False):
        """
        Object containing information about an ensemble of perturbed GCM runs.

        Arguments
        ---------
        directory : str
            Full path to a directory in which all the ensemble members are contained.
        n_mems : int
            Number of ensemble members
        t0 : float
            Perturbation time of the ensemble
        add_pv : boolean, optional
            If True, also loads the paths to the pv data files for the ensemble members as instance
            attributes. False by default.
        """
        self.path_to = directory
        self.N = n_mems
        self.mem_dirs = {i: os.path.join(directory , f'b{i:02d}') for i in range(1, n_mems+1)}
        self.data = dict()
        for i in range(1, n_mems+1):
            try:
                self.data.update({i: os.path.join(self.mem_dirs[i], 'atmos_6_hourly.nc')})
            except:
                self.data.update({i: None})
                print(f"Ensemble member {i} failed to load")

        self.t0 = t0

class EnsembleSet:
    def __init__(self, directory, n_mems, t0_list=None, subdir_string='spec_mag_0.02', add_pv=False):
        """
        Object containing information about a set of ensembles of perturbed GCM runs
        (initialized at different times).

        Arguments
        ---------
        directory : str
            Full path to a directory in which the experiment is contained.
        n_mems : int
            Number of members per ensemble
        t0_list : array-like, optional
            List of ensemble initialization times. If "None", the initialization times are 
            extracted automatically from the ensemble subdirectories present in `directory`.
        subdir_string : str, optional
            The subdirectory in which the ensemble members are contained within each ensemble 
            subdirectory. (Default: `'spec_mag_0.02'`)
        """
        self.expdir = directory
        self.N_mems = n_mems
        if t0_list is not None:
            self.start_times = t0_list
        else:
            self.start_times = np.asarray(
                [float(os.path.basename(t0)) for t0 in glob.glob(os.path.join(directory, 'ensembles', '*'))]
            )
        self.start_times.sort()     # sort start times ascending
        self.ensemble_dict = dict()
        for t0 in self.start_times:
            e = Ensemble(
                directory=os.path.join(directory, 'ensembles', str(t0), subdir_string), 
                n_mems=n_mems, 
                t0=t0)
            self.ensemble_dict.update({t0: e})
        
        self.N_ens = len(self.start_times)

    def get_all_track_dicts(self, feature_type):
        """
        Loads and returns all track dictionaries from control and perturbed tracks of a specified `feature_type`.
        """
        track_var = feature_type[:feature_type.find("_")]       # get tracking variable
        basename  = f"{feature_type}.pickle"
        # control tracks
        tdc = load_trackdict_from_file(join(self.expdir, 'tracks', track_var, basename))
        # perturbed tracks
        tdp = [load_trackdict_from_file(join(self.expdir, memdir, track_var, basename), mem_id=i_mem)
               for e in self.ensemble_dict.values()
               for i_mem, memdir in e.mem_dirs.items()
               ]
        return tdc, tdp
    
    def get_track_dicts_from_ensemble(self, feature_type, t0):
        """
        Loads and returns all track dictionaries from control and perturbed tracks of a specified `feature_type`
        and ensemble start time.
        """
        track_var = feature_type[:feature_type.find("_")]       # get tracking variable
        basename  = f"{feature_type}.pickle"
        tdc = load_trackdict_from_file(join(self.expdir, 'tracks', track_var, basename))
        # perturbed tracks
        tdp = [load_trackdict_from_file(join(self.expdir, memdir, track_var, basename), mem_id=i_mem)
               for i_mem, memdir in self.ensemble_dict[t0].mem_dirs.items()
               ]
        return tdc, tdp

    def match_tracks(self, feature_type, t0_range, ensemble_start=None, T=60, k=2, S=3, filtering=None):
        """ 
        This function accepts two track dictionaries, one representing a "control" set and the other a
        "perturbed" set. It applies the matching method described in Froude et al. (Mon. Wea. Rev.,
        Feb. 2007) to match tracks in the control set with those in the perturbed set. It returns a new
        dictionary whose keys are the tracks in the control set and whose values are lists of tracks in the
        perturbed set which match the control-set track.

        :param str feature_type:
            String specifying the type of the feature being tracked. 
        :param array-like t0_range:
            A 2-element list, tuple, or array giving the range of allowed *starting times* for tracks
            that are to be matched.
        :param int or float, optional T:

        :param int or float, optional k:

        :param int or float, optional S:
        """
        md = dict()     # initialize match dictionary

        if ensemble_start == None:
            tdc, tdp = self.get_all_track_dicts(feature_type)
        else:
            tdc, tdp = self.get_track_dicts_from_ensemble(feature_type, ensemble_start)

        if filtering != None:
            tdc = filter_tracks(tdc, filter_method=filtering)
            tdp = [filter_tracks(td, filter_method=filtering) for td in tdp]

        for key_c in tdc.keys():
            tc = tdc[key_c]
            time_c, lon_c, lat_c = tc.time, tc.lon, tc.lat
            # check that track start is in the appropriate range
            if (time_c[0] >= t0_range[0]) and (time_c[0] <= t0_range[1]):
                # loop over perturbed track dictionaries
                for td in tdp:
                    for key_p in td.keys():
                        tp = td[key_p]
                        time_p, lon_p, lat_p = tp.time, tp.lon, tp.lat

                        # determine how many points overlap in time
                        common_times, it_overl_c, it_overl_p = np.intersect1d(time_c, time_p, return_indices=True)
                        it_overl = list(zip(it_overl_c, it_overl_p))

                        # check if time overlap criterion is satisfied
                        if 100*(2*len(it_overl)/(len(time_c) + len(time_p))) > T:
                            s_array = compute_geodesic_distance(
                                (lon_c[it_overl_c], lat_c[it_overl_c]),
                                (lon_p[it_overl_p], lat_p[it_overl_p])
                            )
                            s_to_compare = np.array(s_array[:k])
                            # print(s_to_compare)
                            # check if "geodesic closeness" criterion is satisfied
                            if len(np.where(s_to_compare > S)[0]) > 0:
                                pass
                            else:
                                if md.get(tc) == None:      # add track as key to dictionary if it isn't there already
                                    md[tc] = [tp]
                                    key_to_append = key_p
                                else:
                                    md[tc].append(tp)
                # print(f"Track {hash(tc)} matched")
        return md