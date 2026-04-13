import os
import sys
import pdb

sys.path.append("..")
from track import *
from tools import extract_window
import observable_functions as of

class Ensemble:
    def __init__(self, directory, n_mems, t0):
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
        """
        self.path_to = directory
        self.N = n_mems
        self.mem_dirs = {i: os.path.join(directory , f'b{i:02d}') for i in range(1, n_mems+1)}
        try:
            self.data = {i: os.path.join(self.mem_dirs[i], 'atmos_6_hourly.nc') for i in range(1, n_mems+1)}
        except:
            self.data = None
        try: 
            self.data_anoms = {i: os.path.join(self.mem_dirs[i], 'pert_anoms.nc') for i in range(1, n_mems+1)}
        except:
            self.data_anoms = None
        self.t0 = t0

    def save_matched_tracks(self, ctds, t_offset, T=60, k=2, S=3):
        """
        Arguments
        ---------
        ctds: 4-member dictionary
            TODO: write description 
        t_offset: int or float
            Specifies how many days after the perturbation time we should look for matching tracks.
        """
        mdcs = dict()
        obs = ['MSLP', 'vor850']
        feats = ['cycs', 'acycs']
        # create matching track dictionaries for each feature type
        for var in obs:
            for obj in feats:
                obj_str = f'{var}_{obj}'
                ptds = []
                # loop over ensemble members to get track dictionaries
                for i in self.mem_dirs.keys():
                    try:
                        ptds.append( load_trackdict_from_file(
                            os.path.join(self.mem_dirs[i], var, f'{obj_str}.pickle'),
                            mem_id = i          # this tags each track with which ensemble member it came from, which is important for the composites!
                        ))
                    except:
                        print(f"tracks from {self.mem_dirs[i]} failed to load")
                mdc = match_tracks(ctds[obj_str], ptds, (self.t0+t_offset, self.t0+t_offset), T=T, k=k, S=S)
                mdcs[obj_str] = mdc
        
        self.matches = mdcs
        return
    
    def get_matched_composites(self, cds, obsf, feat_type, time_offset=0.0, lat_range=(30, 60), 
                               use_anom_fields=False, invert_SH=False, background_err=None):
        """
        For each of the control tracks in the dictionary `self.matches`, produces composite plots
        of the features in the control and perturbed ensembles, as well as of the RMSE in sea-level
        pressure and 850-hPa vorticity, at a specified time.

        Arguments
        ---------  
        """
        # Make sure we have a matched track dictionary to work with first
        assert self.matches != None, "Matching has not been performed yet"
        mdc = self.matches[feat_type]        # get the match dictionary corresponding to our desired feature type

        # get control data array
        cda = obsf(cds)
        if use_anom_fields:
            # cda = cda - cda.mean(dim='time')
            pass

        # load perturbed data arrays
        if use_anom_fields:
            pda_dict = {i: xr.open_dataset(self.data_anoms[i]) for i in range(1, self.N+1)}
        else:
            pda_dict = {i: xr.open_dataset(self.data[i]) for i in range(1, self.N+1)}
        for i in pda_dict.keys():
            pda = obsf(pda_dict[i])
            pda_dict[i] = pda

        n_offset = int(4*time_offset)
        
        pda_comp = 0; n_feat = 0; err_comp = 0

        for ctr in mdc.keys():
            ctr_hash = hash(ctr)            # unique numerical identifier for track

            # get coordinates corresponding to peak intensity
            ct, clon, clat = ctr.get_peak_coords(convert_date=False, n_offset=n_offset)
            if not np.isnan(ct):
                ct_date = cftime.num2date(ct, "days since 0001-01-01", calendar='360_day')

                if np.abs(clat) <= lat_range[-1] and np.abs(clat) >= lat_range[0]:
                    cda_window = extract_window(cda, clon, clat, ct_date)
                    if clat < 0:     # invert axes if in SH due to reflective symmetry about equator
                        cda_window.data = np.flip(cda_window.data, axis=0)
                        if invert_SH:   
                            cda_window = -cda_window    # flip the sign of the vorticity anomalies

                    # now compare to the perturbed track fields

                    if background_err != None:
                        # TODO: think carefully about this. Since the control and perturbed features are in different places,
                        # which background error field should we use? Options that I can think of which aren't too unreasonable
                        # are the error field for the control cyclone (since on average the perturbed features should be randomly
                        # distributed around the control) or an error field at an "average location". 
                        # I'll do the control for now.
                        # 
                        # Also, what happens if we get "negative errors"?
                        err_bkgd = obsf(background_err)
                        tnorm_fl = ct - self.t0
                        # if the peak intensity lies outside the range of our error background array, skip to the next track
                        if tnorm_fl > err_bkgd.time[-1]:
                            continue
                        #err_bkgd_window = extract_window(err_bkgd, clon, clat, tnorm_fl)
                        #if clat < 0:
                        #    err_bkgd_window.data = np.flip(err_bkgd_window.data, axis=0)
                        # pdb.set_trace()
                    
                    ptr_list = mdc[ctr]         # get list of perturbed tracks
                    for ptr in ptr_list:
                        # proceed only if track exists at the time used above
                        if len(np.where(ptr.time == ct)[0]) == 0:
                            pass
                        else:
                            # get coordinates and extract window, but check if track is within range
                            i_t = np.where(ptr.time == ct)[0]
                            plon, plat = ptr.lon[i_t], ptr.lat[i_t]
                            if np.abs(plat) <= lat_range[-1] and np.abs(plat) >= lat_range[0]:
                                # load the data!
                                i_mem = ptr.mem_id
                                pda = pda_dict[i_mem]
                                pda_window = extract_window(pda, plon, plat, ct_date)

                                if background_err != None:
                                    err_bkgd_window = extract_window(err_bkgd, plon, plat, tnorm_fl)
                                    if clat < 0:
                                        err_bkgd_window.data = np.flip(err_bkgd_window.data, axis=0)
                                if plat < 0:     # invert axes if in SH due to reflective symmetry about equator
                                    pda_window.data = np.flip(pda_window.data, axis=0)
                                    if invert_SH:   
                                        pda_window = -pda_window    # flip the sign of the vorticity anomalies
                                if type(pda_comp) != int:
                                    pda_comp.data = pda_comp.data + pda_window.data
                                    err = np.abs(cda_window.data - pda_window.data)
                                    if background_err != None:
                                        # pdb.set_trace()
                                        err = err - err_bkgd_window.data
                                    err_comp.data = err_comp.data + err
                                else:  
                                    pda_comp = pda_comp + pda_window
                                    # instantiate err_comp as DataArray object
                                    err = cda_window.copy()
                                    err.data = np.abs(cda_window.data - pda_window.data)
                                    if background_err != None:
                                        # pdb.set_trace()
                                        err.data = err.data - err_bkgd_window.data
                                    err_comp = err_comp + err
                                n_feat += 1
            print(f"Added track {ctr_hash} to composite")
        if n_feat > 0:
            pda_comp = pda_comp/n_feat
            err_comp = err_comp/n_feat
            
        return pda_comp, err_comp, n_feat

def get_multi_matched_composites(ens_dict, cds, obsf, feat_type, time_offset=0.0, lat_range=(30, 60), 
                                 use_anom_fields=False, invert_SH=False, background_err=None):
    """
    """
    pda_comp = 0; n_feat = 0; err_comp = 0
    mdc_dict = dict()
    for t0, ens in ens_dict.items():
        assert ens.matches != None, "Matching has not been performed yet"
        pda, err, n  = ens.get_matched_composites(cds, obsf, feat_type, time_offset, lat_range, use_anom_fields, invert_SH, background_err)
        if type(pda_comp) != int and type(pda) != int:
            # weight each term by number of features in the composite
            assert type(pda) == type(err), f"Mismatch between type {type(pda)} and type {type(err)}"
            pda_comp.data = pda_comp.data + (pda.data)*n
            err_comp.data = err_comp.data + (err.data)*n
            n_feat += n
        elif type(pda_comp) != int and type(pda) == int:
            assert type(pda) == type(err), f"Mismatch between type {type(pda)} and type {type(err)}"          
            pass 
        else:  
            pda_comp = pda_comp + pda*n
            # instantiate err_comp as DataArray object
            err_comp = err_comp + err*n
            n_feat += n


    if n_feat > 0:
        pda_comp = pda_comp/n_feat
        err_comp = err_comp/n_feat
    return pda_comp, err_comp, n_feat

def get_multi_matched_composite_time_series(ens_dict, cds, obsf, feat_type, time_range, lat_range=(30, 60), 
                                            use_anom_fields=False, invert_SH=False, background_err=None):
    """ """
    times = time_range
    p_data = np.empty(time_range.shape, dtype='object')
    e_data = np.empty(time_range.shape, dtype='object')
    n = np.empty(time_range.shape, dtype='int')
    for jt, t in enumerate(times):
        da, eda, nt = get_multi_matched_composites(ens_dict, cds, obsf, feat_type, time_offset=t, lat_range=(30, 60), 
                                 use_anom_fields=use_anom_fields, invert_SH=invert_SH, background_err=background_err)
        p_data[jt] = da
        e_data[jt] = eda
        n[jt] = nt
    feat_comp = CompTimeSeries(times, p_data, n)
    err_comp = CompTimeSeries(times, e_data, n)
    return feat_comp, err_comp

if __name__ == "__main__":
    print("Got into Main")
    print(" *********** ")

    from os.path import join
    expname = sys.argv[2]
    exproot = join("/orcd/data/talia_tb/001/aqua_gcm_runs", expname)
    obs = ['MSLP', 'vor850']
    feats = ['cycs', 'acycs']
    # create matching track dictionaries for each feature type
    ctds = dict()
    for var in obs:
        for obj in feats:
            obj_str = f'{var}_{obj}'
            td = load_trackdict_from_file(join(exproot, 'tracks', var, f"{obj_str}.pickle"))
            ctds[obj_str] = td

    times = np.arange(365.5, 720.0, 10.0)      # perturbation times
    ensemble_dict = dict()
    for t0 in times:
        edir = join(exproot, 'ensembles', str(t0), 'spec_mag_0.02')
        ens = Ensemble(edir, 10, t0)
        ens.save_matched_tracks(ctds, t_offset=9.0)
        ensemble_dict[t0] = ens

    print("Ensemble dictionaries loaded")

    # load dataset and tracks
    ds = xr.open_dataset(join(exproot, "postprocessed", "base_anoms.nc"))
    ds2 = xr.open_dataset(join(exproot, "postprocessed", "base_thermo_ll.nc"))

    #ds = xr.open_dataset(join(exproot, "postprocessed", "base_anoms.nc"))
    #ds2 = xr.open_dataset(join(exproot, "postprocessed", "base_thermo_ll.nc"))


    background_err_fns = [join(exproot, "other_data", "ps_abs_error_time_series.nc"), 
                          join(exproot, "other_data", "vor_abs_error_time_series.nc"),
                          join(exproot, "other_data", "T850_abs_error_time_series.nc")]
    background_errs = [xr.open_dataset(fn) for fn in background_err_fns]


    #ofuncs = [of.surface_pressure, of.vorticity_850, of.temperature_850, of.q_surf]
    #cdata = [ds, ds, ds2]
    #savestrs = ["ps_lag4", "vor850_lag4", "T850_lag4"]
    #invert_bool=[False, True, False]
    #anom_bool=[True, True, False]
    #for i in range(3):
    #    composite = get_multi_matched_composites(ensemble_dict, cdata[i], ofuncs[i], 'vor850_acycs', time_offset=-1.0, use_anom_fields=anom_bool[i], invert_SH=invert_bool[i])
    #    pickle.dump(composite, open(f"data/{savestrs[i]}_ensemble_composite_default_control_va.pkl", "wb"))
    #
    #ofuncs = [of.surface_pressure, of.vorticity_850, of.temperature_850, of.q_surf]
    #cdata = [ds, ds, ds2]
    ## savestrs = ["ps", "vor850", "T850"]
    #invert_bool=[False, True, False]
    #anom_bool=[True, True, False]
    #for i in range(3):
    #    composite = get_multi_matched_composites(ensemble_dict, cdata[i], ofuncs[i], 'vor850_cycs', time_offset=-1.0, use_anom_fields=anom_bool[i], invert_SH=invert_bool[i])
    #    pickle.dump(composite, open(f"data/{savestrs[i]}_ensemble_composite_default_control_vc.pkl", "wb"))

    i = int(sys.argv[1]) - 1

    tlags = np.arange(-2.0, 1.25, 0.25)
    ofuncs = [of.surface_pressure, of.vorticity_850, of.temperature_850]
    cdata = [ds, ds, ds2]
    savestrs = ["ps", "vor850", "T850"]
    invert_bool=[False, True, False]
    anom_bool=[True, True, False]
    tr_type = ['vor850_cycs', 'vor850_cycs', 'vor850_cycs', 'vor850_acycs', 'vor850_acycs', 'vor850_acycs']
    final_str = ['vc', 'vc', 'vc', 'va', 'va', 'va']
    composite = get_multi_matched_composite_time_series(ensemble_dict, cdata[i % 3], ofuncs[i % 3], tr_type[i], time_range=tlags, 
                                                        use_anom_fields=anom_bool[i % 3], invert_SH=invert_bool[i % 3], 
                                                        background_err=background_errs[i % 3])
    print(f"Composite of {tr_type[i]} {savestrs[i % 3]} complete")
    pickle.dump(composite, open(f"data/{expname}_{savestrs[i % 3]}_ensemble_composite_lessbkgd_{final_str[i]}.pickle", "wb"))