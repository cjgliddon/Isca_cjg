# This document contains all parameters which are shared between control and perturbed model runs.
import os

# GFDL_STORAGE = '/orcd/data/talia_tb/001/aqua_gcm_runs'
GFDL_STORAGE = os.environ["GFDL_STORAGE"]


expname = 'my_exp'          # your experiment name can also include subdirectories, e.g. "my_exp/4xCO2"
from isca import DiagTable, Namelist

my_diag = DiagTable()
my_diag.add_file('atmos_6_hourly', 6, 'hours', time_units='hours')

# Tell model which diagnostics to write
my_diag.add_field('dynamics', 'ps', time_avg=False)
my_diag.add_field('dynamics', 'bk')
my_diag.add_field('dynamics', 'pk')
my_diag.add_field('dynamics', 'sphum', time_avg=False)
my_diag.add_field('dynamics', 'ucomp', time_avg=False)
my_diag.add_field('dynamics', 'vcomp', time_avg=False)
my_diag.add_field('dynamics', 'temp', time_avg=False)
my_diag.add_field('dynamics', 'vor', time_avg=False)
my_diag.add_field('dynamics', 'div', time_avg=False)
my_diag.add_field('dynamics', 'omega', time_avg=False)
my_diag.add_field('dynamics', 'height', time_avg=False)


# Define values for the 'core' namelist
# You can make any changes to parameters here
my_namelist = Namelist({
    # Run length & calendar info
    'main_nml':{
     'days'   : 30,
     'hours'  : 0,
     'minutes': 0,
     'seconds': 0,
     'dt_atmos':300,
     'current_date' : [1,1,1,0,0,0],
     'calendar' : 'thirty_day'
    },

    # Moist physics parameterization
    'idealized_moist_phys_nml': {
        'do_damping': True,
        'turb':True,
        'mixed_layer_bc':True,
        'do_virtual' :False,
        'do_simple': True,
        'roughness_mom':3.21e-05,
        'roughness_heat':3.21e-05,
        'roughness_moist':3.21e-05,                
        'two_stream_gray': True,     #Use grey radiation
        'convection_scheme': 'SIMPLE_BETTS_MILLER', #Use the simple Betts Miller convection scheme from Frierson
    },

    'vert_turb_driver_nml': {
        'do_mellor_yamada': False,     # default: True
        'do_diffusivity': True,        # default: False
        'do_simple': True,             # default: False
        'constant_gust': 0.0,          # default: 1.0
        'use_tau': False
    },
    
    'diffusivity_nml': {
        'do_entrain':False,
        'do_simple': True,
    },

    'surface_flux_nml': {
        'use_virtual_temp': False,
        'do_simple': True,
        'old_dtaudv': True    
    },

    'atmosphere_nml': {
        'idealized_moist_model': True
    },

    #Use a large mixed-layer depth, and the Albedo of the CTRL case in Jucker & Gerber, 2017
    'mixed_layer_nml': {
        'tconst' : 285.,
        'prescribe_initial_dist':True,
        'evaporation':True,   
        'depth': 2.5,                          #Depth of mixed layer used
        'albedo_value': 0.31,                  #Albedo value used             
    },

    'qe_moist_convection_nml': {
        'rhbm':0.7,
        'Tmin':160.,
        'Tmax':350.   
    },

    'betts_miller_nml': {
       'rhbm': .7   , 
       'do_simp': False, 
       'do_shallower': True, 
    },
    
    'lscale_cond_nml': {
        'do_simple':True,
        'do_evap':True
    },
    
    'sat_vapor_pres_nml': {
        'do_simple':True                # Clausius-Clapeyron scaling
    },
    
    'damping_driver_nml': {
        'do_rayleigh': True,
        'trayfric': -0.25,                  # neg. value: time in *days*
        'sponge_pbottom':  5000.,           # Bottom of the model's sponge down to 50hPa (units are Pa)
        'do_conserve_energy': True,             
    },

    'two_stream_gray_rad_nml': {
        'rad_scheme': 'frierson',            # Select radiation scheme to use, which in this case is Frierson
        'do_seasonal': False,                # do_seasonal=false uses the p2 insolation profile from Frierson 2006. do_seasonal=True uses the GFDL astronomy module to calculate seasonally-varying insolation.
        'atm_abs': 0.2,                      # default: 0.0        
    },

    # FMS Framework configuration
    'diag_manager_nml': {
        'mix_snapshot_average_fields': False  # time avg fields are labelled with time in middle of window
    },

    'fms_nml': {
        'domains_stack_size': 600000                        # default: 0
    },

    'fms_io_nml': {
        'threading_write': 'single',                         # default: multi
        'fileset_write': 'single',                           # default: multi
    },

    'spectral_dynamics_nml': {
        'damping_order': 4,             
        'water_correction_limit': 200.e2,
        'reference_sea_level_press':1.0e5,
        'num_levels':30,               # How many model pressure levels to use
        'valid_range_t':[100.,800.],
        'initial_sphum':[2.e-6],
        'vert_coord_option':'uneven_sigma', #automatically calculates the sigma levels using a subroutine in vert_coordinate.F90
        'surf_res':0.5,
        'scale_heights' : 4.0,
        'exponent':2.5,
        'robert_coeff':0.03
    },
})
