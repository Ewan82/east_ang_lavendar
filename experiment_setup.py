#######################################
# File to specify setup of experiment #
#######################################
# core py modules:
import os
# local modules:
import observations
import plot


# set directory for JULES output
#output_directory = os.getcwd()+'/output'
output_directory = '/work/scratch/ewanp82/east_ang_lavendar/output'
# set directory containing JULES NML files
nml_directory = os.getcwd()+'/chess_da_ea_ens'  # +'/chess_da_cardt_smappixel'
# set model executable
model_exe = '/home/users/ewanp82/models/jules5.3/build/bin/chess_emma.exe'
# set function to extract JULES modelled observations for ensemble JULES
jules_hxb = observations.extract_jules_hxb
jules_hxi = observations.extract_jules_hx
# set function to extract ensemble ensemble of modelled observations
jules_hxb_ens = observations.extract_jules_hxb_ens
# set function to extract observations to be assimilated
obs_fn = observations.extract_twin_data
# set JULES parameters to optimised during data assimilation
"""
opt_params = {'b': [15.70, (0., 32.)],
              'c': [0.3, (0.0, 1.5)],
              'e': [0.037, (0.0, 0.15)],
              'f': [0.142, (0.0, 1.0)],
              'h': [0.63, (0.0, 1.5)],
              'i': [1.58, (0.0, 3.2)],
              'k': [0.64, (0.0, 1.5)],
              'l': [1.26, (0.0, 3.0)],
              }
"""
opt_params = {'a': [0.63052, (0.05, 4.0)],
              'c': [1.16518, (0.1, 8.0)],
              'e': [0.25929, (0.01, 2.0)],
              'g': [0.40220, (0.02, 3.0)],
              }

# set error on ensemble parameter estimates
prior_err = 0.2
# set size of ensemble to be used in data assimilation experiments
ensemble_size = 50
# set number of processors to use in parallel runs of JULES ensemble
num_processes = 50
# set seed value for any random number generation within experiments
seed_value = 0
# plotting save function
save_plots = plot.save_plots
# plotting output director
plot_output_dir = os.getcwd()+'/output/plot'

