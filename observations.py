# 3rd party modules:
import numpy as np
import netCDF4 as nc
import random
import datetime as dt
from contextlib import contextmanager
import pickle
import multiprocessing as mp
# local modules:
import experiment_setup as es


@contextmanager
def poolcontext(*args, **kwargs):
    """
    Function to control the parallel run of other functions
    :param args:
    :param kwargs:
    :return:
    """
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def find_nearest_idx_tol(array, value, tol=dt.timedelta(days=1.)):
    """
    Find nearest value in an array for a given tolerance
    :param array: array of values (arr)
    :param value: value for which to find nearest element (float, obj)
    :param tol: distance tolerance (float, obj)
    :return: nearest value in array, index of nearest value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if abs(array[idx] - value) <= tol:
        ret_val = idx
    else:
        ret_val = np.nan
    return ret_val


def extract_twin_data(obs='/group_workspaces/jasmin2/hydro_jules/data/epinnington/smap_9km_uk/smap_9km_2016.nc'):
    """
    Function for extracting observations to be assimilated in data assimilation experiments. Here we are running a twin
    experiment using a "model truth" to draw observations from.
    :param mod_truth: location of netCDF file containing "model truth" output (str)
    :param seed_val: seed value for adding random noise to the observations (int)
    :return: dictionary containing observations and observation errors (dictionary)
    """
    # open obs netCDF file
    obs = 'data/smap_ea_9km_regrid.nc'
    smap = nc.Dataset(obs, 'r')
    # set position of observations
    smap_sm = smap.variables['Soil_Moisture_Retrieval_Data_AM_soil_moisture'][1:,:,:]
    #smap_times = nc.num2date(smap_time_var[:,64,69], 'seconds since 2000-01-01 20:00:00')
    smap_times = nc.num2date(smap.variables['time'][1:], smap.variables['time'].units)
    smap_flag = smap.variables['Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag'][1:,:,:]
    idxs = np.where((smap_flag == 0.0) & (smap_sm.data != -9999.0))
    t_unique = np.unique(idxs[0])
    t_idx = [np.where(idxs[0] == x)[0] for x in t_unique]
    #print(t_idx[0])
    sm_obs = smap_sm[idxs[0], idxs[1], idxs[2]]
    #sm_obs_t = [smap_sm[idxs[0][x], idxs[1][x], idxs[2][x]] for x in t_idx]
    sm_obs_t = [sm_obs[x] for x in t_idx]
    #sm_mean = [np.mean(smap_sm[idxs[0][x], idxs[1][x], idxs[2][x]]) for x in t_idx]
    #sm_err = np.mean(sm_obs[sm_obs > 0]) * 0.15
    #print(sm_err, np.mean(sm_obs[sm_obs > 0]))
    #sm_rmats = [np.eye(len(x)) * sm_err**2 for x in sm_obs_t]
    sm_rmats = [np.eye(len(x)) * (0.15 * x)**2 for x in sm_obs_t]
    #sm_obs = smap_sm[idxs[0], idxs[1], idxs[2]]
    #sm_err = np.ones(len(sm_obs)) * np.mean(sm_obs[sm_obs > 0]) * 0.15  # 10% error in mod obs

    #sm_obs_t = []
    #sm_rmat_t = []

    """
    dat = nc.Dataset('/work/scratch/ewanp82/jules5.3/chess_da_ea_vg_default/output_folder/test.daily.nc', 'r')
    times = nc.num2date(dat.variables['time'][:], dat.variables['time'].units)
    smap_flag = smap.variables['Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag'][1:,:,:]
    smap_sm = smap.variables['Soil_Moisture_Retrieval_Data_AM_soil_moisture'][1:,:,:]
    idxs = np.where((smap_flag == 0.0) & (smap_sm.data != -9999.0))
    t_idx = np.array([np.where(times == t)[0][0] for t in smap_times[:]])
    jules_sm = dat.variables['smcl'][t_idx, 0, 0, :]
    jj_idx = pickle.load(open('hx_jules_idx.p', 'r'))
    hx_sm = [np.mean(jules_sm[idxs[0][x],jj_idx[x]]) for x in range(len(idxs[0]))]
    mod_sm = np.ones_like(smap_sm)*np.nan
    for x in range(len(idxs[0])):
        ju_idx = np.where((dat.variables['latitude'][0, :] > smap.variables['lat'][idxs[1][x]] - 0.05749795663238899) &
                          (dat.variables['latitude'][0, :] < smap.variables['lat'][idxs[1][x]] + 0.05749795663238899) &
                          (dat.variables['longitude'][0, :] > smap.variables['lon'][idxs[2][x]] - 0.05749795663238899) &
                          (dat.variables['longitude'][0, :] < smap.variables['lon'][idxs[2][x]] + 0.05749795663238899))[0]
        mod_sm[idxs[0][x],idxs[1][x],idxs[2][x]]=np.mean(jules_sm[idxs[0][x],ju_idx])
    latlon_idx = np.where((dat.variables['latitude'][0, :] > smap.variables['lat'][y] - 0.05749795663238899) &
                       (dat.variables['latitude'][0, :] < smap.variables['lat'][y] + 0.05749795663238899) &
                       (dat.variables['longitude'][0, :] > smap.variables['lon'][x] - 0.05749795663238899) &
                       (dat.variables['longitude'][0, :] < smap.variables['lon'][x] + 0.05749795663238899))[0]
    jules_sm = np.mean(dat.variables['smcl'][t_idx, 0, 0, latlon_idx], axis=1) / 100.

    smap_times = nc.num2date(smap_time_var[:, 60, 88], 'seconds since 2000-01-01 20:00:00')
    smap_none_idx = np.where(smap_times != None)[0]
    smap_times = smap_times[smap_none_idx]
    smap_sm = smap_sm[smap_none_idx]
    sm_obs = smap_sm[smap_sm.nonzero()]
    smap_times = smap_times[smap_sm.nonzero()]

    sm_err = np.ones(len(sm_obs)) * np.mean(sm_obs[sm_obs > 0]) * 0.15  # 10% error in mod obs
    """
    # close netCDF file
    smap.close()
    return {'obs': sm_obs,
            #'obs_err': sm_err,
            'sm_obs': sm_obs,
            #'sm_err': sm_err,
            'sm_times': smap_times,
            'sm_obs_t': sm_obs_t, 'sm_rmats': sm_rmats}


def extract_jules_hx(nc_file):
    """
    Function extracting the modelled observations from JULES netCDF files
    :param nc_file: netCDF file to extract observations from (str)
    :return: dictionary containing observations (dict)
    """
    if type(nc_file) == tuple:
        ens_no = nc_file[0]
        nc_file = nc_file[1]
        print(ens_no, nc_file)
    else:
        ens_no = None
    nc_dat = nc.Dataset(nc_file, 'r')
    times = nc.num2date(nc_dat.variables['time'][:], nc_dat.variables['time'].units)
    #obs = '/group_workspaces/jasmin2/hydro_jules/data/epinnington/smap_9km_uk/smap_9km_2016.nc'
    obs = 'data/smap_ea_9km_regrid.nc'
    smap = nc.Dataset(obs, 'r')
    smap_times = nc.num2date(smap.variables['time'][1:], smap.variables['time'].units)
    smap_flag = smap.variables['Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag'][1:, :, :]
    smap_sm = smap.variables['Soil_Moisture_Retrieval_Data_AM_soil_moisture'][1:, :, :]
    idxs = np.where((smap_flag == 0.0) & (smap_sm.data != -9999.0))
    t_unique = np.unique(idxs[0])
    smap_t_idx = [np.where(idxs[0] == x)[0] for x in t_unique]
    t_idx = np.array([np.where(times == t)[0][0] for t in smap_times[:]])
    jules_sm = nc_dat.variables['smcl'][t_idx, 0, 0, :]
    jj_idx = pickle.load(open('hx_jules_idx.p', 'rb'))
    sm_hx = np.array([np.mean(jules_sm[idxs[0][x], jj_idx[x]])/100. for x in range(len(idxs[0]))])
    if ens_no is not None:
        print(ens_no, nc_file)
        sm_hx_t = (ens_no, [sm_hx[x] for x in smap_t_idx])
    else:
        sm_hx_t = [sm_hx[x] for x in smap_t_idx]
    #sm_hx_tmean = np.array([np.mean(sm_hx[x]) for x in smap_t_idx])
    # close netCDF file
    nc_dat.close()
    smap.close()
    return sm_hx_t


def extract_jules_hxb():
    """
    Function to extract modelled observations for ensemble JULES run
    :return: dictionary containing observations (dict)
    """
    #return extract_jules_hx(es.output_directory+'/background/xb0.daily.nc')
    #return extract_jules_hx('/work/scratch/ewanp82/jules5.3/chess_da_cardt_smappixel/output/test.daily.nc')
    #return extract_jules_hx('/home/users/ewanp82/projects/smap_pt_lav/output/background/test.daily.nc')
    return extract_jules_hx('/work/scratch/ewanp82/east_ang_lavendar/output/background/prior.daily.nc')


def extract_jules_hxb_ens():
    """
    Function to extract ensemble of modelled observations from ensemble model ensemble
    :return: ensemble of modelled observations (lst)
    """
    print('Running ensemble')
    f_names = [(x, es.output_directory + '/ensemble/ens' + str(x) + '.daily.nc') for x in range(es.ensemble_size)]
    mp.freeze_support()
    with poolcontext(processes=es.num_processes) as pool:
        hm_xbs = pool.map(extract_jules_hx, f_names)
    pool.close()
    pool.join()
    #return 'Ensemble has been run'
    """
    hm_xbs = []
    for xb_fname in range(0, es.ensemble_size):
        print(xb_fname)
        #hxbi_dic = extract_jules_hx(es.output_directory + '/ensemble0/ens' + str(xb_fname) + '.daily.nc')
        #print(es.output_directory + '/ensemble/ens' + str(xb_fname) + '.daily.nc')
        hxbi_dic = extract_jules_hx(es.output_directory + '/ensemble/ens' + str(xb_fname) + '.daily.nc')
        hxbi = hxbi_dic['obs']
        hm_xbs.append(hxbi)
    """
    return hm_xbs
