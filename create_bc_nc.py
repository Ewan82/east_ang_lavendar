import numpy as np
import netCDF4 as nc
import pickle
import itertools as itt
import shutil as sh
#third party modules
import cosby_pedo_tran as cpt


def find_nearest2(arr1, val1, arr2, val2):
    """
    Find nearest value in an array
    :param array: array of values
    :param value: value for which to find nearest element
    :return: nearest value in array, index of nearest value
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    idx = (np.abs(arr1 - val1) + np.abs(arr2 -val2)).argmin()
    #print 'jules lat', arr1[idx-1], arr1[idx], arr1[idx]
    #print 'jules lon', arr2[idx-1], arr2[idx], arr2[idx]
    near_val1 = arr1.flatten()[idx]
    near_val2 = arr2.flatten()[idx]
    idx1 = np.where(arr1 == near_val1)
    idx2 = np.where(arr2 == near_val2)
    y_idx = np.intersect1d(idx1[0],idx2[0])[0]
    x_idx = np.intersect1d(idx1[1],idx2[1])[0]
    return y_idx, x_idx


def create_bc_nc():
    soil_rec = pickle.load(open('hwsd_mu_texture.p', 'rb'))
    #soil_rec = np.recfromcsv('emma_hwsd_bc/HWSD_DATA.txt', usecols=(1,6,24,25,26,41,42,43))
    hwsd = nc.Dataset('emma_hwsd_bc/hwsd_uk_1km_grid.nc', 'r')
    bc_params = nc.Dataset('bc_params.nc', 'a')
    j=0
    for mu in np.unique(hwsd.variables['mu'][:])[:-1]:
        j+=1
        print(j, ' out of ', len(np.unique(hwsd.variables['mu'][:])[:-1]))
    #for mu in [10770]:
        t_sand = soil_rec['t_sand'][soil_rec['mu_global'] == mu][0]
        t_silt = soil_rec['t_silt'][soil_rec['mu_global'] == mu][0]
        t_clay = soil_rec['t_clay'][soil_rec['mu_global'] == mu][0]
        s_sand = soil_rec['s_sand'][soil_rec['mu_global'] == mu][0]
        s_silt = soil_rec['s_silt'][soil_rec['mu_global'] == mu][0]
        s_clay = soil_rec['s_clay'][soil_rec['mu_global'] == mu][0]
        t_bc = cpt.cosby(t_clay/100., t_sand/100., t_silt/100.)
        s_bc = cpt.cosby(s_clay/100., s_sand/100., s_silt/100.)
        loc = np.where(hwsd.variables['mu'][:] == float(mu))
        loc_itt = itt.izip(loc[1], loc[2])
        keys = ['b', 'vsat', 'sathh', 'satcon', 'vwilt', 'vcrit', 'hcap', 'hcon']
        for k in enumerate(keys):
            for i in loc_itt:
                #print k[1]
                #print bc_params.variables[k[1]][:2, i[0], i[1]]
                #print t_bc[k[0]]
                #print bc_params.variables[k[1]][2:, i[0], i[1]]
                #print s_bc[k[0]]
                bc_params.variables[k[1]][:2, i[0], i[1]] = t_bc[k[0]]
                bc_params.variables[k[1]][2:, i[0], i[1]] = s_bc[k[0]]
    bc_params.close()
    hwsd.close()
    return 'soil params updated!'


def create_bc_nc_9layer():
    soil_rec = pickle.load(open('hwsd_mu_texture.p', 'rb'))
    #soil_rec = np.recfromcsv('emma_hwsd_bc/HWSD_DATA.txt', usecols=(1,6,24,25,26,41,42,43))
    hwsd = nc.Dataset('emma_hwsd_bc/hwsd_uk_1km_grid.nc', 'r')
    bc_params = nc.Dataset('bc_params_9layer.nc', 'a')
    j=0
    for mu in np.unique(hwsd.variables['mu'][:])[:-1]:
        j+=1
        print(j, ' out of ', len(np.unique(hwsd.variables['mu'][:])[:-1]))
    #for mu in [10770]:
        t_sand = soil_rec['t_sand'][soil_rec['mu_global'] == mu][0]
        t_silt = soil_rec['t_silt'][soil_rec['mu_global'] == mu][0]
        t_clay = soil_rec['t_clay'][soil_rec['mu_global'] == mu][0]
        s_sand = soil_rec['s_sand'][soil_rec['mu_global'] == mu][0]
        s_silt = soil_rec['s_silt'][soil_rec['mu_global'] == mu][0]
        s_clay = soil_rec['s_clay'][soil_rec['mu_global'] == mu][0]
        t_bc = cpt.cosby(t_clay/100., t_sand/100., t_silt/100.)
        s_bc = cpt.cosby(s_clay/100., s_sand/100., s_silt/100.)
        loc = np.where(hwsd.variables['mu'][:] == float(mu))
        loc_itt = itt.izip(loc[1], loc[2])
        keys = ['b', 'vsat', 'sathh', 'satcon', 'vwilt', 'vcrit', 'hcap', 'hcon']
        for i in loc_itt:
            for k in enumerate(keys):
                #print k[1]
                #print bc_params.variables[k[1]][:2, i[0], i[1]]
                #print t_bc[k[0]]
                #print bc_params.variables[k[1]][2:, i[0], i[1]]
                #print s_bc[k[0]]
                bc_params.variables[k[1]][:7, i[0], i[1]] = t_bc[k[0]]
                bc_params.variables[k[1]][7:, i[0], i[1]] = s_bc[k[0]]
    bc_params.close()
    hwsd.close()
    return 'soil params updated!'


def create_bc_nc_params(ens_number_xi=(0,[15.70, 0.3, 0.037, 0.142, 0.63, 1.58, 0.64, 1.26]), xa=False):
    #ens_number_xi=(0,[15.70, 0.3, 0.037, 0.142, 0.63, 1.58, 0.64, 1.26])
    if xa is True:
        ens_dir = 'data/soil_ancil_xaens'
    else:
        ens_dir = 'data/soil_ancil_ens'
    ens_number = ens_number_xi[0]
    param_lst = ens_number_xi[1]
    bc_fname = ens_dir + '/ens' + str(ens_number) + '.nc'
    soil_rec = pickle.load(open('hwsd_mu_texture.p', 'rb'))
    #soil_rec = np.recfromcsv('emma_hwsd_bc/HWSD_DATA.txt', usecols=(1,6,24,25,26,41,42,43))
    hwsd = nc.Dataset('emma_hwsd_bc/hwsd_uk_1km_grid.nc', 'r')
    sh.copy('bc_params.nc', bc_fname)
    bc_params = nc.Dataset(bc_fname, 'a')
    j=0
    for mu in np.unique(hwsd.variables['mu'][:])[:-1]:
        j+=1
        print(j, ' out of ', len(np.unique(hwsd.variables['mu'][:])[:-1]))
    #for mu in [10770]:
        t_sand = soil_rec['t_sand'][soil_rec['mu_global'] == mu][0]
        t_silt = soil_rec['t_silt'][soil_rec['mu_global'] == mu][0]
        t_clay = soil_rec['t_clay'][soil_rec['mu_global'] == mu][0]
        s_sand = soil_rec['s_sand'][soil_rec['mu_global'] == mu][0]
        s_silt = soil_rec['s_silt'][soil_rec['mu_global'] == mu][0]
        s_clay = soil_rec['s_clay'][soil_rec['mu_global'] == mu][0]
        t_bc = cpt.cosby(t_clay/100., t_sand/100., t_silt/100., b=param_lst[0], c=param_lst[1], e=param_lst[2],
                         f=param_lst[3], h=param_lst[4], i=param_lst[5], k=param_lst[6], l=param_lst[7])
        s_bc = cpt.cosby(s_clay/100., s_sand/100., s_silt/100., b=param_lst[0], c=param_lst[1], e=param_lst[2],
                         f=param_lst[3], h=param_lst[4], i=param_lst[5], k=param_lst[6], l=param_lst[7])
        loc = np.where(hwsd.variables['mu'][:] == float(mu))
        loc_itt = itt.izip(loc[1], loc[2])
        keys = ['b', 'vsat', 'sathh', 'satcon', 'vwilt', 'vcrit', 'hcap', 'hcon']
        for i in loc_itt:
            for k in enumerate(keys):
                #print k[1]
                #print bc_params.variables[k[1]][:2, i[0], i[1]]
                #print t_bc[k[0]]
                #print bc_params.variables[k[1]][2:, i[0], i[1]]
                #print s_bc[k[0]]
                bc_params.variables[k[1]][:2, i[0], i[1]] = t_bc[k[0]]
                bc_params.variables[k[1]][2:, i[0], i[1]] = s_bc[k[0]]
    bc_params.close()
    hwsd.close()
    return 'soil params updated!'


def create_bc_nc_params2(ens_number_xi=(0,[15.70, 0.3, 0.037, 0.142, 0.63, 1.58, 0.64, 1.26]), xa=False, params=0):
    #ens_number_xi=(0,[15.70, 0.3, 0.037, 0.142, 0.63, 1.58, 0.64, 1.26])
    if xa is True:
        ens_dir = 'data/soil_ancil_xaens'
    else:
        ens_dir = 'data/soil_ancil_ens'
    ens_number = ens_number_xi[0]
    param_lst = ens_number_xi[1]
    bc_fname = ens_dir + '/ens' + str(ens_number) + '.nc'
    soil_rec = pickle.load(open('hwsd_mu_texture.p', 'rb'))
    #soil_rec = np.recfromcsv('emma_hwsd_bc/HWSD_DATA.txt', usecols=(1,6,24,25,26,41,42,43))
    hwsd = nc.Dataset('emma_hwsd_bc/hwsd_uk_1km_grid.nc', 'r')
    sh.copy('bc_params.nc', bc_fname)
    bc_params = nc.Dataset(bc_fname, 'a')
    j=0
    for mu in np.unique(hwsd.variables['mu'][0,244:258,505:512]):
        j+=1
        print(j, ' out of ', len(np.unique(hwsd.variables['mu'][0,244:258,505:512])))
    #for mu in [10770]:
        t_sand = soil_rec['t_sand'][soil_rec['mu_global'] == mu][0]
        t_silt = soil_rec['t_silt'][soil_rec['mu_global'] == mu][0]
        t_clay = soil_rec['t_clay'][soil_rec['mu_global'] == mu][0]
        s_sand = soil_rec['s_sand'][soil_rec['mu_global'] == mu][0]
        s_silt = soil_rec['s_silt'][soil_rec['mu_global'] == mu][0]
        s_clay = soil_rec['s_clay'][soil_rec['mu_global'] == mu][0]
        t_bc = cpt.cosby(t_clay/100., t_sand/100., t_silt/100., b=param_lst[0], c=param_lst[1], e=param_lst[2],
                         f=param_lst[3], h=param_lst[4], i=param_lst[5], k=param_lst[6], l=param_lst[7])
        s_bc = cpt.cosby(s_clay/100., s_sand/100., s_silt/100., b=param_lst[0], c=param_lst[1], e=param_lst[2],
                         f=param_lst[3], h=param_lst[4], i=param_lst[5], k=param_lst[6], l=param_lst[7])
        loc = np.where(hwsd.variables['mu'][0,244:258,505:512] == float(mu))
        loc_itt = itt.izip(loc[0]+244, loc[1]+505)
        keys = ['b', 'vsat', 'sathh', 'satcon', 'vwilt', 'vcrit', 'hcap', 'hcon']
        for i in loc_itt:
            for k in enumerate(keys):
                #print k[1]
                #print bc_params.variables[k[1]][:2, i[0], i[1]]
                #print t_bc[k[0]]
                #print bc_params.variables[k[1]][2:, i[0], i[1]]
                #print s_bc[k[0]]
                bc_params.variables[k[1]][:2, i[0], i[1]] = t_bc[k[0]]
                bc_params.variables[k[1]][2:, i[0], i[1]] = s_bc[k[0]]
    bc_params.close()
    hwsd.close()
    return 'soil params updated!'


def create_bc_nc_4params(ens_number_xi=(0,[3.10, 0.505, 2.17, 5.55]), xa=False, params=0):
    # =(0,[3.10, 0.505, 2.17, 5.55])
    #ens_number_xi=(0,[15.70, 0.3, 0.037, 0.142, 0.63, 1.58, 0.64, 1.26])
    if xa is True:
        ens_dir = 'data/soil_ancil_xaens'
    else:
        ens_dir = 'data/soil_ancil_ens'
    ens_number = ens_number_xi[0]
    param_lst = ens_number_xi[1]
    bc_fname = ens_dir + '/ens' + str(ens_number) + '.nc'
    soil_rec = pickle.load(open('hwsd_mu_texture.p', 'rb'))
    #soil_rec = np.recfromcsv('emma_hwsd_bc/HWSD_DATA.txt', usecols=(1,6,24,25,26,41,42,43))
    hwsd = nc.Dataset('emma_hwsd_bc/hwsd_uk_1km_grid.nc', 'r')
    sh.copy('bc_params.nc', bc_fname)
    bc_params = nc.Dataset(bc_fname, 'a')
    j=0
    for mu in np.unique(hwsd.variables['mu'][0, 244:258, 505:512]):
        j+=1
        print(j, ' out of ', len(np.unique(hwsd.variables['mu'][0, 244:258, 505:512])))
    #for mu in [10770]:
        t_sand = soil_rec['t_sand'][soil_rec['mu_global'] == mu][0]
        t_silt = soil_rec['t_silt'][soil_rec['mu_global'] == mu][0]
        t_clay = soil_rec['t_clay'][soil_rec['mu_global'] == mu][0]
        s_sand = soil_rec['s_sand'][soil_rec['mu_global'] == mu][0]
        s_silt = soil_rec['s_silt'][soil_rec['mu_global'] == mu][0]
        s_clay = soil_rec['s_clay'][soil_rec['mu_global'] == mu][0]
        t_bc = cpt.cosby(t_clay/100., t_sand/100., t_silt/100., a=param_lst[0], d=param_lst[1], g=param_lst[2],
                         j=param_lst[3])
        s_bc = cpt.cosby(s_clay/100., s_sand/100., s_silt/100., a=param_lst[0], d=param_lst[1], g=param_lst[2],
                         j=param_lst[3])
        loc = np.where(hwsd.variables['mu'][0,244:258,505:512] == float(mu))
        loc_itt = itt.izip(loc[0]+244, loc[1]+505)
        keys = ['b', 'vsat', 'sathh', 'satcon', 'vwilt', 'vcrit', 'hcap', 'hcon']
        for i in loc_itt:
            for k in enumerate(keys):
                #print k[1]
                #print bc_params.variables[k[1]][:2, i[0], i[1]]
                #print t_bc[k[0]]
                #print bc_params.variables[k[1]][2:, i[0], i[1]]
                #print s_bc[k[0]]
                bc_params.variables[k[1]][:2, i[0], i[1]] = t_bc[k[0]]
                bc_params.variables[k[1]][2:, i[0], i[1]] = s_bc[k[0]]
                #print bc_params.variables[k[1]][:2, i[0], i[1]]
                #print t_bc[k[0]]
                #print bc_params.variables[k[1]][2:, i[0], i[1]]
                #print s_bc[k[0]]
    bc_params.close()
    hwsd.close()
    return 'soil params updated!'


def create_bc_nc_params_test(ens_number_xi, xa=False, params=0):
    # =(0,[3.10, 0.505, 2.17, 5.55])
    #ens_number_xi=(0,[15.70, 0.3, 0.037, 0.142, 0.63, 1.58, 0.64, 1.26])
    if xa is True:
        ens_dir = 'data/soil_ancil_xaens'
    else:
        ens_dir = 'data/soil_ancil_ens'
    ens_number = ens_number_xi[0]
    param_lst = ens_number_xi[1]
    bc_fname = ens_dir + '/ens' + str(ens_number) + '.nc'
    soil_rec = pickle.load(open('hwsd_mu_texture.p', 'rb'))
    #soil_rec = np.recfromcsv('emma_hwsd_bc/HWSD_DATA.txt', usecols=(1,6,24,25,26,41,42,43))
    hwsd = nc.Dataset('emma_hwsd_bc/hwsd_uk_1km_grid.nc', 'r')
    sh.copy('bc_params.nc', bc_fname)
    bc_params = nc.Dataset(bc_fname, 'a')
    j=0
    for mu in np.unique(hwsd.variables['mu'][0,244:246,505:507]):
        j+=1
        print(j, ' out of ', len(np.unique(hwsd.variables['mu'][0,244:246,505:507])))
    #for mu in [10770]:
        t_sand = soil_rec['t_sand'][soil_rec['mu_global'] == mu][0]
        t_silt = soil_rec['t_silt'][soil_rec['mu_global'] == mu][0]
        t_clay = soil_rec['t_clay'][soil_rec['mu_global'] == mu][0]
        s_sand = soil_rec['s_sand'][soil_rec['mu_global'] == mu][0]
        s_silt = soil_rec['s_silt'][soil_rec['mu_global'] == mu][0]
        s_clay = soil_rec['s_clay'][soil_rec['mu_global'] == mu][0]
        t_bc = cpt.cosby(t_clay/100., t_sand/100., t_silt/100., a=param_lst[0], d=param_lst[1], g=param_lst[2],
                         j=param_lst[3])
        print(t_bc)
        s_bc = cpt.cosby(s_clay/100., s_sand/100., s_silt/100., a=param_lst[0], d=param_lst[1], g=param_lst[2],
                         j=param_lst[3])
        loc = np.where(hwsd.variables['mu'][0,244:246,505:507] == float(mu))
        loc_itt = itt.izip(loc[0]+244, loc[1]+505)
        print(bc_params.variables['vsat'][0, 244:246, 505:507])
        keys = ['b', 'vsat', 'sathh', 'satcon', 'vwilt', 'vcrit', 'hcap', 'hcon']
        for i in loc_itt:
            for k in enumerate(keys):
                #print k[1]
                #print bc_params.variables[k[1]][:2, i[0], i[1]]
                #print t_bc[k[0]]
                #print bc_params.variables[k[1]][2:, i[0], i[1]]
                #print s_bc[k[0]]
                bc_params.variables[k[1]][:2, i[0], i[1]] = t_bc[k[0]]
                bc_params.variables[k[1]][2:, i[0], i[1]] = s_bc[k[0]]
                #print bc_params.variables[k[1]][:2, i[0], i[1]]
                #print t_bc[k[0]]
                #print bc_params.variables[k[1]][2:, i[0], i[1]]
    #print bc_params.variables['vsat'][0,244:246,505:507]
    bc_params.close()
    hwsd.close()
    return 'soil params updated!'


#a = 0.63052, b = 0.10262, c = 1.16518, d = 0.16063, e = 0.25929, f = 0.10590, g = 0.40220
def create_vg_nc_params(ens_number_xi=(0,[0.63052, 1.16518, 0.25929, 0.40220]), xa=False, hwsd_mu=10769.0):
    soil_rec = np.recfromcsv('hwsd_mu_texture_allfields.csv')  # = pickle.load(open('hwsd_mu_texture.p', 'rb'))
    #soil_rec = np.recfromcsv('emma_hwsd_bc/HWSD_DATA.txt', usecols=(1,6,24,25,26,41,42,43))
    #hwsd = nc.Dataset('emma_hwsd_bc/hwsd_uk_1km_grid.nc', 'r')
    if xa is True:
        ens_dir = 'data/soil_ancil_xaens'
    else:
        ens_dir = 'data/soil_ancil_ens'
    ens_number = ens_number_xi[0]
    param_lst = ens_number_xi[1]
    bc_fname = ens_dir + '/ens' + str(ens_number) + '.nc'
    sh.copy('data/soilpropsvg_CARDT.nc', bc_fname)
    bc_params = nc.Dataset(bc_fname, 'a')

    t_sand = soil_rec['t_sand'][soil_rec['mu_global'] == hwsd_mu][0]
    t_silt = soil_rec['t_silt'][soil_rec['mu_global'] == hwsd_mu][0]
    t_clay = soil_rec['t_clay'][soil_rec['mu_global'] == hwsd_mu][0]
    t_bulk = soil_rec['t_ref_bulk_density'][soil_rec['mu_global'] == hwsd_mu][0]
    #print(t_bulk)
    t_orgc = soil_rec['t_oc'][soil_rec['mu_global'] == hwsd_mu][0]
    t_ph = soil_rec['t_ph_h2o'][soil_rec['mu_global'] == hwsd_mu][0]
    t_cec = soil_rec['t_cec_soil'][soil_rec['mu_global'] == hwsd_mu][0]
    #print(t_ph)
    s_sand = soil_rec['s_sand'][soil_rec['mu_global'] == hwsd_mu][0]
    s_silt = soil_rec['s_silt'][soil_rec['mu_global'] == hwsd_mu][0]
    s_clay = soil_rec['s_clay'][soil_rec['mu_global'] == hwsd_mu][0]
    s_bulk = soil_rec['s_ref_bulk_density'][soil_rec['mu_global'] == hwsd_mu][0]
    s_orgc = soil_rec['s_oc'][soil_rec['mu_global'] == hwsd_mu][0]
    s_ph = soil_rec['s_ph_h2o'][soil_rec['mu_global'] == hwsd_mu][0]
    s_cec = soil_rec['s_cec_soil'][soil_rec['mu_global'] == hwsd_mu][0]
    t_vg = cpt.toth_van_g(t_clay, t_sand, t_silt, t_bulk, t_orgc, t_cec, t_ph, topsoil=1, a=param_lst[0],
                          c=param_lst[1], e=param_lst[2], g=param_lst[3])
    #print(t_vg)
    s_vg = cpt.toth_van_g(s_clay, s_sand, s_silt, s_bulk, s_orgc, s_cec, s_ph, topsoil=0, a=param_lst[0],
                          c=param_lst[1], e=param_lst[2], g=param_lst[3])
    #print(s_vg)
    keys = ['oneovernminusone', 'oneoveralpha', 'satcon', 'vsat', 'vwilt', 'vcrit', 'hcap', 'hcon']
    for k in enumerate(keys):
        #print(k[1], bc_params.variables[k[1]][0, 0], t_vg[k[0]])
        bc_params.variables[k[1]][:2, 0] = t_vg[k[0]]
        bc_params.variables[k[1]][2:, 0] = s_vg[k[0]]
    bc_params.close()
    return 'soil params updated!'


def create_vg_nc_4params(ens_number_xi=(0, [0.63052, 1.16518, 0.25929, 0.40220]), xa=False, params=0):
    if xa is True:
        ens_dir = 'data/soil_ancil_xaens'
    else:
        ens_dir = 'data/soil_ancil_ens'
    ens_number = ens_number_xi[0]
    param_lst = ens_number_xi[1]
    bc_fname = ens_dir + '/ens' + str(ens_number) + '.nc'
    soil_rec = np.recfromcsv('hwsd_mu_texture_allfields.csv')  # = pickle.load(open('hwsd_mu_texture.p', 'rb'))
    hwsd = nc.Dataset('emma_hwsd_bc/hwsd_uk_1km_grid.nc', 'r')
    sh.copy('vg_params.nc', bc_fname)
    bc_params = nc.Dataset(bc_fname, 'a')
    mu_arr = np.unique(hwsd.variables['mu'][0, 188:430, 451:656]).data
    mu_arr = mu_arr[mu_arr != -9999.0]
    j = 0
    for hwsd_mu in mu_arr:
        j += 1
        #print(j, ' out of ', len(mu_arr))
        #print(hwsd_mu)
        t_sand = soil_rec['t_sand'][soil_rec['mu_global'] == hwsd_mu][0]
        t_silt = soil_rec['t_silt'][soil_rec['mu_global'] == hwsd_mu][0]
        t_clay = soil_rec['t_clay'][soil_rec['mu_global'] == hwsd_mu][0]
        t_bulk = soil_rec['t_ref_bulk_density'][soil_rec['mu_global'] == hwsd_mu][0]
        # print(t_bulk)
        t_orgc = soil_rec['t_oc'][soil_rec['mu_global'] == hwsd_mu][0]
        t_ph = soil_rec['t_ph_h2o'][soil_rec['mu_global'] == hwsd_mu][0]
        t_cec = soil_rec['t_cec_soil'][soil_rec['mu_global'] == hwsd_mu][0]
        #print(t_sand, t_silt, t_clay, t_bulk, t_orgc, t_ph, t_cec)
        s_sand = soil_rec['s_sand'][soil_rec['mu_global'] == hwsd_mu][0]
        s_silt = soil_rec['s_silt'][soil_rec['mu_global'] == hwsd_mu][0]
        s_clay = soil_rec['s_clay'][soil_rec['mu_global'] == hwsd_mu][0]
        s_bulk = soil_rec['s_ref_bulk_density'][soil_rec['mu_global'] == hwsd_mu][0]
        s_orgc = soil_rec['s_oc'][soil_rec['mu_global'] == hwsd_mu][0]
        s_ph = soil_rec['s_ph_h2o'][soil_rec['mu_global'] == hwsd_mu][0]
        s_cec = soil_rec['s_cec_soil'][soil_rec['mu_global'] == hwsd_mu][0]
        t_vg = cpt.toth_van_g(t_clay, t_sand, t_silt, t_bulk, t_orgc, t_cec, t_ph, topsoil=1, a=param_lst[0],
                              c=param_lst[1], e=param_lst[2], g=param_lst[3], hwsd_mu=hwsd_mu, ens_num=ens_number)
        #print(t_vg)
        if hwsd_mu == 10748.0:
            s_vg = cpt.toth_van_g(t_clay, t_sand, t_silt, t_bulk, t_orgc, t_cec, t_ph, topsoil=0, a=param_lst[0],
                                  c=param_lst[1], e=param_lst[2], g=param_lst[3], hwsd_mu=hwsd_mu, ens_num=ens_number)
            print(t_sand, t_silt, t_clay, t_bulk, t_orgc, t_ph, t_cec)
            print(s_sand, s_silt, s_clay, s_bulk, s_orgc, s_ph, s_cec)
            print(len(np.where(hwsd.variables['mu'][0, 188:430, 451:656] == float(hwsd_mu))[0]))
        else:
            s_vg = cpt.toth_van_g(s_clay, s_sand, s_silt, s_bulk, s_orgc, s_cec, s_ph, topsoil=0, a=param_lst[0],
                                  c=param_lst[1], e=param_lst[2], g=param_lst[3], hwsd_mu=hwsd_mu, ens_num=ens_number)
        loc = np.where(hwsd.variables['mu'][0, 188:430, 451:656] == float(hwsd_mu))
        loc_itt = zip(loc[0]+188, loc[1]+451)
        keys = ['oneovernminusone', 'oneoveralpha', 'satcon', 'vsat', 'vwilt', 'vcrit', 'hcap', 'hcon']
        for i in loc_itt:
            for k in enumerate(keys):
                #print(k[1])
                #print(bc_params.variables[k[1]][:2, i[0], i[1]])
                #print(t_vg[k[0]])
                #print bc_params.variables[k[1]][2:, i[0], i[1]]
                #print s_vg[k[0]]
                bc_params.variables[k[1]][:2, i[0], i[1]] = t_vg[k[0]]
                bc_params.variables[k[1]][2:, i[0], i[1]] = s_vg[k[0]]
                #print(bc_params.variables[k[1]][:2, i[0], i[1]])
                #print(t_vg[k[0]])
                #print bc_params.variables[k[1]][2:, i[0], i[1]]
                #print s_vg[k[0]]
    bc_params.close()
    hwsd.close()
    return 'soil params updated!'