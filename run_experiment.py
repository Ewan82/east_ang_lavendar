# core python modules:
import os
import sys
from functools import partial
from contextlib import contextmanager
# 3rd party modules:
import multiprocessing as mp
import shutil as sh
import glob
import pickle
import numpy as np
# local modules:
import fourdenvar
import experiment_setup as es
import run_jules as rjda
import jules
import create_bc_nc as cbn


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


def ens_member_run(ens_number_xi, seed_val=0, params=None, xa=False):
    """
    Function to run a ensemble or posterior ensemble member
    :param ens_number_xi: tuple of ensemble member number (int) and corresponding parameter vector (arr)
    :param seed_val: seed value used for any perturbations within experiment (int)
    :param params: parameters to update in ensemble member run (lst)
    :param xa: specify if this is a ensemble or posterior ensemble member (bool)
    :return: string confirming ensemble member has run (str)
    """
    if xa is True:
        ens_dir = '/ensemble_xa'
    else:
        ens_dir = '/ensemble'
    ens_number = ens_number_xi[0]
    xi = ens_number_xi[1]
    out_dir = 'nmls/ens_' + str(ens_number) + '_output_seed' +str(seed_val) + '/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for file in glob.glob(es.nml_directory + '/*'):
        sh.copy(file, out_dir)

    j = jules.Jules(out_dir)
    cwd = os.getcwd()
    if xa is True:
        j.nml_dic['ancillaries']['jules_soil_props']['file'] = \
            cwd+'/data/soil_ancil_xaens/ens' + str(ens_number) + '.nc'
    else:
        j.nml_dic['ancillaries']['jules_soil_props']['file'] = \
            cwd+'/data/soil_ancil_ens/ens' + str(ens_number) + '.nc'
    j.nml_dic['output']['jules_output']['run_id'] = 'ens' + str(ens_number)
    j.nml_dic['output']['jules_output']['output_dir'] = cwd+'/output' + ens_dir
    cwd = os.getcwd()
    os.chdir(out_dir)
    j.write_nml()
    os.system('bsub < runjules.sh')
    os.chdir(cwd)

    #sh.rmtree(out_dir)
    return 'ensemble member '+ str(ens_number) + ' run!'


def ens_run(x_ens, seed_val=0, xa=False, params=None):
    """
    Perform a parallel run of JULES models given an ensemble of paramter vectors
    :param x_ens: ensemble of paramter vectors (arr)
    :param seed_val: seed value used for any perturbations in the experiment (int)
    :param xa: switch if this is a ensemble or posterior ensemble run (bool)
    :param params: list of paramters being updated in experiment (lst)
    :return: string confirming if the ensemble has been run (str)
    """
    print('Running ensemble')
    mp.freeze_support()
    with poolcontext(processes=es.num_processes) as pool:
        res = pool.map(partial(ens_member_run, seed_val=seed_val, params=params, xa=xa), enumerate(x_ens))
    pool.close()
    pool.join()
    return 'Ensemble has been run'


def create_bc_nc_ens(x_ens, xa=False, params=0):
    """
    Perform a parallel run of JULES models given an ensemble of paramter vectors
    :param x_ens: ensemble of paramter vectors (arr)
    :param seed_val: seed value used for any perturbations in the experiment (int)
    :param xa: switch if this is a ensemble or posterior ensemble run (bool)
    :param params: list of paramters being updated in experiment (lst)
    :return: string confirming if the ensemble has been run (str)
    """
    print('Running ensemble')
    mp.freeze_support()
    with poolcontext(processes=es.num_processes) as pool:
        res = pool.map(partial(cbn.create_vg_nc_4params, xa=xa, params=params), enumerate(x_ens))
    pool.close()
    pool.join()
    return 'Ensemble has been run'


def cost(xi_xi_nc, xb=None, b_mat=None, rmat_lst=None, yoblist=None):
    """
    Calculates cost function for given xi
    :param xi: parameter vector (arr)
    :param xi_nc: netcdf chess_emma output file (str)
    :return: value of cost fn. (float)
    """
    xi = xi_xi_nc[0]
    xi_nc = xi_xi_nc[1]
    print(xi_nc)
    modcost = np.dot((xi - xb), np.dot(np.linalg.inv(b_mat), (xi - xb).T))
    hxi = es.jules_hxi(xi_nc)
    obcost = np.sum([np.dot(np.dot((hxi[x] - yoblist[x]), np.linalg.inv(rmat_lst[x])),
                            (hxi[x] - yoblist[x]).T) for x in range(len(yoblist))])
    return (xi_nc, 0.5 * modcost + 0.5 * obcost)


if __name__ == "__main__":
    # instantiate JULES data assimilation class
    #jda = fourdenvar.FourDEnVar(seed_val=int(sys.argv[1]))
    #seed_val = int(sys.argv[1])
    #params = jda.p_keys
    # if 'ancils' in system arguments create soil ancillaries
    if 'ancils' in sys.argv:
        create_bc_nc_ens(jda.xbs, xa=False)
    # if 'run_xb' is in system arguments then run JULES with ensemble parameters
    if 'run_xb' in sys.argv:
        nml_dir = 'output_seed' + str(seed_val) + '_xb/'
        if not os.path.exists(nml_dir):
            os.makedirs(nml_dir)
        for file in glob.glob(es.nml_directory + '/*.nml'):
            sh.copy(file, nml_dir)
        rj = rjda.RunJulesDa(params=params, values=jda.xb, nml_dir=nml_dir)
        rj.run_jules_dic(output_name='xb' + str(jda.seed_val), out_dir=es.output_directory+'/background/')
        sh.rmtree(nml_dir)
    # if 'run_xbs' is in system arguments then run prior ensemble
    if 'run_xbs' in sys.argv:
        # remove any old output in folders
        old_outs = glob.glob(es.output_directory + '/ensemble' + str(seed_val) + '/*.nc')
        for f in old_outs:
            os.remove(f)
        # run ensemble ensemble
        ens_run(jda.xbs, seed_val=jda.seed_val, params=params)
    # if 'assim' is in system arguments then run posterior ensemble
    if 'assim' in sys.argv:
        jda = fourdenvar.FourDEnVar(assim=True, seed_val=int(sys.argv[1]))
        params = jda.p_keys
        # find posterior estimate and posterior ensemble
        xa = jda.find_min_ens_inc()
        xa_ens = jda.a_ens(xa[1])
        # pickle posterior parameter ensemble array
        f = open(es.output_directory+'/xa2_ens' + str(es.ensemble_size) + '_seed' + sys.argv[1] + '.p', 'wb')
        pickle.dump(xa_ens, f)
        f.close()
    if 'find_likely' in sys.argv:
        jda = fourdenvar.FourDEnVar(assim=True, seed_val=int(sys.argv[1]))
        xa_files = ['/work/scratch/ewanp82/east_ang_lavendar/output/ensemble_xa/ens'+str(x)+'.daily.nc'
                    for x in range(50)]
        xa_ens = pickle.load(open('output/xa_ens50_seed0_smallerr.p', 'rb')).tolist()
        xa_files.pop(-1)
        xa_ens.pop(-1)
        print('Running ensemble')
        mp.freeze_support()
        with poolcontext(processes=es.num_processes) as pool:
            res = pool.map(partial(cost, xb=jda.xb, b_mat=jda.b_mat, rmat_lst=jda.rmat_lst, yoblist=jda.yoblist),
                           zip(xa_ens, xa_files))
        pool.close()
        pool.join()
        f = open('output/xb_ens50_likelihoods.p', 'wb')
        pickle.dump(res, f)
        f.close()
    # if 'run_xa' is in system arguments then run posterior ensemble
    if 'run_xa' in sys.argv:
        jda = fourdenvar.FourDEnVar(assim=True, seed_val=int(sys.argv[1]))
        params = jda.p_keys
        # find posterior estimate and posterior ensemble
        xa = jda.find_min_ens_inc()
        xa_ens = jda.a_ens(xa[1])
        xa_median = np.median(xa_ens, axis=0)
        nml_dir = 'output_seed' + str(seed_val) + '_xa/'
        if not os.path.exists(nml_dir):
            os.makedirs(nml_dir)
        for file in glob.glob(es.nml_directory + '/*.nml'):
            sh.copy(file, nml_dir)
        rj = rjda.RunJulesDa(params=params, values=xa_median, nml_dir=nml_dir)
        rj.run_jules_dic(output_name='xa' + str(jda.seed_val), out_dir=es.output_directory + '/analysis_med/')
        sh.rmtree(nml_dir)
        dumps = glob.glob(es.output_directory+'/analysis_med/'+'*.dump*.nc')
        for f in dumps:
            os.remove(f)
        # pickle posterior parameter ensemble array
        f = open(es.output_directory+'/xa_ens' + str(es.ensemble_size) + '_seed' + sys.argv[1] + '.p', 'wb')
        pickle.dump(xa_ens, f)
        f.close()
        # remove any old output in folders
        old_outs = glob.glob(es.output_directory + '/ensemble_xa_' + str(seed_val) + '/*.nc')
        for f in old_outs:
            os.remove(f)
        # run posterior ensemble
        ens_run(xa_ens, seed_val=jda.seed_val, xa=True, params=params)
    if 'plot' in sys.argv:
        es.save_plots(es.output_directory+'/xa_ens' + str(es.ensemble_size) + '_seed' + sys.argv[1] + '.p',
                      es.plot_output_dir)
    print('Experiment has been run')