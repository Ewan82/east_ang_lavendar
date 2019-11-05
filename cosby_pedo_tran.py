import numpy as np
#from SALib.sample import saltelli
#from SALib.analyze import sobol


def head(suction, g=9.80665, rho_water=1000.):
    """
    Converts suction (KPa) to head (m)
    :param suction: value of suction in KPa (float)
    :param g: gravity (float)
    :param rho_water: density of water (float)
    :return: head in m (float)
    """
    return (-1000.*suction)/(g*rho_water)


def cosby(f_clay, f_sand, f_silt, a=3.10, b=15.70, c=0.3, d=0.505, e=0.037, f=0.142, g=2.17, h=0.63, i=1.58, j=5.55, k=0.64,
          l=1.26, psi_c=-33., psi_w=-1500.):
    """
    Function calculating JULES soil hydrological parameters for Brooks & Corey model from soil texture fraction
    :param f_clay: fraction of clay (float)
    :param f_sand: fraction of sand (float)
    :param f_silt: fraction of silt (float)
    :param a: defined fn parameter (float)
    :param b: defined fn parameter (float)
    :param c: defined fn parameter (float)
    :param d: defined fn parameter (float)
    :param e: defined fn parameter (float)
    :param f: defined fn parameter (float)
    :param g: defined fn parameter (float)
    :param h: defined fn parameter (float)
    :param i: defined fn parameter (float)
    :param j: defined fn parameter (float)
    :param k: defined fn parameter (float)
    :param l: defined fn parameter (float)
    :return: b exponent, theta_sat, saturated soil moisture pressure (m),
             saturate soil hydraulic conductivity (kg/m2/s), theta_wilt, theta_crit
    """
    # Hydrological parameters
    psi_c = psi_c
    psi_w = psi_w
    b = a + b*f_clay - c*f_sand
    v_sat = d - e*f_clay - f*f_sand

    psi_s = 0.01*np.exp(g - h*f_clay - i*f_sand)
    k_s = np.exp(-j - k*f_clay + l*f_sand)
    head_c = head(psi_c)
    head_w = head(psi_w)
    v_wilt = v_sat*(psi_s/head_w)**(1/b)
    v_crit = v_sat*(psi_s/head_c)**(1/b)
    # Heat capacity, combines heat capapcities of clay, sand and silt linearly
    c_s = (1 - v_sat)*(f_clay*2.373e6 + f_sand*2.133e6 + f_silt*2.133e6)
    # Thermal conductivity
    lam = (0.025**(v_sat))*(1.16**(f_clay*(1-v_sat)))*(1.57**(f_sand*(1-v_sat)))*(1.57**(f_silt*(1-v_sat)))
    return np.array([b, v_sat, psi_s, k_s, v_wilt, v_crit, c_s, lam])


def van_g(oneovernminusone, oneoveralpha, k_sat, v_sat, psi_c=-33., psi_w=-1500.):
    """
    Function calculating JULES soil hydrological parameters for Brooks & Corey model from soil texture fraction
    :param oneoverminusone: Van-Genuchten soil parameter (float)
    :param oneoveralpha: Van-Genuchten soil parameter (float)
    :param k_sat: Hydraulic conductivity of soil at saturdation (float)
    :param v_sat: Value of saturated soil moisture (float)
    :param psi_c: Critical point (float)
    :param psi_w: Wilting point (float)
    :return: b exponent, theta_sat, saturated soil moisture pressure (m),
             saturate soil hydraulic conductivity (kg/m2/s), theta_wilt, theta_crit
    """
    # Hydrological parameters
    van_g_n = 1. + 1./(oneovernminusone)
    m = 1. - 1./van_g_n
    van_g_alpha = 1./oneoveralpha
    head_c = head(psi_c)
    head_w = head(psi_w)
    v_wilt = (1. + (van_g_alpha*head_w)**van_g_n)**(-m) * v_sat
    v_crit = (1. + (van_g_alpha*head_c)**van_g_n)**(-m) * v_sat
    # Heat capacity, combines heat capapcities of clay, sand and silt linearly
    c_s = (1 - v_sat)*1.942e6
    return np.array([oneoverminusone, oneoveralpha, k_sat, v_sat, v_wilt, v_crit, c_s])


def toth_van_g(f_clay, f_sand, f_silt, bulk_d, organic_c, cec, ph, topsoil, psi_c=-33., psi_w=-1500.,
               a=0.63052, b=0.10262, c=1.16518, d=0.16063, e=0.25929, f=0.10590, g=0.40220, hwsd_mu=None,
               ens_num=None):
    if f_sand >= 2.00:
        v_res = 0.041
    elif f_sand < 2.00:
        v_res = 0.179
    v_sat = a - b * bulk_d**2 + 0.0002904 * ph**2 + 0.0003335 * f_clay
    log10_alpha = -c + 0.40515 * (1 / (organic_c + 1)) - d * bulk_d**2 - 0.008372 * f_clay \
                  - 0.01300 * f_silt + 0.002166 * ph**2 + 0.08233 * topsoil
    log10_nminusone = -e + 0.25680 * (1 / (organic_c + 1)) - f * bulk_d**2 - 0.009004 * f_clay \
                      - 0.001223 * f_silt
    # v_sat = 0.83080 - 0.28217 * bulk_d + 0.0002728 * f_clay + 0.000187 * f_silt
    # log10_alpha = -0.43348 - 0.41729 * bulk_d - 0.04762 * organic_c + 0.21810 * topsoil - 0.01581 * f_clay\
    #              - 0.01207 * f_silt
    # log10_nminusone = 0.22236 - 0.30189 * bulk_d -0.05558 * topsoil - 0.005306 * f_clay - 0.003084 * f_silt\
    #                  - 0.01072 * organic_c
    log10_KS = g + 0.26122 * ph + 0.44565 * topsoil - 0.02329 * f_clay - 0.01265 * f_silt - 0.01038 * cec
    van_g_alpha = 10**(log10_alpha) * 100.0
    #print(van_g_alpha)
    van_g_n = 10**(log10_nminusone) + 1
    #print(van_g_n)
    KS = 10**(log10_KS)
    k_sat = 100*KS/(24*60*60)
    m = 1. - 1./van_g_n
    head_c = head(psi_c)
    head_w = head(psi_w)
    v_wilt = (1. + (van_g_alpha * head_w) ** van_g_n) ** (-m) * v_sat
    v_crit = (1. + (van_g_alpha * head_c) ** van_g_n) ** (-m) * v_sat
    # Heat capacity, combines heat capapcities of clay, sand and silt linearly
    if van_g_alpha < 1e-3:
        print(van_g_alpha, 1/van_g_alpha)
        print(hwsd_mu, topsoil, ens_num, a, c, e, g)
    if (van_g_n-1) < 1e-3:
        print((van_g_n-1), 1/(van_g_n-1))
        print(hwsd_mu, topsoil, ens_num, a, c, e, g)
    c_s = (1 - v_sat) * 1.942e6
    rho_r = 2700
    lam = (0.135*1000*bulk_d + 64.7) / (rho_r - 0.947*1000*bulk_d)
    return np.array([1/(van_g_n -1), 1/van_g_alpha, k_sat, v_sat - v_res, v_wilt - v_res,
                     v_crit - v_res, c_s, lam])
