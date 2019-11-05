import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import create_bc_nc as cbn

dat = nc.Dataset('/work/scratch/ewanp82/jules5.3/chess_da_ea/output_folder / test.daily.nc', 'r')
grid = nc.Dataset(
    '/group_workspaces/jasmin2/jules_bd/data/CHESS_v1.0/ancils_uncompressed / chess_soilparams_hwsd_vg.nc', 'r')
lat = grid.variables['lat'][:]
lon = grid.variables['lon'][:]
sm = np.zeros((366, 656, 1057))
for x in range(49588):  # make this faster!
    yidx, xidx = cbn.find_nearest2(lon, dat.variables['longitude'][0, x], lat, dat.variables['latitude'][0, x])
    sm[:, xidx, yidx] = dat.variables['smcl'][:, 0, 0, x] / 100.
idx = np.where(sm[0] != 0.0)
sm_ea = sm[:, min(idx[0]):max(idx[0]), min(idx[1]):max(idx[1])]
vsat = grid.variables['vsat'][0, min(idx[1]):max(idx[1]), min(idx[0]):max(idx[0])]
fig, ax = plt.subplots()
sm_ea_wet = sm_ea / vsat.T  # can use this to plot soil wetness (soil moisture normalised by theta_sat, (0, 1))
cs = ax.imshow(np.rot90(sm_ea[0]), vmin=0.0, vmax=0.5, cmap='YlGnBu')
cbar = fig.colorbar(cs)
cbar.ax.set_ylabel('Soil Moisture (m3 m-3)')
times = nc.num2date(dat.variables['time'][:], dat.variables['time'].units)
months = [dt.datetime.strftime(times[x], '%b %Y') for x in range(len(times))]
ax.set_title(months[0])
for x in range(366):
    cs.set_array(np.rot90(sm3[x]))
    ax.set_title(months[x])
    fig.savefig('sm_plot/sm' + str(x).zfill(3) + '.png', bbox_inches='tight', dpi=80)
