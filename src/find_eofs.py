import netCDF4
from eofs.multivariate.standard import MultivariateEof
from eofs.standard import Eof
from painter import *
from config import *
from nc_funcs import * 

# OLR_clim, U200_clim, U850_clim = find_era_climatology()

OLR_REF 	= xr.open_dataset(f'{era_olr_path}/era5-olr-day-2p5grid-all.nc').sel(time=slice(date_clim_start,date_clim_end)).sel(lat=slice(latS,latN))
U200_REF	= xr.open_dataset(f'{era_u200_path}/era5-u200hpa-day-2p5grid-all.nc').sel(time=slice(date_clim_start,date_clim_end)).sel(lat=slice(latS, latN))
U850_REF	= xr.open_dataset(f'{era_u850_path}/era5-u850hpa-day-2p5grid-all.nc').sel(time=slice(date_clim_start,date_clim_end)).sel(lat=slice(latS, latN))

print(' ----- Subtracting timemean from ERA ----- ')
OLR_REF  = OLR_REF  - OLR_REF.mean("time") 
U200_REF = U200_REF - U200_REF.mean("time")
U850_REF = U850_REF - U850_REF.mean("time") 

OLR_clim, U200_clim, U850_clim = find_climatology(OLR_REF, U200_REF, U850_REF)

print(' ----- Subtracting climatologies from ERA ----- ')
OLR_REF_anom  = OLR_REF.groupby('time.dayofyear')  - OLR_clim
U200_REF_anom = U200_REF.groupby('time.dayofyear') - U200_clim
U850_REF_anom = U850_REF.groupby('time.dayofyear') - U850_clim

OLR_REF_anom, U200_REF_anom, U850_REF_anom = apply_bandpass(OLR_REF_anom, U200_REF_anom, U850_REF_anom, f_low, f_high)

print(' ----- Removing interannual variability (120d rolling mean) ----- ')  
OLR_REF_anom1  = OLR_REF_anom  -  OLR_REF_anom.rolling(time=120, center=False).mean().dropna('time')
U200_REF_anom1 = U200_REF_anom - U200_REF_anom.rolling(time=120, center=False).mean().dropna('time')
U850_REF_anom1 = U850_REF_anom - U850_REF_anom.rolling(time=120, center=False).mean().dropna('time')

#======================================
olr  = OLR_REF_anom1['olr']
u850 = U850_REF_anom1['u']
u200 = U200_REF_anom1['u']
olr  = olr.mean('lat')
u850 = u850.mean('lat')
u200 = u200.mean('lat')

print("---------------")
olr  = olr/olr_norm
u850 = u850/u850_norm
u200 = u200/u200_norm

solver = MultivariateEof([np.array(olr), np.array(u850), np.array(u200)], center=True)
eof_list = solver.eofs(neofs=2, eofscaling=0)
eof1_olr = np.fromstring(eof_list[0][0].tostring(), dtype=float)
eof2_olr = np.fromstring(eof_list[0][1].tostring(), dtype=float)
eof1_u200 = np.fromstring(eof_list[1][0].tostring(), dtype=float)
eof2_u200 = np.fromstring(eof_list[1][1].tostring(), dtype=float)
eof1_u850 = np.fromstring(eof_list[2][0].tostring(), dtype=float)
eof2_u850 = np.fromstring(eof_list[2][1].tostring(), dtype=float)


fig, ax = plt.subplots()
plt.xlim(0, 360)
plt.ylim(-0.15, 0.15)
plt.plot(np.arange(0, 360, 2.5), eof1_olr, '-', color='blue', ms=2, label='eof1_olr')
plt.plot(np.arange(0, 360, 2.5), eof2_olr, '-', color='red', ms=2, label='eof2_olr')
# plt.plot(np.arange(0, 360, 2.5), eof1_u200, '-o', color='blue', ms=2, label='eof1_u200')
# plt.plot(np.arange(0, 360, 2.5), eof2_u200, '-o', color='red', ms=2, label='eof2_u200')
# plt.plot(np.arange(0, 360, 2.5), eof1_u850, '--', color='blue', ms=2, label='eof1_u850')
# plt.plot(np.arange(0, 360, 2.5), eof2_u850, '--', color='red', ms=2, label='eof2_u850')
ax.plot([0.0, 1.0],[0.5, 0.5], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" )
ax.plot([0.0, 1.0],[0.5, 0.5], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" ) #horizontal
ax.plot([0.14, 0.14],[0, 1], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" ) #vertical
ax.plot([0.28, 0.28],[0, 1], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" ) #vertical
plt.legend()
# plt.savefig('eofV13.1-test.png')
plt.show()

