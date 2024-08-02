import scipy.signal as signal

from eofs.multivariate.standard import MultivariateEof
from eofs.standard import Eof

from config import *
import xfilter   


#======================================
def calc_norm_factor(olr, u200, u850):
	olr_var = olr.var("time")
	u850_var = u850.var("time")
	u200_var = u200.var("time")
	olr_norm = np.sqrt(olr_var.mean("lon"))
	u850_norm = np.sqrt(u850_var.mean("lon"))
	u200_norm = np.sqrt(u200_var.mean("lon"))
	print("olr_norm", olr_norm)
	print("u850_norm", u850_norm)
	print("u200_norm", u200_norm)
	return olr_norm, u200_norm, u850_norm 

#======================================
def butterworth_lowpass_filter(data, order=2, cutoff_freq=1.0/10.0, axis=0):
    B, A = signal.butter(order, cutoff_freq, output="ba")
    return signal.filtfilt(B, A, data, axis=0)

#======================================
def apply_bandpass(OLR, U200, U850, f_low, f_high):
	print(' ----- Appliance of Bandpass ----- ')
	##--- Bandpass application variants ---##
	#1 OK
	# command0 = "cdo bandpass,3.65,18.25 -del29feb ./OLR_REF.nc      ./OLR_REF_bp.nc    "
	# os.system("/bin/bash -c \"" + command0 + "\"") 

	#2 NOT OK .swap_dims({"time": "dayofyear"}) .swap_dims({"dayofyear": "time"})
	# OLR_REF_anom =  dsp.bandpass(OLR_REF_anom.to_array(), f_low, f_high, dim="time").to_dataset(dim = 'variable')

	#3 OK 
	OLR = OLR.to_array().filter.bandpass([f_low, f_high], dim='time').to_dataset(dim = 'variable')
	U200 = U200.to_array().filter.bandpass([f_low, f_high], dim='time').to_dataset(dim = 'variable')
	U850 = U850.to_array().filter.bandpass([f_low, f_high], dim='time').to_dataset(dim = 'variable')
	return OLR, U200, U850

#======================================
def find_climatology(OLR, U200, U850):
	print(' ----- Computing climatologies ----- ')
	OLR_clim  = OLR.groupby('time.dayofyear').mean('time')
	U200_clim = U200.groupby('time.dayofyear').mean('time')
	U850_clim = U850.groupby('time.dayofyear').mean('time')

	OLR_clim_s  = dsp.lowpass(OLR_clim.to_array(),  1/90.0, dim="dayofyear")
	U200_clim_s = dsp.lowpass(U200_clim.to_array(), 1/90.0, dim="dayofyear")
	U850_clim_s = dsp.lowpass(U850_clim.to_array(), 1/90.0, dim="dayofyear")

	OLR_clim_s  = OLR_clim_s.to_dataset(dim = 'variable') #name 
	U200_clim_s = U200_clim_s.to_dataset(dim = 'variable')
	U850_clim_s = U850_clim_s.to_dataset(dim = 'variable')

	return OLR_clim_s, U200_clim_s, U850_clim_s


#======================================
def get_one_ans_memb(var_name, hour, ans_memb = 0): 
	ncfile = f'{nc_in_slav}/{var_name}/erfclim.{config["fcast_fn_y"]}{config["fcast_fn_m"]}{config["fcast_fn_d"]}{hour}_{ans_memb}-{var_name}.nc' # one member
	if (os.path.isfile(ncfile)):
		# print("Found : ", ncfile)
		VAR_SLAV = xr.open_dataset(ncfile).sel(lat=slice(latS,latN))
		VAR_SLAV = set_slav_datetime(VAR_SLAV) # current
		# VAR_SLAV = set_slav_datetime(VAR_SLAV, True) # 2008

	else:
		print("Error - no such file or directory: ", ncfile)
	return VAR_SLAV 


#======================================
def find_anomaly(VAR_PREV, VAR, VAR_CLIM):
	print(' ----- Subtracting climatologies from prev ----- ')
	VAR_PREV_anom = VAR_PREV.groupby('time.dayofyear')  - VAR_CLIM

	print(' ----- Subtracting ERA climatologies from forc ----- ')
	VAR_FORC_anom  = VAR.groupby('time.dayofyear')  - VAR_CLIM 

	print(' ----- Merge ERA and SLAV ----- ')
	VAR_anom  = xr.concat([VAR_PREV_anom,  VAR_FORC_anom],  "time") 

	print(' ----- Removing interannual variability (120d rolling mean) ----- ')
	VAR_anom1  = VAR_FORC_anom  -  VAR_anom.rolling(time=120, center=False).mean().dropna('time')
	
	return VAR_anom1

#======================================
def find_pcs(OLR_anom, U200_anom, U850_anom):
	olr  = OLR_anom['olr']
	u850 = U850_anom['u']
	u200 = U200_anom['u']
	olr  = olr.mean('lat')
	u850 = u850.mean('lat')
	u200 = u200.mean('lat')
	# print(' ----- Calculate normalization factors ----- ')
	# olr_norm, u200_norm, u850_norm = calc_norm_factor(olr, u200, u850)
	olr  = olr/olr_norm
	u850 = u850/u850_norm
	u200 = u200/u200_norm
	#======================================
	solver = MultivariateEof([np.array(olr), np.array(u850), np.array(u200)], center=True)
	eof_list = solver.eofs(neofs=2, eofscaling=0)
	pseudo_pcs = np.squeeze( solver.projectField([-1*np.array(olr), np.array(u850), np.array(u200)], eofscaling=1, neofs=2, weighted=False) )
	psc1, psc2 = [], []
	for pc in pseudo_pcs:
		psc1.append(pc[0])
		psc2.append(pc[1]) 
	return  psc1, psc2

#======================================

#======================================