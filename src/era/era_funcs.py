import xfilter   

from config import *

#======================================
def get_era_data():
	OLR    = xr.open_dataset(f'{era_olr_path}/era5-olr-day-2p5grid-all.nc').sel(time=slice(date_clim_start,date_clim_end)).sel(lat=slice(latS,latN))
	U200   = xr.open_dataset(f'{era_u200_path}/era5-u200hpa-day-2p5grid-all.nc').sel(time=slice(date_clim_start,date_clim_end)).sel(lat=slice(latS, latN))
	U850   = xr.open_dataset(f'{era_u850_path}/era5-u850hpa-day-2p5grid-all.nc').sel(time=slice(date_clim_start,date_clim_end)).sel(lat=slice(latS, latN))

	return OLR, U200, U850

#======================================
def subtract_timemean(OLR, U200, U850):
	# print(' ----- Subtracting timemean from ERA ----- ')
	OLR  = OLR  - OLR.mean("time") 
	U200 = U200 - U200.mean("time")
	U850 = U850 - U850.mean("time")

	return OLR, U200, U850

#======================================
def find_era_climatology():
	print(' ----- Computing ERA climatologies ----- ')
	OLR, U200, U850 = get_era_data()
	
	# print(' ----- Subtract timemean from data ')
	OLR, U200, U850 = subtract_timemean(OLR, U200, U850)

	OLR_clim  = OLR.groupby('time.dayofyear').mean('time')
	U200_clim = U200.groupby('time.dayofyear').mean('time')
	U850_clim = U850.groupby('time.dayofyear').mean('time')

	# print(' ----- Smoothing climatologies ----- ')

	OLR_clim_s  = dsp.lowpass(OLR_clim.to_array(),  1/90.0, dim="dayofyear") ### Good  1/90 - 1/80
	U200_clim_s = dsp.lowpass(U200_clim.to_array(), 1/90.0, dim="dayofyear") ## bandpass - worse
	U850_clim_s = dsp.lowpass(U850_clim.to_array(), 1/90.0, dim="dayofyear")

	OLR_clim_s  = OLR_clim_s.to_dataset(dim = 'variable') #name 
	U200_clim_s = U200_clim_s.to_dataset(dim = 'variable')
	U850_clim_s = U850_clim_s.to_dataset(dim = 'variable')

	return OLR_clim_s, U200_clim_s, U850_clim_s

#======================================
def get_ref_data_era():
	print(' ----- Reference data era5 -----')
	OLR_REF  	= xr.open_dataset(f'{era_olr_path}/era5-olr-day-2p5grid-all.nc').sel(time=slice(config["date_ref_start"],config["date_ref_end"])).sel(lat=slice(latS,latN))
	U200_REF 	= xr.open_dataset(f'{era_u200_path}/era5-u200hpa-day-2p5grid-all.nc').sel(time=slice(config["date_ref_start"],config["date_ref_end"])).sel(lat=slice(latS, latN))
	U850_REF 	= xr.open_dataset(f'{era_u850_path}/era5-u850hpa-day-2p5grid-all.nc').sel(time=slice(config["date_ref_start"],config["date_ref_end"])).sel(lat=slice(latS, latN))

	return OLR_REF, U200_REF, U850_REF

#======================================
def get_forc_data_era():
	print(' ----- Forecast ERA5 ----- ') # 
	OLR_FORC   = xr.open_dataset(f'{era_olr_path}/era5-olr-day-2p5grid-all.nc').sel(time=slice(config["date_forc_start"],config["date_forc_end"])).sel(lat=slice(latS,latN))
	U200_FORC  = xr.open_dataset(f'{era_u200_path}/era5-u200hpa-day-2p5grid-all.nc').sel(time=slice(config["date_forc_start"],config["date_forc_end"])).sel(lat=slice(latS, latN))
	U850_FORC  = xr.open_dataset(f'{era_u850_path}/era5-u850hpa-day-2p5grid-all.nc').sel(time=slice(config["date_forc_start"],config["date_forc_end"])).sel(lat=slice(latS, latN))

	return OLR_FORC, U200_FORC, U850_FORC

#======================================
