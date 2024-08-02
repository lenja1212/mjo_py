from config import *

#======================================
def find_slav_clim_ex_cur():
	print(' ----- Computing SLAV climatologies ----- ')
	OLR_clim = 0
	U200_clim = 0
	U850_clim = 0
	memb_num = 0

	for year in years:
		year = str(int(year) - 1)[2:4]

		if year != config["fcast_fn_y"]:
			for memb in range(100):
				member = str(memb)
				if memb < 10:
					member = "0" + member
				for hour in hour_shifts:
					ncfile_olr = f'{nc_in_slav}/olr/erfclim.{year}{config["fcast_fn_m"]}{config["fcast_fn_d"]}{hour}_{member}-olr.nc' # all members
					ncfile_u200 = f'{nc_in_slav}/u200hpa/erfclim.{year}{config["fcast_fn_m"]}{config["fcast_fn_d"]}{hour}_{member}-u200hpa.nc' # all members
					ncfile_u850 = f'{nc_in_slav}/u850hpa/erfclim.{year}{config["fcast_fn_m"]}{config["fcast_fn_d"]}{hour}_{member}-u850hpa.nc' # all members
					
					if (os.path.isfile(ncfile_olr) and os.path.isfile(ncfile_u200) and os.path.isfile(ncfile_u850)):
						# print("Found : ", ncfile_olr)
						
						OLR = xr.open_dataset(ncfile_olr).sel(lat=slice(latS,latN))
						U200 = xr.open_dataset(ncfile_u200).sel(lat=slice(latS,latN))
						U850 = xr.open_dataset(ncfile_u850).sel(lat=slice(latS,latN))
						
						OLR = set_slav_datetime(OLR) 
						U200 = set_slav_datetime(U200)
						U850 = set_slav_datetime(U850)

						OLR_clim += OLR
						U200_clim += U200
						U850_clim += U850
						memb_num += 1
					# else:
						# print("Error - no such file or directory: ", ncfile_olr)

	return OLR_clim/memb_num, U200_clim/memb_num, U850_clim/memb_num

#======================================
def find_slav_climatology(all = True):
	if(all): # Slav ensmean as salv clim
		OLR_clim   = xr.open_dataset(f'{slav_olr_path}/erfclim.1230.ensmean-olr.nc').sel(lat=slice(latS,latN))
		U200_clim  = xr.open_dataset(f'{slav_u200_path}/erfclim.1230.ensmean-u200hpa.nc').sel(lat=slice(latS, latN))
		U850_clim  = xr.open_dataset(f'{slav_u850_path}/erfclim.1230.ensmean-u850hpa.nc').sel(lat=slice(latS, latN))

		OLR_clim  = set_slav_datetime(OLR_clim)  
		U200_clim = set_slav_datetime(U200_clim) 
		U850_clim = set_slav_datetime(U850_clim)
	
	else: # Find slav clim for all years but current (config)
		OLR_clim, U200_clim, U850_clim   = find_slav_clim_ex_cur()

	OLR_clim  = OLR_clim.groupby('time.dayofyear').mean('time')
	U200_clim = U200_clim.groupby('time.dayofyear').mean('time')
	U850_clim = U850_clim.groupby('time.dayofyear').mean('time')

	return OLR_SLAV_clim, U200_SLAV_clim, U850_SLAV_clim

#======================================
def get_average_slav_memb(var_name, ans_memb = 0): 
	VAR_arr = 0
	memb_num = 0 
	ncfile = 0
	
	for memb in range(100):
		member = str(memb)
		if memb < 10: 
			member = "0" + member

		# hour_shifts = ["00"] # TODO func param
		hour_shifts = ["00", "06", "12", "18"]
		
		for hour in hour_shifts:
			if(SLAV_2008):
				fcast_fn_d_i = int(fcast_fn_d)-1
				ncfile = f'{nc_in_slav}/{var_name}/erfclim.{fcast_fn_y}{fcast_fn_m}{str(fcast_fn_d_i)}{hour}_{member}-{var_name}.nc' # 2008 all members
			else:
				ncfile = f'{nc_in_slav}/{var_name}/erfclim.{config["fcast_fn_y"]}{config["fcast_fn_m"]}{config["fcast_fn_d"]}{hour}_{member}-{var_name}.nc' # all members

			if (os.path.isfile(ncfile)):
				print("Found : ", ncfile)
				VAR_SLAV = xr.open_dataset(ncfile).sel(lat=slice(latS,latN))

				if(SLAV_2008):
					VAR_SLAV = set_slav_datetime(VAR_SLAV, True)
				else:
					VAR_SLAV = set_slav_datetime(VAR_SLAV)


				VAR_arr += VAR_SLAV
				memb_num += 1
			else:
				print("Error - no such file or directory: ", ncfile)

	# print("members found: ", memb_num)

	return VAR_arr / memb_num

#======================================
def get_one_slav_memb(var_name, hour, ans_memb = 0): 
	ncfile = f'{nc_in_slav}/{var_name}/erfclim.{config["fcast_fn_y"]}{config["fcast_fn_m"]}{config["fcast_fn_d"]}{hour}_{ans_memb}-{var_name}.nc' # one member
	
	if (os.path.isfile(ncfile)):
		VAR_SLAV = xr.open_dataset(ncfile).sel(lat=slice(latS,latN))

		if(SLAV_2008):
			VAR_SLAV = set_slav_datetime(VAR_SLAV, True)
		else:
			VAR_SLAV = set_slav_datetime(VAR_SLAV)

	else:
		print("Error - no such file or directory: ", ncfile)

	return VAR_SLAV 

#======================================
def set_slav_datetime(SLAV_FIELD): 
	day_e_initital = 0
	date_end = ""
	day_delta = timedelta(days=0) 

	if SLAV_2008 :
		if calendar.isleap(int(config["year_e"])):
			day_delta = timedelta(days=13)
		else:
			day_delta = timedelta(days=14)

	date_forc_end = date.fromisoformat(config["date_forc_end"])
	date_end = date_forc_end + day_delta 
	date_end = str(date_end).replace('-', '/')

	times = pd.date_range(config["date_ref_end"], date_end, freq='D')#slav
	# print("times: ", times)
	
	SLAV_DT = SLAV_FIELD.assign_coords(time=times)
	SLAV_DT = SLAV_DT.sel(time=slice(f'{config["year_s"]}/{config["month_s"]}/{config["day_s"]}',f'{config["year_e"]}/{config["month_e"]}/{config["day_e"]}'))
	
	return SLAV_DT

#======================================
def get_forc_data_slav(mean = True):
	print(' ----- Forecast SLAV ----- ')

	if(mean): # #SLAV  ALL members
		OLR_FORC =  get_average_slav_memb("olr", "00")
		U200_FORC = get_average_slav_memb("u200hpa", "00")
		U850_FORC = get_average_slav_memb("u850hpa", "00")
	
	else: # #SLAV  One member
		OLR_FORC =  get_one_slav_memb("olr", "00", "00")
		U200_FORC = get_one_slav_memb("u200hpa", "00", "00")
		U850_FORC = get_one_slav_memb("u850hpa", "00", "00")

	if(SLAV_2008):
		OLR_FORC['olr'] =  -1 * OLR_FORC['olr']

	return OLR_FORC, U200_FORC, U850_FORC