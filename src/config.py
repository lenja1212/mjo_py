from datetime import datetime, timedelta, date
import calendar
import os
import numpy as np
import pandas as pd
import netCDF4
import xarray as xr
import xrscipy.other.signal as dsp

#======================================
def get_bureau_pc(): #get 31 bureau pc values
	print(' ----- Get Burreau PCs ----- ')
	D = np.array(np.loadtxt(f'{BUREAU_PATH}/rmm.74toRealtime.txt', skiprows=2, usecols=range(7)))
	years  = D[:,0]
	months = D[:,1]
	days   = D[:,2]
	rmm1   = D[:,3]
	rmm2   = D[:,4]
	ps1_bur = []
	ps2_bur = []

	start_date = datetime.strptime(f'{config["year_s"]}/{config["month_s"]}/{config["day_s"]}', "%Y/%m/%d")
	end_date = start_date + timedelta(days=60)
	year_bur_e  = str(end_date)[0:4] 
	month_bur_e = str(end_date)[5:7]
	day_bur_e   = str(end_date)[8:10]
	print("bureau date start: ", f'{config["year_s"]}-{config["month_s"]}-{config["day_s"]}' )
	print("bureau date end  : ", str(end_date)[0:10])

	for it in range(len(years)):
		curr_date =  datetime.strptime(f'{int(years[it])}/{int(months[it])}/{int(days[it])}', "%Y/%m/%d")
		if start_date <= curr_date <= end_date:
			ps1_bur.append(rmm1[it])
			ps2_bur.append(rmm2[it])
	return ps1_bur, ps2_bur


#======================================
def get_prev_119():
	date_forc = date.fromisoformat(config["date_forc_start"])
	day_delta_1 = timedelta(days=1) 
	day_delta_119 = timedelta(days=119)
	date_ref_start = date_forc - day_delta_119 
	date_ref_end = date_forc - day_delta_1 
	return date_ref_start.strftime("%Y-%m-%d"), date_ref_end.strftime("%Y-%m-%d")

#======================================
def update_config(year):
	days_in_nc = timedelta(days=119)
	config["date_forc_start"] = f'{year}-01-01' # {year}-01-01 {year}-11-01

	date_forc_start = date.fromisoformat(config["date_forc_start"])
	date_forc_end = date_forc_start + days_in_nc
	# year = int(year)+1

	config["date_forc_end"]   = str(date_forc_end).replace('/', '-') 
	config["one_year"]		  = False
	set_config()

#======================================
def get_config_year(): #for one year
	days_in_nc = timedelta(days=119)
	config["date_forc_start"] = "1993-01-01" 
	
	date_forc_start = date.fromisoformat(config["date_forc_start"])
	date_forc_end = date_forc_start + days_in_nc

	config["date_forc_end"]   = date_forc_end.strftime('%Y-%m-%d')
	config["one_year"]		  = True

	set_config()

#======================================
def set_config():
	config["date_ref_start"], config["date_ref_end"] = get_prev_119() 
	config["fcast_fn_y"] 	 = config["date_ref_end"][2:4]
	config["fcast_fn_m"] 	 = config["date_ref_end"][5:7]

	if(SLAV_2008):
		config["fcast_fn_d"] = "29"
	else:
		config["fcast_fn_d"] = "30"

	config["year_s"]  		 = config["date_forc_start"][0:4]
	config["month_s"] 		 = config["date_forc_start"][5:7]
	config["day_s"]   		 = config["date_forc_start"][8:10]
	config["year_e"]  		 = config["date_forc_end"][0:4]
	config["month_e"] 		 = config["date_forc_end"][5:7]
	config["day_e"]   		 = config["date_forc_end"][8:10]
	config["out_year_dir"]   = f'{output_path}/mjo-rmm_{config["fcast_fn_y"]}'
	config["pc_out_dir"]     = f'{output_path}/mjo-rmm_{config["fcast_fn_y"]}/pcs'
	config["pic_out_dir"]    = f'{output_path}/mjo-rmm_{config["fcast_fn_y"]}/pic'
	config["metrix_out_dir"] = f'{output_path}/mjo-rmm_{config["fcast_fn_y"]}/metrix'
	config["pcs_txt_file_all_memb"] = f'{config["pc_out_dir"]}/mjo-rmm_all_members'
	config["pic_out_file_all_memb"] = f'{config["pic_out_dir"]}/mjo-rmm_all_members_{config["fcast_fn_y"]}{config["fcast_fn_m"]}{config["fcast_fn_d"]}-slavALL3Memb'
	config["pic_out_file_cor"]      =      f'{config["metrix_out_dir"]}/mjo-rmm_cor_{config["fcast_fn_y"]}{config["fcast_fn_m"]}{config["fcast_fn_d"]}-slavALL3Memb'
	config["pic_out_file_rmse"]     =     f'{config["metrix_out_dir"]}/mjo-rmm_rmse_{config["fcast_fn_y"]}{config["fcast_fn_m"]}{config["fcast_fn_d"]}-slavALL3Memb'
	config["pic_out_file_msss"]     =     f'{config["metrix_out_dir"]}/mjo-rmm_msss_{config["fcast_fn_y"]}{config["fcast_fn_m"]}{config["fcast_fn_d"]}-slavALL3Memb'

	print("date_ref_start: ", config["date_ref_start"]) 
	print("date_ref_end: ", config["date_ref_end"])
	print("date_forc_start: ", config["date_forc_start"]) 
	print("date_forc_end: ", config["date_forc_end"]) 

#======================================
def create_dirs():
	if not os.path.exists(config["out_year_dir"]):
		os.makedirs(config["out_year_dir"])
	if not os.path.exists(config["pc_out_dir"]):
		os.makedirs(config["pc_out_dir"])
	if not os.path.exists(config["pic_out_dir"]):
		os.makedirs(config["pic_out_dir"])
	if not os.path.exists(config["metrix_out_dir"]):
		os.makedirs(config["metrix_out_dir"])

#================== GLOBAL CONSTS 
latS = -15.
latN = 15.

# Default values (can be recalculated)
olr_norm  =  15.1
u200_norm =  4.81
u850_norm =  1.81

hour_shifts = ["00", "06", "12", "18"]
var_arr = [ "olr", "u200hpa",  "u850hpa"]
years = ["1992", "1993", "1994", "1995", "1996", "1997", "1998", "1999",
		 "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009",
		 "2010", "2011", "2012", "2013", "2014", "2015", "2016"]

months_30 = ["01", "03", "05", "07", "08", "10", "12"]
months_31 = ["04", "06", "09", "11"]

date_clim_start='1979-09-03'
date_clim_end='2001-12-31'

f_low  = 1/100
f_high = 1/20

SLAV_2008 = False

nc_in_path  = "/home/leonid/Desktop/MSU/MJO/input/"
output_path = "/home/leonid/Desktop/MSU/MJO/output/"
BUREAU_PATH = nc_in_path
GMC_DATA_PATH = f'/home/leonid/Downloads/NDJFMA/'


nc_in_slav = f"{nc_in_path}/slav" #slav slav-2008
nc_in_era  = f"{nc_in_path}/era5"

slav_olr_path  = f"{nc_in_slav}/olr/"
slav_u200_path = f"{nc_in_slav}/u200hpa/"
slav_u850_path = f"{nc_in_slav}/u850hpa/"
era_olr_path   = f"{nc_in_era}/olr/"
era_u200_path  = f"{nc_in_era}/u200hpa/"
era_u850_path  = f"{nc_in_era}/u850hpa/"

config = {
	"date_forc_start" 	  	: "",
	"date_forc_end"   	  	: "",
	"fcast_fn_y"		  	: "",
	"fcast_fn_m"		  	: "",
	"fcast_fn_d"		  	: "",
	"year_s"   			  	: "",
	"month_s"  			  	: "",
	"day_s"    			  	: "",
	"year_e"   			  	: "",
	"month_e"  			  	: "",
	"day_e"    			  	: "",
	"out_year_dir"          : "",
	"pc_out_dir"            : "",
	"pic_out_dir"           : "",
	"metrix_out_dir"        : "",
	"pcs_txt_file_all_memb" : "",
	"pic_out_file_all_memb" : "",
	"pic_out_file_cor"      : "",
	"pic_out_file_rmse"     : "",
	"pic_out_file_msss"     : "",
	"date_ref_start" 	    : "",
	"date_ref_end" 		    : "",
	"one_year"				: ""
}

# #Bureau 2015-01-01  http://www.bom.gov.au/climate/mjo/
