from config import *

#======================================
def set_datetime(FIELD): 
    day_e_initital = 0
    date_end = ""
    day_delta = 0

    if calendar.isleap(int(config["year_e"])):
        day_delta = timedelta(days=0) 
    else:
        day_delta = timedelta(days=1)

    date_forc_end_ = date.fromisoformat(config["date_forc_end"])
    date_end = date_forc_end_ + day_delta 
    date_end = str(date_end).replace('-', '/')

    times = pd.date_range("1991-11-01", date_end, freq='D')# gmc
    # print("times: ", times)

    SLAV_DT = FIELD.assign_coords(time=times)
    SLAV_DT = SLAV_DT.sel(time=slice(f'{config["year_s"]}/{config["month_s"]}/{config["day_s"]}',f'{config["year_e"]}/{config["month_e"]}/{config["day_e"]}'))

    return SLAV_DT

#======================================
def get_average_gmc_memb(var_name): 
    VAR_arr = 0
    memb_num = 0 

    for memb in range(100):
        member = str(memb)
        
        ncfile = f'{GMC_DATA_PATH}/gmc{member}-{var_name}.nc'

        if (os.path.isfile(ncfile)):
            # print("Found : ", ncfile)
            VAR_SLAV = xr.open_dataset(ncfile).sel(lat=slice(latS,latN))

            VAR_SLAV = set_datetime(VAR_SLAV) # current

            VAR_arr += VAR_SLAV
            members += 1
        # else:
            # print("Error - no such file or directory: ", ncfile)

    # print("members found: ", memb_num)
    
    return VAR_arr / members

#======================================
def get_forc_data_gmc():
    print(' ----- Forecast data gmc -----')
    OLR_SLAV  = xr.open_dataset(f'{GMC_DATA_PATH}/gmc3-olr.nc').sel(lat=slice(latS,latN)).sel(time=slice(config["date_forc_start"],config["date_forc_end"]))
    U200_SLAV = xr.open_dataset(f'{GMC_DATA_PATH}/gmc3-u200.nc').sel(lat=slice(latS,latN)).sel(time=slice(config["date_forc_start"],config["date_forc_end"]))
    U850_SLAV = xr.open_dataset(f'{GMC_DATA_PATH}/gmc3-u850.nc').sel(lat=slice(latS,latN)).sel(time=slice(config["date_forc_start"],config["date_forc_end"]))

    # OR 

    # OLR_SLAV =  get_average_gmc_memb("olr")
    # U200_SLAV = get_average_gmc_memb("u200")
    # U850_SLAV = get_average_gmc_memb("u850")

    return OLR_SLAV, U200_SLAV, U850_SLAV
