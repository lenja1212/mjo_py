import netCDF4
from eofs.multivariate.standard import MultivariateEof
from eofs.standard import Eof
from painter import *
from config import *
from nc_funcs import * 


OLR_clim, U200_clim, U850_clim = find_era_climatology()

# OLR_clim     = xr.open_dataset(f'{era_olr_path}/era5-olr-day-2p5grid-all.nc').sel(time=slice(date_clim_start,date_clim_end)).sel(lat=slice(latS,latN))
# U200_clim    = xr.open_dataset(f'{era_u200_path}/era5-u200hpa-day-2p5grid-all.nc').sel(time=slice(date_clim_start,date_clim_end)).sel(lat=slice(latS, latN))
# U850_clim    = xr.open_dataset(f'{era_u850_path}/era5-u850hpa-day-2p5grid-all.nc').sel(time=slice(date_clim_start,date_clim_end)).sel(lat=slice(latS, latN))

# print(' ----- Subtracting timemean from ERA ----- ')
# OLR_clim  = OLR_clim  - OLR_clim.mean("time") 
# U200_clim = U200_clim - U200_clim.mean("time")
# U850_clim = U850_clim - U850_clim.mean("time") 

# OLR_clim, U200_clim, U850_clim = find_climatology(OLR_clim, U200_clim, U850_clim)


for year in years:
    #======================================
    print("Year: ", year)
    # update_config(year) # OR get_config()
    get_config_year()
    create_dirs()
    ps1_bur, ps2_bur = get_bureau_pc() #Bureau

    #======================================
    ## ERA5 REF
    OLR_REF  	= xr.open_dataset(f'{era_olr_path}/era5-olr-day-2p5grid-all.nc').sel(time=slice(config["date_ref_start"],config["date_ref_end"])).sel(lat=slice(latS,latN))
    U200_REF 	= xr.open_dataset(f'{era_u200_path}/era5-u200hpa-day-2p5grid-all.nc').sel(time=slice(config["date_ref_start"],config["date_ref_end"])).sel(lat=slice(latS, latN))
    U850_REF 	= xr.open_dataset(f'{era_u850_path}/era5-u850hpa-day-2p5grid-all.nc').sel(time=slice(config["date_ref_start"],config["date_ref_end"])).sel(lat=slice(latS, latN))

    #======================================
    ## SLAV make suitable dates in slav.nc files to select dates further
    # print(' ----- Daily mean ----- ') # #SLAV 
    # OLR_SLAV1 =  average_ans_memb("olr", "00")
    # U200_SLAV1 = average_ans_memb("u200hpa", "00")
    # U850_SLAV1 = average_ans_memb("u850hpa", "00")

    # print(' ----- Inverse SLAV 2008 OLR  ----- ')
    # OLR_SLAV1['olr'] =  -1 * OLR_SLAV1['olr']

    ## -----     Test SLAV = ERA5
    print(' ----- Test SLAV = ERA5 ----- ') # 
    OLR_SLAV1   = xr.open_dataset(f'{era_olr_path}/era5-olr-day-2p5grid-all.nc').sel(time=slice(config["date_forc_start"],config["date_forc_end"])).sel(lat=slice(latS,latN))
    U200_SLAV1  = xr.open_dataset(f'{era_u200_path}/era5-u200hpa-day-2p5grid-all.nc').sel(time=slice(config["date_forc_start"],config["date_forc_end"])).sel(lat=slice(latS, latN))
    U850_SLAV1  = xr.open_dataset(f'{era_u850_path}/era5-u850hpa-day-2p5grid-all.nc').sel(time=slice(config["date_forc_start"],config["date_forc_end"])).sel(lat=slice(latS, latN))

    #====================================== #SLAV
    ## -----     Slav ensmean as salv clim
    # OLR_clim1   = xr.open_dataset(f'{slav_olr_path}/erfclim.1230.ensmean-olr.nc').sel(lat=slice(latS,latN))
    # U200_clim1  = xr.open_dataset(f'{slav_u200_path}/erfclim.1230.ensmean-u200hpa.nc').sel(lat=slice(latS, latN))
    # U850_clim1  = xr.open_dataset(f'{slav_u850_path}/erfclim.1230.ensmean-u850hpa.nc').sel(lat=slice(latS, latN))
    # OLR_clim1  = set_slav_datetime(OLR_clim1)  
    # U200_clim1 = set_slav_datetime(U200_clim1) 
    # U850_clim1 = set_slav_datetime(U850_clim1)

    # OR Find slav clim for all years but current

    # OLR_clim1, U200_clim1, U850_clim1   = find_slav_climatology()

    # OLR_clim1  = OLR_clim1.groupby('time.dayofyear').mean('time')
    # U200_clim1 = U200_clim1.groupby('time.dayofyear').mean('time')
    # U850_clim1 = U850_clim1.groupby('time.dayofyear').mean('time')


    # -----     Slav 2008 ensmean as salv clim ----- NOT USED
    ##

    #======================================
    print(' ----- Subtracting climatologies from ERA ----- ')
    OLR_REF_anom  = OLR_REF.groupby('time.dayofyear') - OLR_clim
    U200_REF_anom = U200_REF.groupby('time.dayofyear') - U200_clim
    U850_REF_anom = U850_REF.groupby('time.dayofyear') - U850_clim

    # print(' ----- Subtracting ERA climatologies from SLAV ----- ')
    OLR_SLAV1_anom  = OLR_SLAV1.groupby('time.dayofyear')  - OLR_clim  #ERA CLIM
    U200_SLAV1_anom = U200_SLAV1.groupby('time.dayofyear') - U200_clim #ERA CLIM
    U850_SLAV1_anom = U850_SLAV1.groupby('time.dayofyear') - U850_clim #ERA CLIM

    # print(' ----- Subtracting SLAV climatologies from SLAV ----- ')
    # OLR_SLAV1_anom  = OLR_SLAV1.groupby('time.dayofyear')  - OLR_clim1  #SLAV CLIM
    # U200_SLAV1_anom = U200_SLAV1.groupby('time.dayofyear') - U200_clim1 #SLAV CLIM
    # U850_SLAV1_anom = U850_SLAV1.groupby('time.dayofyear') - U850_clim1 #SLAV CLIM

    print(' ----- Merge ERA and SLAV ----- ')
    OLR_SLAV1_merg  = xr.concat([OLR_REF_anom,  OLR_SLAV1_anom],  "time") 
    U200_SLAV1_merg = xr.concat([U200_REF_anom, U200_SLAV1_anom], "time") 
    U850_SLAV1_merg = xr.concat([U850_REF_anom, U850_SLAV1_anom], "time") 

    print(' ----- Removing interannual variability (120d rolling mean) ----- ')
    OLR_SLAV1_anom1  = OLR_SLAV1_anom  -  OLR_SLAV1_merg.rolling(time=120, center=False).mean().dropna('time')
    U200_SLAV1_anom1 = U200_SLAV1_anom - U200_SLAV1_merg.rolling(time=120, center=False).mean().dropna('time')
    U850_SLAV1_anom1 = U850_SLAV1_anom - U850_SLAV1_merg.rolling(time=120, center=False).mean().dropna('time')
    #======================================

    olr  = OLR_SLAV1_anom1['olr']
    u850 = U850_SLAV1_anom1['u']
    u200 = U200_SLAV1_anom1['u']
    olr  = olr.mean('lat')
    u850 = u850.mean('lat')
    u200 = u200.mean('lat')

    # print(' ----- Calculate normalization factors ----- ')
    # olr_norm, u200_norm, u850_norm = calc_norm_factor(olr, u200, u850)
    #######

    olr  = olr/olr_norm
    u850 = u850/u850_norm
    u200 = u200/u200_norm

    # print(olr)
    
    solver = MultivariateEof([np.array(olr), np.array(u850), np.array(u200)], center=True)
    eof_list = solver.eofs(neofs=2, eofscaling=0) # CARL SHRECK
    pseudo_pcs = np.squeeze( solver.projectField([-1*np.array(olr), np.array(u850), np.array(u200)], eofscaling=1, neofs=2, weighted=False) )
    psc1, psc2 = [], []
    for pc in pseudo_pcs:
        psc1.append(pc[0])
        psc2.append(pc[1]) 

    #======================================
    pcstxtfile = f'{output_path}/member_123000-00'
    member_counter = 2
    all_members_dfs = [] 

    df = pd.DataFrame({"PC1": psc1, "PC2": psc2})
    df.to_csv(f'{pcstxtfile}.txt', index=False, float_format="%.5f")
    all_members_dfs.append(df.head(31))
    pcstxtfile = f'{output_path}/member_123000-99'
    df = pd.DataFrame({"PC1": ps1_bur, "PC2": ps2_bur})
    df.to_csv(f'{pcstxtfile}.txt', index=False, float_format="%.5f")
    all_members_dfs.append(df.head(31))

    with open(f'{config["pcs_txt_file_all_memb"]}.txt','w') as file: #Save all_members_dfs into file as dataframe  
        for dframe in all_members_dfs :
            dframe.to_csv(file, index=False, header=False)

    drawAllPc(f'{config["pcs_txt_file_all_memb"]}.txt', f'{config["pic_out_file_all_memb"]}', member_counter)
    drawCor(f'{config["pcs_txt_file_all_memb"]}.txt',   f'{config["pic_out_file_cor"]}',      member_counter)
    drawRmse(f'{config["pcs_txt_file_all_memb"]}.txt',  f'{config["pic_out_file_rmse"]}',     member_counter)
    drawMsss(f'{config["pcs_txt_file_all_memb"]}.txt',  f'{config["pic_out_file_msss"]}',     member_counter)

    print()
    if config["one_year"]:
        exit()
    #======================================

    