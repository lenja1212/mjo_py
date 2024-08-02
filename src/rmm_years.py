import netCDF4
from eofs.multivariate.standard import MultivariateEof
from eofs.standard import Eof
from painter import *
from config import *
from nc_funcs import * 
from era.era_funcs import * 
from slav.slav_funcs import * 
from gmc.gmc_funcs import * 

#================== Climatology ==========
OLR_ERA_clim, U200_ERA_clim, U850_ERA_clim = find_era_climatology()
# OLR_SLAV_clim, U200_SLAV_clim, U850_SLAV_clim = find_slav_climatology()
#==================

for year in years:
    print("Year: ", year)
    # if year == "2000": #only for slav 00 member 
        # continue

    # update_config(year) # OR get_config()
    get_config_year() # OR update_config()
    create_dirs()
    
    ps1_bur, ps2_bur = get_bureau_pc() #Bureau

    #================== Reference data (era5) ==========
    OLR_REF, U200_REF, U850_REF = get_ref_data_era();

    #================== "Forecast" data  ============== 
    # print(' ----- Daily mean ----- ')
    OLR_FORC, U200_FORC, U850_FORC = get_forc_data_slav()
    # OLR_FORC, U200_FORC, U850_FORC = get_forc_data_gmc()
    # OLR_FORC, U200_FORC, U850_FORC = get_forc_data_era()

    #======================================
    OLR_anom  = find_anomaly(OLR_REF, OLR_FORC, OLR_ERA_clim)
    U200_anom = find_anomaly(U200_REF, U200_FORC, U200_ERA_clim)
    U850_anom = find_anomaly(U850_REF, U850_FORC, U850_ERA_clim)

    #======================================

    olr  = OLR_anom['olr']
    u850 = U850_anom['u']
    u200 = U200_anom['u']
    olr  = olr.mean('lat')
    u850 = u850.mean('lat')
    u200 = u200.mean('lat')

    # print(' ----- Calculate normalization factors ----- ') # Worse
    # olr_norm, u200_norm, u850_norm = calc_norm_factor(olr, u200, u850)
    #######

    olr  = olr/olr_norm #-1 for gmc data 
    u850 = u850/u850_norm
    u200 = u200/u200_norm

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
    all_members_dfs.append(df.head(31)) #60
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