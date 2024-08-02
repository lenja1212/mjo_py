# This program purpose is to read calculated gmc model RMMs
# and find, plot their metrix

import netCDF4
import matplotlib.pyplot as plt
from nc_funcs import * 
from pathlib import Path


members_number = 25 # 25 24-slav_123000_00
#======================================
def saveFig(pc_png_file, make_dir=True): #False
    fig_path = ""
    fig_name = os.path.basename(pc_png_file)
    if make_dir:
        exists = os.path.exists(pc_png_file)
        if not exists:
           os.makedirs(pc_png_file)
        fig_path = f'{pc_png_file}/{fig_name}.png'
    else:
        fig_path = f'{pc_png_file}.png'
    plt.savefig(fig_path)
    plt.close()
    print("Graph saved: ", fig_path)

#======================================
def saveMetric(var_name, var, path):
	metrix_path = ""
	df_name = os.path.basename(path)	
	df = pd.DataFrame({f'{var_name}': var})

	exists = os.path.exists(path)
	if not exists:
		os.makedirs(path)
	metrix_path = f'{path}/{df_name}'
	df.to_csv(f'{metrix_path}.txt', index=False, float_format="%.5f")
	print("saveMetric path: ", metrix_path)

#======================================
def get_bureau_pc(start_year, start_month, start_day, end_year, forc_days): #get 31 bureau pc values
	config["year_s"] = start_year
	config["month_s"] = start_month
	config["day_s"] = start_day
	print(' ----- Get Burreau PCs ----- ')
	D = np.array(np.loadtxt("rmm.74toRealtime.txt", skiprows=2, usecols=range(7)))
	years  = D[:,0]
	months = D[:,1]
	days   = D[:,2]
	rmm1   = D[:,3]
	rmm2   = D[:,4]
	ps1_bur = []
	ps2_bur = []

	start_date = datetime.strptime(f'{config["year_s"]}/{config["month_s"]}/{config["day_s"]}', "%Y/%m/%d")
	end_date = start_date + timedelta(forc_days) #30 61
	year_bur_e  = str(end_date)[0:4] 
	month_bur_e = str(end_date)[5:7]
	day_bur_e   = str(end_date)[8:10]
	print("bureau date start: ", f'{config["year_s"]}/{config["month_s"]}/{config["day_s"]}' )
	print("bureau date end  : ", str(end_date)[0:10])
	# print(start_date.day)
	for year in range(start_year, end_year+1):
		# if year == 2000:
			# continue
		for it in range(len(years)): #any column
			if( ( int(years[it]) == year )  and
				(months[it] == int(config["month_s"])) and 
				(days[it]   == int(config["day_s"]) )  
			):
				ps1_bur.append(rmm1[it:(it+forc_days)])
				ps2_bur.append(rmm2[it:(it+forc_days)])

	return ps1_bur, ps2_bur

#======================================
def getMembersPc(rmm1_path, rmm2_path):
	pc1_all = []
	pc2_all = []

	with open(rmm1_path) as f1: #HIN REA
	    for line in f1:
	        if line != ['']:
	            pc1_all.append(float(line))

	with open(rmm2_path) as f1: #HIN REA
	    for line in f1:
	        if line != ['']:
	            pc2_all.append(float(line))

	pc1_all_year_memb = np.array_split(pc1_all, members_number)
	pc2_all_year_memb = np.array_split(pc2_all, members_number)

	return pc1_all_year_memb, pc2_all_year_memb

#===========================================================
def getSlavMembersPc(start_year, end_year,forc_days):
	pc1_all = []
	pc2_all = []
	members_number = 0
	for year in range(start_year, end_year+1):
		# if year == 2000:
			# continue
		print("year: ", year)
		rmm_text_file = f'/home/leonid/Desktop/MSU/MJO/output/mjo-rmm_{str(year)[2:4]}/pcs/mjo-rmm_all_members.txt'
		with open(rmm_text_file) as f1:
			day = 0
			for line in f1:
				if day == forc_days:
					break
				if line != ['']:
					line = line.split(",")
					pc1_all.append( float(line[0]))
					pc2_all.append( float(line[1]))
				day += 1
		members_number += 1

	pc1_all_year_memb = np.array_split(pc1_all, members_number)
	pc2_all_year_memb = np.array_split(pc2_all, members_number)

	return pc1_all_year_memb, pc2_all_year_memb

#======================================
def findCor(pc1_all_memb, pc2_all_memb, ps1_bur, ps2_bur): ### W:  
	print("findCor")
	corr = []
	dayly_pc1 = np.array(pc1_all_memb)
	dayly_pc2 = np.array(pc2_all_memb)
	ps1_bur = np.array(ps1_bur)
	ps2_bur = np.array(ps2_bur)
	numenator_pc1 = np.sum(ps1_bur * dayly_pc1) 
	numenator_pc2 = np.sum(ps2_bur * dayly_pc2) 
	denom_1 = np.sqrt( np.sum( np.power(ps1_bur, 2) + np.power(ps2_bur, 2) ) )
	denom_2 = np.sqrt( np.sum( np.power(dayly_pc1, 2)  + np.power(dayly_pc2, 2) ) )

	corr = ( (numenator_pc1 + numenator_pc2) / (denom_1 * denom_2) )

	return corr

#======================================	
def findRmse(pc1_all_memb, pc2_all_memb, ps1_bur, ps2_bur): 
	print("findRmse")
	rmse = []
	dayly_pc1 = np.array(pc1_all_memb)
	dayly_pc2 = np.array(pc2_all_memb)
	delta_1 = ps1_bur - dayly_pc1
	delta_2 = ps2_bur - dayly_pc2
	numenator_1 = np.sum( np.power(delta_1, 2 ) )  
	numenator_2 = np.sum( np.power(delta_2, 2 ) ) 
		
	rmse =  np.sqrt( (numenator_1 + numenator_2) / len(dayly_pc1) )
	print("rmse: ", rmse)
	return rmse

#======================================
def findMsss(pc1_all_memb, pc2_all_memb, ps1_bur, ps2_bur):
	print("findMsss")
	msss = []
	mse_c = []
	ps1_bur = np.array(ps1_bur)
	ps2_bur = np.array(ps2_bur)
	rmse = findRmse(pc1_all_memb, pc2_all_memb, ps1_bur, ps2_bur)
	mse_f = np.power(rmse, 2)
	mse_c =  np.sum(  np.power(ps1_bur, 2)  + np.power(ps2_bur, 2) ) / len(pc1_all_memb)

	print("msss: ", 1 - mse_f / mse_c)

	msss = 1 - (mse_f / mse_c)

	return msss

#======================================
def drawCor(corr, pc_png_file: Path):
    print("drawCor")
    fig, ax = plt.subplots()
    plt.xlabel('days')
    plt.ylabel('Correlation')
    plt.axhline(y=0.6, color='black', linestyle='-', linewidth=0.5)
    plt.plot(np.arange(len(corr[0:31])), corr[0:31], marker='.', color='black', ms=2.5, linewidth=1.2) # -1: start from the second day of plav 
    saveFig(pc_png_file)
    saveMetric("Cor", corr, pc_png_file)

#===========================================================
def drawRmse(rmse, pc_png_file: Path):
    print("drawRmse")
    fig, ax = plt.subplots()
    plt.xlabel('days')
    plt.ylabel('RMSE')
    plt.axhline(y=1.4, color='black', linestyle='-', linewidth=0.5)
    plt.plot(np.arange(len(rmse[0:31])), rmse[0:31], marker='.', color='black', ms=2.5,  linewidth=1.2)
    saveFig(pc_png_file)
    saveMetric("RMSE", rmse, pc_png_file)

#===========================================================
def drawMsss(msss, pc_png_file: Path): 
    print("drawMsss")
    fig, ax = plt.subplots()
    plt.xlabel('days')
    plt.ylabel('MSSS')
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
    plt.plot(np.arange(len(msss[0:31])), msss[0:31], marker='.', color='black', ms=2.5, linewidth=1.2)
    saveFig(pc_png_file)
    saveMetric("MSSS", msss, pc_png_file)
#===========================================================

def drawPc(pc1_all_memb, pc2_all_memb):
	print("drawAllPc")
	# print("path pc: ", pc_text_file)

####### 
#TODO somehow add era
	memb_to_draw = len(pc1_all_memb) - 1 # memb_to_draw; do not draw the last element - it's era
# Prepare figure fields
	# fig, ax = plt.subplots(layout="constrained")
	plt.rc('legend', fontsize=9) # legend font size 
	fig, ax = plt.subplots()
	fig.set_figheight(6)
	fig.set_figwidth(6)
	plt.xlim(-4, 4)
	plt.ylim(-4, 4)
	plt.xlabel('$RMM1$')
	plt.ylabel('$RMM2$')
	text31 = np.arange(1, 61, 1, dtype=int)

# Find 25-75 and middle
	elements_pc1_max, elements_pc2_max, elements_pc1_min, elements_pc2_min, pc1_mean_arr, pc2_mean_arr = getMaxMinMedMemb(pc1_all_memb, pc2_all_memb, memb_to_draw)
# Draw max/min lines    
    # ax.plot(elements_pc1_max, elements_pc2_max, marker='.', color='green', ms=2.5, label='jan', linewidth=1.2)
    # ax.plot(elements_pc1_min, elements_pc2_min, marker='.', color='blue', ms=2.5, label='jan', linewidth=1.2)
# Draw middle line
	plt.plot(pc1_mean_arr, pc2_mean_arr, marker='.', color='black', ms=2.5, linewidth=1.2, label='Mean')
# Draw unpertubed ansamble member - (unshifted member)
	plt.plot(pc1_all_memb[0], pc2_all_memb[0], marker='.', color='red', ms=2.5, linewidth=1.2, label='Unperturbed')
# Draw last ansamble member - (controll ERA)
	plt.plot(pc1_all_memb[-1], pc2_all_memb[-1], marker='.', color='blue', ms=2.5, linewidth=1.2, label='Bureau')
# Draw all, but one members 
	for i in range(0, memb_to_draw):
		jan_arr_pc1 = pc1_all_memb[i]
		jan_arr_pc2 = pc2_all_memb[i]
	#Fill area between all members
		ax.fill(
			np.append(pc1_mean_arr, jan_arr_pc1[::-1]),
			np.append(pc2_mean_arr, jan_arr_pc2[::-1]),
			"lightgrey"
		)
    #Fill 50% middle
	ax.fill(
		np.append(elements_pc1_min, elements_pc1_max[::-1]),
		np.append(elements_pc2_min, elements_pc2_max[::-1]),
		"darkgray"
	)
	ax.fill(
		np.append(elements_pc1_min, pc1_mean_arr[::-1]),
		np.append(elements_pc2_min, pc2_mean_arr[::-1]),
		"darkgray"
	)
	ax.fill(
		np.append(elements_pc1_max, pc1_mean_arr[::-1]),
		np.append(elements_pc2_max, pc2_mean_arr[::-1]),
		"darkgray"
	)     
# Add dashed lines
	ax.plot([0, 0.414], [0, 0.414], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" )
	ax.plot([0.586, 1], [0.586, 1], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" )
	ax.plot([1, 0.586], [0, 0.414], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" )
	ax.plot([0.414, 0], [0.586, 1], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" )
	ax.plot([0, 0.375], [0.5, 0.5], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" )
	ax.plot([0.625, 1], [0.5, 0.5], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" )
	ax.plot([0.5, 0.5], [0, 0.375], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" )
	ax.plot([0.5, 0.5], [0.625, 1], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" )
# Add Circle
	circle1 = plt.Circle((0, 0), 1, color='k', fill=False, linewidth=1)
	ax.add_artist(circle1)
# Add phase numbers
	ax.text(-3.5, -1.5, "1", fontsize=12)
	ax.text(-1.5, -3.5, "2", fontsize=12)
	ax.text(1.5, -3.5, "3", fontsize=12)
	ax.text(3.5, -1.5, "4", fontsize=12)
	ax.text(3.5, 1.5, "5", fontsize=12)
	ax.text(1.5, 3.5, "6", fontsize=12)
	ax.text(-1.5, 3.5, "7", fontsize=12)
	ax.text(-3.5, 1.5, "8", fontsize=12)
# Add subtitles X,Y
	ax.text(-3.8, -0.6, "West. Hem. \n and Africa", rotation = 90, fontsize=12)
	ax.text(-0.5, 3.5, "Western\n Pacific", rotation = 0, rotation_mode='anchor', fontsize=12)
	ax.text(3.4, -0.5, " Maritime\nContinent", rotation = -90, fontsize=12)
	ax.text(-0.4, -3.9, "Indian\nOcean", rotation = 0, fontsize=12)
# Add date number
	plt.annotate("START", (pc1_mean_arr[0], pc2_mean_arr[0] + 0.1), fontsize=6)
	for i in range(1, len(text31), 2):#, 1): # Mark every 3 day as number
		plt.annotate(text31[i], (pc1_mean_arr[i], pc2_mean_arr[i] + 0.05), fontsize=5)
		plt.annotate(text31[i], (pc1_all_memb[-1][i], pc2_all_memb[-1][i] + 0.05), fontsize=5)

	legend = plt.legend( loc = "upper right", frameon = False) # No legend shift
	plt.tick_params(top = True, right = True, axis='both', direction='in')

	saveFig(f'/home/leonid/Desktop/MSU/MJO/output/gmc-{year}/gmc-rmmR')
	# plt.show()

# ======================= MAIN
hin_rmm1_path = "/home/leonid/Downloads/RMM1_HIN_1991+25.txt"
hin_rmm2_path = "/home/leonid/Downloads/RMM2_HIN_1991+25.txt"
rea_rmm1_path = "/home/leonid/Downloads/RMM1_REA_1991+25.txt"
rea_rmm2_path = "/home/leonid/Downloads/RMM2_REA_1991+25.txt"

start_year = 1991
end_year = 2015
start_month = 1# 10
start_day = 1# 31
forc_days = 31 #61
CORR = []
RMSE = []
MSSS = []

# pc1_all_year_memb, pc2_all_year_memb = getMembersPc(hin_rmm1_path, hin_rmm2_path)
# pc1_all_year_memb_ref, pc2_all_year_memb_ref = getMembersPc(rea_rmm1_path, rea_rmm2_path)

pc1_all_year_memb, pc2_all_year_memb = getSlavMembersPc(start_year, end_year, forc_days)
pc1_all_year_memb_ref, pc2_all_year_memb_ref = get_bureau_pc(start_year+1, start_month, start_day, end_year+1, forc_days) #Bureau


year_membrs = 0
for day in range(forc_days):
	print("day: ", day)
	pc1_all_memb = []
	pc2_all_memb = []
	pc1_all_memb_ref = []
	pc2_all_memb_ref = []

	for year in range(2016 - 1991): #2016 2015 - 1991 12300-slav
		pc1_memb = pc1_all_year_memb[year][day]
		pc2_memb = pc2_all_year_memb[year][day]
		pc1_all_memb.append(pc1_memb)
		pc2_all_memb.append(pc2_memb)

		pc1_memb_ref = pc1_all_year_memb_ref[year][day]
		pc2_memb_ref = pc2_all_year_memb_ref[year][day]
		pc1_all_memb_ref.append(pc1_memb_ref)
		pc2_all_memb_ref.append(pc2_memb_ref)

	CORR.append(findCor(pc1_all_memb, pc2_all_memb, pc1_all_memb_ref, pc2_all_memb_ref))
	RMSE.append(findRmse(pc1_all_memb, pc2_all_memb, pc1_all_memb_ref, pc2_all_memb_ref))
	MSSS.append(findMsss(pc1_all_memb, pc2_all_memb, pc1_all_memb_ref, pc2_all_memb_ref))
	print("=======================")
# drawPc(pc1_all_memb, pc2_all_memb)
# drawCor(CORR,  f'/home/leonid/Desktop/MSU/MJO/output/gmc-One-corr-slav') #gmcH REA ERA  slav
# drawRmse(RMSE, f'/home/leonid/Desktop/MSU/MJO/output/gmc-One-rmse-slav') #gmcH REA
# drawMsss(MSSS, f'/home/leonid/Desktop/MSU/MJO/output/gmc-One-msss-slav') #gmcH REA

drawCor(CORR,  f'/home/leonid/Desktop/MSU/MJO/output/era5-slav-corr-sh-all') # ERA  slav
drawRmse(RMSE, f'/home/leonid/Desktop/MSU/MJO/output/era5-slav-rmse-sh-all') # ERA  slav
drawMsss(MSSS, f'/home/leonid/Desktop/MSU/MJO/output/era5-slav-msss-sh-all') # ERA  slav
# ======================= 

