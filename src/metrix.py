import numpy as np
import pandas as pd
from pathlib import Path

#======================================
def findCor(pc1_all_memb, pc2_all_memb, memb_to_draw): ### W:  
	# print("findCor")
	# print("memb_to_draw: ", memb_to_draw)
	corr = []
	pc_days = len(pc1_all_memb[0])
	for i in range(0, pc_days-1): # start from the second day of era
		dayly_pc1, dayly_pc2 = getDaylyPc(pc1_all_memb, pc2_all_memb, memb_to_draw, i)
		numenator_pc1 = pc1_all_memb[-1][i] * np.sum(dayly_pc1) #a1 * b1; pc1_all_memb[-1][i] - observed; main: RMM1 pc1_all_memb[-1][i] 
		numenator_pc2 = pc2_all_memb[-1][i] * np.sum(dayly_pc2) #a2 * b2; pc2_all_memb[-1][i] - observed; main: RMM2 pc2_all_memb[-1][i]
		denom_1 = np.sqrt( memb_to_draw * ( np.power(pc1_all_memb[-1][i], 2) + np.power(pc2_all_memb[-1][i], 2) ) )  
		denom_2 = np.sqrt( np.sum(np.power(dayly_pc1, 2)) + np.sum(np.power(dayly_pc2, 2)) )

		# print("i: ",i)
		# print("dayly_pc1: ",dayly_pc1)
		# print("dayly_pc2: ",dayly_pc2)
		# print("np.sum(dayly_pc1): ",np.sum(dayly_pc1))
		# print("np.sum(dayly_pc2): ",np.sum(dayly_pc2))
		# print("pc1_all_memb[-1][i]: ",pc1_all_memb[-1][i])
		# print("pc2_all_memb[-1][i]: ",pc2_all_memb[-1][i])
		# print("numenator_pc1: ", numenator_pc1)
		# print("numenator_pc2: ", numenator_pc2)
		# print("denom_1: ", denom_1)
		# print("denom_2: ", denom_2)
		# print("cor: ", (numenator_pc1 + numenator_pc2) / (denom_1 * denom_2))
		# print("============================")
		
		corr.append( (numenator_pc1 + numenator_pc2) / (denom_1 * denom_2) )
	return corr

#======================================	
def findRmse(pc1_all_memb, pc2_all_memb, memb_to_draw): 
	# print("findRmse")
	rmse = []
	# print("memb_to_draw: ",memb_to_draw)
	pc_days = len(pc1_all_memb[0])
	for i in range(0, pc_days-1): # start from the second day of era
		dayly_pc1, dayly_pc2 = getDaylyPc(pc1_all_memb, pc2_all_memb, memb_to_draw, i)
		delta_1 = pc1_all_memb[-1][i] - np.array(dayly_pc1)
		delta_2 = pc2_all_memb[-1][i] - np.array(dayly_pc2)
		numenator_1 = np.sum( np.power(delta_1, 2 ) )  
		numenator_2 = np.sum( np.power(delta_2, 2 ) ) 
		
		# print("i: ",i)
		# print("dayly_pc1: ",dayly_pc1)
		# print("dayly_pc2: ",dayly_pc2)
		# print("pc1_all_memb[-1][i]: ",pc1_all_memb[-1][i])
		# print("pc2_all_memb[-1][i]: ",pc2_all_memb[-1][i])
		# print("delta_1: ",delta_1)
		# print("delta_2: ",delta_2)
		# print("numenator_1: ",numenator_1)
		# print("numenator_2: ",numenator_2)
		# print("rmse: ",np.sqrt( (numenator_1 + numenator_2) / memb_to_draw ))
		# print("============================")

		rmse.append( np.sqrt( (numenator_1 + numenator_2) / memb_to_draw ) )
	return rmse

#======================================
def findMsss(pc1_all_memb, pc2_all_memb, memb_to_draw):
	# print("findMsss")
	msss = []
	mse_c = []
	pc_days = len(pc1_all_memb[0])
	rmse = findRmse(pc1_all_memb, pc2_all_memb, memb_to_draw)
	mse_f = np.power(rmse, 2)
	for i in range(0, pc_days-1): # days in pc # start from the second day of era
		mse_c_numen =  ( np.power(pc1_all_memb[-1][i], 2)  + np.power(pc2_all_memb[-1][i], 2) )
		mse_c.append(mse_c_numen ) 

		# print("i: ", i)	
		# print("a1: ", pc1_all_memb[-1][i])
		# print("a2: ", pc2_all_memb[-1][i])
		# print("rmse: ", rmse[i])
		# print("mse_f: ", mse_f[i])
		# print("mse_c: ", mse_c[i])
		# print("a/b: ", mse_f[i] / mse_c[i])
		# print("msss: ", 1 - mse_f[i] / mse_c[i])
		# print("-------------------")

	msss = 1 - mse_f / mse_c
	return msss

#======================================
def getMaxMinMedMemb(pc1_all_memb, pc2_all_memb, memb_to_draw): #!!!
	elements_pc1_max = []
	elements_pc1_min = []
	elements_pc2_max = []
	elements_pc2_min = []
	pc1_mean_arr = []
	pc2_mean_arr = []
	for i in range(0, len(pc1_all_memb[0])): #days to draw
		pc1_mean = 0 
		pc2_mean = 0
		dayly_pc1, dayly_pc2 = getDaylyPc(pc1_all_memb, pc2_all_memb, memb_to_draw, i)

		pc1_mean_arr.append(np.median(dayly_pc1))
		pc2_mean_arr.append(np.median(dayly_pc2))

		elements_pc1_max.append(np.percentile(dayly_pc1, 75))
		elements_pc1_min.append(np.percentile(dayly_pc1, 25))
		elements_pc2_max.append(np.percentile(dayly_pc2, 75))
		elements_pc2_min.append(np.percentile(dayly_pc2, 25))

	return elements_pc1_max, elements_pc2_max, elements_pc1_min, elements_pc2_min, pc1_mean_arr, pc2_mean_arr

#======================================
def getDaylyPc(pc1_all_memb, pc2_all_memb, memb_to_draw, day):
	dayly_pc1 = []
	dayly_pc2 = []
	for j in range(0, memb_to_draw): # i-th day in all months  #len(pc1_all_memb)
		dayly_pc1.append(pc1_all_memb[j][day])
		dayly_pc2.append(pc2_all_memb[j][day])
	return dayly_pc1, dayly_pc2

#======================================
def saveMetric(var_name, var, path):
	print("saveMetric path: ", path)
	df = pd.DataFrame({f'{var_name}': var})
	df.to_csv(f'{path}.txt', index=False, float_format="%.5f")

#======================================    