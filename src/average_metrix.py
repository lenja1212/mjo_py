from painter import *
from config import *
import numpy as np

# Cредняя корреляция + 50% перцентиль + 100% перцентиль
#======================================
def getAllMetrixMembers(var_name): #TODO cor -> var_name
	metrics = []
	var_num = 0
	for year in years:#[0:4]:
		year = int(year) - 1
		# if (year != 2010) and (year != 2011):
		print("year: ",year)
		metric_text_file = f'/home/leonid/Desktop/MSU/MJO/output/mjo-rmm_{str(year)[2:4]}/metrix/mjo-rmm_cor_{str(year)[2:4]}1230-slavALL20Memb.txt' # Era - Slav
		# metric_text_file = f'/home/leonid/Desktop/MSU/MJO/output/gmc-{year}/gmc-cor.txt' # Era - GMC HIN
		# metric_text_file = f'/home/leonid/Desktop/MSU/MJO/output/gmc-{year}/gmcR-cor.txt' # Era - GMC REA 
		print("path metric: ", metric_text_file)
		with open(metric_text_file) as f1:
			var_num += 1
			next(f1)
			for line in f1:
				if line != ['']:
					# print(line)
					metrics.append(float(line))

	print("var_num: ", var_num)
	metrics = np.array_split(metrics, var_num)
	return metrics

#======================================
def getMetricPercentile(metrics, percentile):
	perc_metrics = []
	for day in range(0, len(metrics[0])): # days to draw
		dayly_metric = getDaylyMetric(metrics, len(metrics), day)
		perc_metrics.append(np.percentile(dayly_metric, percentile))
		if day == 23:
			print("dayly_metric: ", dayly_metric)
			print("percentile: ", np.percentile(dayly_metric, percentile))
	return perc_metrics

#======================================
def getDaylyMetric(pc1_all_memb, memb_to_draw, day): # memb_to_draw = years
	dayly_metric = []
	for j in range(0, memb_to_draw): # i-th day in all months
		dayly_metric.append(pc1_all_memb[j][day])
	return dayly_metric

#======================================
def getMetricMean(metrics):
	mean_metric = []
	for day in range(0, len(metrics[0])): # days to draw
		dayly_metric = getDaylyMetric(metrics, len(metrics), day)
		mean_metric.append(np.mean(dayly_metric))
	return mean_metric

#======================================


#*********** Start date of ansamble members ***********
start_dates = ["1230"] #, "0730"] # add dates here 
for sdate in start_dates:
	cor_metrics = getAllMetrixMembers("cor") 
	metric_perc_50 = getMetricPercentile(cor_metrics, 50)
	metric_perc_100 = getMetricPercentile(cor_metrics, 100)
	mean_metric = getMetricMean(cor_metrics)

	fig, ax = plt.subplots()
	plt.xlim(0, 30)
	# plt.ylim(0.25, 1)
	plt.xlabel('days')
	plt.ylabel('Corr')
	for cor in cor_metrics:
		plt.plot(np.arange(len(cor)), cor, color='black', ms=2.5,  linewidth=0.5)

	# print(metric_perc_50[0:31])
	print(len(mean_metric))
	plt.plot(np.arange(len(mean_metric)), mean_metric, marker='.', color='red', ms=2.5,  linewidth=1.2, label='Mean')
	plt.plot(np.arange(len(metric_perc_50)), metric_perc_50, marker='.', color='blue', ms=2.5,  linewidth=1.2, label='50 percentile')
	plt.plot(np.arange(len(metric_perc_100)), metric_perc_100, marker='.', color='green', ms=2.5,  linewidth=1.2, label='100 percentile')

	plt.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
	# plt.axhline(y=0.6, color='black', linestyle='-', linewidth=0.5)
	plt.legend()
	metric_png_file = f'/home/leonid/Desktop/MSU/MJO/output/mjo-rmm_metrix_all/cor/{sdate}-era-slav-0' # Era - Slav
	# metric_png_file = f'/home/leonid/Desktop/MSU/MJO/output/mjo-rmm_metrix_all/cor/{sdate}-era-gmc' # Era - GMC gmc gmcR
	# metric_png_file = f'/home/leonid/Desktop/MSU/MJO/output/mjo-rmm_metrix_all/cor/{sdate}-era-gmcR' # Era - GMC gmc gmcR

	saveFig(metric_png_file)

#TODO dir creator 