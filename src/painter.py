import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
from metrix import * 

#===========================================================
def getMembersPc(pc_text_file: Path, members_number):
    pc1_all = []
    pc2_all = []
#Read data
    with open(pc_text_file) as f1:
        for line in f1:
            if line != ['']:
                line = line.split(",")
                pc1_all.append(float(line[0]))
                pc2_all.append(float(line[1]))
    pc1_all_memb = np.array_split(pc1_all, members_number)
    pc2_all_memb = np.array_split(pc2_all, members_number)
    return pc1_all_memb, pc2_all_memb

#===========================================================
def rotate_vector(data, angle):
    theta = np.radians(angle)
    co = np.cos(theta)
    si = np.sin(theta)
    rotation_matrix = np.array(((co, -si), (si, co)))
    rotated_vector = data.dot(rotation_matrix)
    return rotated_vector

#===========================================================
def form_data(arr1, arr2):
  data =[]
  for i in range(len(arr1)):
    data.append([arr1[i], arr2[i]])
  return np.array(data)


#===========================================================
def drawPc_OLD(pc_text_file: Path, pc_png_file: Path, inverse1 = 1, inverse2 = 1): # 3moths
    pc1 = []
    pc2 = []
    days = np.arange(0, 93, 1, dtype=int)
    #  PCs2     PsCs
    print("path pc: ", pc_text_file)

    with open(pc_text_file) as f1:
        next(f1)
        for line in f1:
            if line != ['']:
                line = line.split(",")
                pc1.append(inverse1 * float(line[0]))
                pc2.append(inverse2 * float(line[1]))

    # plt.figure(figsize=(6,6))
    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(6)
    jan_arr_pc1 = pc1[0:31]
    feb_arr_pc1 = pc1[31:59]
    mar_arr_pc1 = pc1[59:90]
    jan_arr_pc2 = pc2[0:31]
    feb_arr_pc2 = pc2[31:59]
    mar_arr_pc2 = pc2[59:90]
    text31 = np.arange(1, 32, 1, dtype=int)
    len_mar = 28
    text28 = np.arange(1, 29, 1, dtype=int)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('$RMM1$')
    plt.ylabel('$RMM2$')

# #ok
    rotated_data = rotate_vector(form_data(jan_arr_pc1, jan_arr_pc2), 0) # 0 -> -10
    plt.plot(rotated_data[:, 0], rotated_data[:, 1], '-o', color='red', ms=2, label='jan', linewidth=2)
    plt.annotate("START", (rotated_data[:,0][0], rotated_data[:,1][0] + 0.2), fontsize=8)
    for i in range(1,len(text31)):
        plt.annotate(text31[i], (rotated_data[:,0][i], rotated_data[:,1][i] + 0.2), fontsize=5)

    plt.plot([jan_arr_pc1[-1], feb_arr_pc1[0]], [jan_arr_pc2[-1], feb_arr_pc2[0]], '-', color='chartreuse', ms=2, linewidth=2)
    rotated_data = rotate_vector(form_data(feb_arr_pc1, feb_arr_pc2), 0)

    plt.plot(rotated_data[:,0], rotated_data[:,1],'-o', color='chartreuse', ms=2, label='feb')
    for i in range(0,len_mar-1):#len(text31)-1):  
        plt.annotate(text31[i], (rotated_data[:,0][i], rotated_data[:,1][i] + 0.2), fontsize=5)

    plt.plot([feb_arr_pc1[-1], mar_arr_pc1[0]], [feb_arr_pc2[-1], mar_arr_pc2[0]], '-', color='blue', ms=2)
    rotated_data = rotate_vector(form_data(mar_arr_pc1, mar_arr_pc2), 0)
    # plt.plot(mar_arr_pc1, mar_arr_pc2, '-o', color='blue', ms=2, label='mar')
    plt.plot(rotated_data[:,0], rotated_data[:,1], '-o', color='blue', ms=2, label='mar')

    for i in range(0,len_mar-1):#len(text31)-1):  
        plt.annotate(text31[i], (rotated_data[:,0][i], rotated_data[:,1][i] + 0.2), fontsize=5)
    plt.annotate("FINISH", (mar_arr_pc1[-1], mar_arr_pc2[-1] + 0.2), fontsize=8)
    plt.legend()
    # print("coordinate of the las point:", mar_arr_pc1[-1], mar_arr_pc2[-1])
    
#add Circle
    circle1 = plt.Circle((0, 0), 1, color='k', fill=False, linewidth=1)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" )
    ax.plot([1, 0], [0, 1], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" )
    ax.plot([0, 1], [0.5, 0.5], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" )
    ax.plot([0.5, 0.5], [0, 1], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" )
    # plt.text(-0.6, 0, "Weak MJO", fontsize=10, weight='bold')
    ax.add_artist(circle1)

    is_exist = os.path.exists(pc_png_file)
    if not is_exist:
        os.makedirs(pc_png_file)
    fig_name = os.path.basename(pc_png_file)


    # plt.savefig(f'{pc_png_file}/{fig_name}_{inverse1}_{inverse2}.png')
    saveFig(pc_png_file, False)
    plt.close()

#===========================================================
def drawPc(pc_text_file: Path, pc_png_file: Path, inverse1 = 1, inverse2 = 1): #only one month
    pc1 = []
    pc2 = []
    print("path pc: ", pc_text_file)
    with open(pc_text_file) as f1:
        next(f1)
        for line in f1:
            if line != ['']:
                line = line.split(",")
                pc1.append(inverse1 * float(line[0]))
                pc2.append(inverse2 * float(line[1]))

    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(6)
    jan_arr_pc1 = pc1[0:31]
    jan_arr_pc2 = pc2[0:31]
    text31 = np.arange(1, 32, 1, dtype=int)
    len_mar = 28 #TODO check if month is march
    text28 = np.arange(1, 29, 1, dtype=int)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('$RMM1$')
    plt.ylabel('$RMM2$')

    rotated_data = form_data(jan_arr_pc1, jan_arr_pc2) # 0 -> -10
    plt.plot(rotated_data[:, 0], rotated_data[:, 1], '-o', color='red', ms=2, label='jan', linewidth=2)
    plt.annotate("START", (rotated_data[:,0][0], rotated_data[:,1][0] + 0.2), fontsize=8)
    for i in range(1,len(text31)):
        plt.annotate(text31[i], (rotated_data[:,0][i], rotated_data[:,1][i] + 0.2), fontsize=5)
    plt.legend()
# Add Circle
    circle1 = plt.Circle((0, 0), 1, color='k', fill=False, linewidth=1)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" )
    ax.plot([1, 0], [0, 1], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" )
    ax.plot([0, 1], [0.5, 0.5], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" )
    ax.plot([0.5, 0.5], [0, 1], transform=ax.transAxes, color='k', linewidth = 0.5, ls="--" )
    ax.add_artist(circle1)
    saveFig(pc_png_file, False)
    plt.close()

#===========================================================
def drawAllPc(pc_text_file: Path, pc_png_file: Path, members_number): #ALL participants of the ensemble
    print("drawAllPc")
    print("path pc: ", pc_text_file)
    pc1_all_memb, pc2_all_memb = getMembersPc(pc_text_file, members_number)
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
    text31 = np.arange(1, 31, 1, dtype=int)

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

# Save into the file
    saveFig(pc_png_file)
    
#===========================================================
def drawCor(pc_text_file: Path, pc_png_file: Path, members_number):
    print("drawCor")
    # print("members_number: ", members_number)
    print("path pc: ", pc_text_file)
    pc1_all_memb, pc2_all_memb = getMembersPc(pc_text_file, members_number)
    memb_to_draw = len(pc1_all_memb) - 1 # memb_to_draw; do not draw the last element - it's era
    corr = findCor(pc1_all_memb, pc2_all_memb, memb_to_draw)

    fig, ax = plt.subplots()
    plt.xlabel('days')
    plt.ylabel('Correlation')
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
    plt.plot(np.arange(len(corr)), corr, marker='.', color='black', ms=2.5, linewidth=1.2) # -1: start from the second day of plav 
    saveFig(pc_png_file)
    saveMetric("Cor", corr, pc_png_file)

#===========================================================
def drawRmse(pc_text_file: Path, pc_png_file: Path, members_number):
    print("drawRmse")
    print("path pc: ", pc_text_file)
    pc1_all_memb, pc2_all_memb = getMembersPc(pc_text_file, members_number)
    memb_to_draw = len(pc1_all_memb) - 1 # memb_to_draw; do not draw the last element - it's era
    rmse = findRmse(pc1_all_memb, pc2_all_memb, memb_to_draw)

    fig, ax = plt.subplots()
    plt.xlabel('days')
    plt.ylabel('RMSE')
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
    plt.plot(np.arange(len(rmse)), rmse, marker='.', color='black', ms=2.5,  linewidth=1.2)
    saveFig(pc_png_file)
    saveMetric("RMSE", rmse, pc_png_file)

#===========================================================
def drawMsss(pc_text_file: Path, pc_png_file: Path, members_number): 
    print("drawMsss")
    print("path pc: ", pc_text_file)
    pc1_all_memb, pc2_all_memb = getMembersPc(pc_text_file, members_number)
    memb_to_draw = len(pc1_all_memb) - 1 # memb_to_draw; do not draw the last element - it's era
    msss = findMsss(pc1_all_memb, pc2_all_memb, memb_to_draw)

    fig, ax = plt.subplots()
    plt.xlabel('days')
    plt.ylabel('MSSS')
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)
    plt.plot(np.arange(len(msss)), msss, marker='.', color='black', ms=2.5, linewidth=1.2)
    saveFig(pc_png_file)
    saveMetric("MSSS", msss, pc_png_file)

#===========================================================
def saveFig(pc_png_file, make_dir=False):
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

