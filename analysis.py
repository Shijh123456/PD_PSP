import os, shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib
from hmmleida import DataLoad

atlas = 'schaefer400'
method = 'hmm'
num_states = 20
abnormal_state = 8
input_dir = f'/data/sjh/PD_PSP_input_{atlas}'
result_dir = f'/home/sjh/snap/snapd-desktop-integration/common/PD_PSP/result/{method}_result/results_{atlas}_age'
analysis_dir = f'/home/sjh/snap/snapd-desktop-integration/common/PD_PSP/result/analysis_result/{method}/analysis_{atlas}_age/k={num_states}'
if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)

pl = DataLoad(data_path=input_dir, results_path=result_dir)


def f1(sub):
    save_path = f'{analysis_dir}/1.plot_states_in_bold:{sub}.png'
    pl.plot_states_in_bold(f'{sub}', k=num_states, alpha=0.5, darkstyle=False,save_path=save_path)
def f2():
    save_path = f'{analysis_dir}/2.barplot_centroids.png'
    pl.barplot_centroids(k=num_states, state='all',save_path=save_path)
def f3():
    save_path = f'{analysis_dir}'
    pl.group_transitions(k=num_states, metric='mean', cmap='YlGnBu', darkstyle=False,save_path=save_path)
def f4():
    for i in range (num_states):
        save_path = f'{analysis_dir}/4.overlap_withyeo_state{i+1}_all.png'
#        pl.overlap_withyeo(parcellation="D:/。/schaefer100MNI.nii.gz", n_areas=100, k=num_states,
 #                          state=i+1, darkstyle=False,
  #                         save_path=save_path)
        pl.overlap_withyeo(parcellation=f"/home/sjh/snap/snapd-desktop-integration/common/PD/atlas/{atlas}/{atlas}MNI.nii.gz",
                           n_areas=100, k=num_states, state=i+1, darkstyle=False,save_path=save_path,set_negative_zero=False)

def f5():
    groups = ['HC','PSP','PD']
    for group in groups:
        save_path = f'{analysis_dir}'
        pl.group_static_fc(group=group, plot=True, cmap='jet', darkstyle=False, save_path=save_path)

def f6():
    groups = ['HC', 'PSP', 'PD']
    for group in groups:
        save_path = f'{analysis_dir}'
        pl.group_dynamics_fc(k=num_states,group=group, plot=True, cmap='coolwarm', darkstyle=False, save_path=save_path,atlas=atlas)

def f7():
    for i in range (num_states):
        save_path = f'{analysis_dir}/7.explore_state_{i+1}.png'
        pl.explore_state(k=num_states, state=i+1, darkstyle=False, save_path=save_path)

def f8():
    # save_path = f"{analysis_dir}/8.states_network_glass.png"
    pl.plot_states_network_glass(k=num_states, darkstyle=False, save_path=f"{analysis_dir}/8.states_network_glass.png")
    
def f9():
    pl.plot_state_dwell_occupy(k=num_states,darkstyle=False,save_path=f"{analysis_dir}/9.dwell_occupy_1.png",psp_jia=True,abnormal_state=abnormal_state)
    

#f1('sub-013')
f2()
f4()
#f6()

