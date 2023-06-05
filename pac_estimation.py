import numpy as np
import pandas as pd

#!pip install pyEDFlib
#import pyedflib
#!pip install ipympl

from scipy.fftpack import fft, ifft, fftfreq
from scipy import signal
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from scipy.stats import binned_statistic, entropy, norm
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

import sys
import os
import time
#import pickle
import dill as pickle

import concurrent.futures
from numba import njit, prange

from tqdm.notebook import tqdm
from collections import defaultdict
import itertools

import matplotlib.pyplot as plt
import seaborn as sns
#from IPython.display import display
sns.set(context='notebook', style='ticks', palette='bright', font='sans-serif', font_scale=1, color_codes=True, rc=None)
plt.rcParams['figure.figsize'] = (14, 8)

#print("SYS.PATH: ", sys.path[:3])
#sys.path.insert(0, r"C:\Users\User\[[Python]]\[AlexeyT]\PAC_PROJECT")

from utility_functions import *
from lfp_class import LFP
from pac_class import MyPAC
from patient_class import Patient

print("Succesfully imported libraries and modules\n")

def main():
    
    # 1) Load .pkl file -> Patient

    #pickle_filepath = r"H:\Alexey_Timchenko\PAC\Patient1\Patient1.pkl"
    patient_number = input("Enter patient number: ")
    patient_name = "Patient" + patient_number
    
    confirm = input(f"Confirm PAC estimation for {patient_name}? y/n: ") == 'y'
    
    if not confirm:
        print("Exiting...")
        return
    
    compute_diag = input(f"Compute (normal) PAC for {patient_name}? y/n: ") == 'y'
    compute_nondiag = input(f"Compute cross-electrode PAC for {patient_name}? y/n: ") == 'y'
    if_multiprocess = input("Use concurrent.futures multiprocessing? (100% LOADS CPU) y/n: ") == 'y'
    

    with open("path_data.txt") as f:
        data_dir = f.readline()
    
    patient_pickle_filepath = os.path.join(data_dir, patient_name, patient_name + '.pkl')
    patient = load_patient_from_pickle(patient_pickle_filepath)
    
    patient.root_dir = os.path.join(data_dir, patient_name) # changing root_dir if calculation is done on another device

    print(f"Loaded {patient.name}")

    print("Conditions: \n") 
    print(*[s for s in patient.conditions if '180sec' in s], sep='\n')
    print("Placements: \n") 
    print(*patient.sorted_placements, sep='\n')

    # 2) Preprocess MyPAC objects to leave only fit_surrogates method to estimate
    conditions = ["1Day OFF Rest 180sec", "1Day ON Rest 180sec", "5Day OFF Rest 180sec", "5Day ON Rest 180sec",
                  "1Day OFF RH Move 180sec", "1Day OFF LH Move 180sec", "1Day ON RH Move 180sec", "1Day ON LH Move 180sec",
                  "5Day OFF RH Move 180sec", "5Day OFF LH Move 180sec", "5Day ON RH Move 180sec", "5Day ON LH Move 180sec"]
    
    # checking if these conditions exist in Patient instance (some of them are not present, e.g. Patient2)
    conditions_2use = []
    for i in range(len(conditions)):
        if conditions[i] in patient.conditions:
            conditions_2use.append(conditions[i])
    
    placements = patient.placements
    
    cross_placements = ["L4-3A", "L4-3B", "L4-3C", "L2A-3A", "L2B-3B", "L2C-3C", "L1-2A", "L1-2B", "L1-2C",\
                        "R4-3A", "R4-3B", "R4-3C", "R2A-3A", "R2B-3B", "R2C-3C", "R1-2A", "R1-2B", "R1-2C"]
    
    # Counting the number of total cross-electrode PACs to estimate
    n_left_cross = 0
    for placement_phase in cross_placements:
        for placement_amplitude in cross_placements:
            for condition in conditions_2use:
                if "Rest" not in condition:
                    continue
                # Counting only cross
                if placement_phase == placement_amplitude:
                    continue
                lfp_phase = patient.lfp[condition][placement_phase]
                lfp_amplitude = patient.lfp[condition][placement_amplitude]  
                pac_filename = create_pac_name(lfp_phase, lfp_amplitude) + ".pkl"
                if os.path.isfile(os.path.join(patient.root_dir, "pac", pac_filename)):
                    continue     
                n_left_cross += 1
                
    n_left_diag = 0
    for placement_phase in patient.sorted_placements:
        for placement_amplitude in patient.sorted_placements:
            for condition in conditions_2use:
                if "Rest" not in condition:
                    continue
                # counting only intra-electrode PACs
                if placement_phase != placement_amplitude:
                    continue
                lfp_phase = patient.lfp[condition][placement_phase]
                lfp_amplitude = patient.lfp[condition][placement_amplitude]  
                pac_filename = create_pac_name(lfp_phase, lfp_amplitude) + ".pkl"
                if os.path.isfile(os.path.join(patient.root_dir, "pac", pac_filename)):
                    continue     
                n_left_diag += 1
                
    n_left_total = n_left_diag + n_left_cross
    
    counter_total = 0
    counter_diag = 0
    
    # first diagonal ones:
    if compute_diag:
        print("Computing intra-electrode PACs...")
        # DIAGONAL ELEMENTS (ONE ELECTRODE PAC)
        for condition in conditions_2use:
            print("Condition: ", condition)
            for placement_phase in placements:
                for placement_amplitude in placements:
                    #print(f"Placements: {placement_phase} - {placement_amplitude}")
                    # if phase is on the right "R" and ampl is "L" do not calculate PAC
                    if placement_phase[0] != placement_amplitude[0]:
                        continue 

                    # SKIP NON-DIAGONAL (INTER-ELECTRODE) ELEMENTS
                    if placement_phase != placement_amplitude:
                        continue
                    
                    lfp_phase = patient.lfp[condition][placement_phase]
                    lfp_amplitude = patient.lfp[condition][placement_amplitude]
                    
                    # checking if file already exists
                    pac_filename = create_pac_name(lfp_phase, lfp_amplitude) + ".pkl"
                    print(pac_filename)
                    if os.path.isfile(os.path.join(patient.root_dir, "pac", pac_filename)):
                        print(f"{pac_filename} already exists")
                        continue
                        
                    # pac calculation
                    t0 = time.perf_counter()  
                    pac = MyPAC(beta_params=(10, 36, 1, 2), hfo_params=(140, 500, 20, 0), verbose=True, multiprocess=bool(if_multiprocess), use_numba=True)
                    pac.filter_fit_surrogates(lfp_phase, lfp_amplitude, n_surrogates=700, n_splits=1)
                    print(f"Surrogate estimation completed in {round(time.perf_counter() - t0)}")
                    pac.save(patient.root_dir)
                    
                    counter_diag += 1
                    counter_total += 1
                
                    print(f"Calculated {counter_total} / {n_left_total} PACs ")
                    estimation_time = round(time.perf_counter() - t0)
                    minutes_left = (int(estimation_time) * (n_left_total - counter_total)) // 60
                    print(f"Approximate total time left: {minutes_left // 60} H {minutes_left % 60} min")
                    
                    
                    
    if not compute_nondiag:
        print("Exiting...")
        return

    # NON-DIAGONAL ELEMENTS (CROSS-ELECTRODE PAC)
    print("Starting estimating cross-electrode PAC...")
    
    print(f"Calculated {counter_total} / {n_left_total} PACs ")
    
    for placement_phase in cross_placements:
        for placement_amplitude in cross_placements:
            for condition in conditions_2use:
                
                # only considering Rest OFF vs ON
                if "Rest" not in condition:
                    continue
                
                # if phase is on the right "R" and ampl is "L" do not calculate PAC
                if placement_phase[0] != placement_amplitude[0]:
                    continue
                
                # checking if it is same-electrode PAC
                if placement_phase == placement_amplitude:
                    continue
                
                lfp_phase = patient.lfp[condition][placement_phase]
                lfp_amplitude = patient.lfp[condition][placement_amplitude]
                
                # checking if file already exists
                pac_filename = create_pac_name(lfp_phase, lfp_amplitude) + ".pkl"
                print(pac_filename)
                if os.path.isfile(os.path.join(patient.root_dir, "pac", pac_filename)):
                    print(f"{pac_filename} already exists")
                    continue
                    
                # pac calculation
                t0 = time.perf_counter()
                """ATTENTION! CHANGED PARAMS FOR CROSS-ELECTRODE PAC ESTIMATION FOR FASTER PERFORMANCE"""
                pac = MyPAC(beta_params=(10, 36, 1, 2), hfo_params=(140, 500, 20, 0), verbose=True, multiprocess=bool(if_multiprocess), use_numba=True)
                pac.filter_fit_surrogates(lfp_phase, lfp_amplitude, n_surrogates=700, n_splits=1)
                print(f"Surrogate estimation completed in {round(time.perf_counter() - t0)}")
                pac.save(patient.root_dir)
                
                counter_cross += 1
                counter_total += 1
                
                print(f"Calculated {counter_total} / {n_left_total} PACs ")
                estimation_time = round(time.perf_counter() - t0)
                minutes_left = (int(estimation_time) * (n_left_total - counter_total)) // 60
                print(f"Approximate time left: {minutes_left // 60} H {minutes_left % 60} min")

if __name__ == '__main__':
    main()