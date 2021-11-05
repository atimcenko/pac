import numpy as np
import pandas as pd

#!pip install pyEDFlib
import pyedflib
#!pip install ipympl

from scipy.fftpack import fft, ifft, fftfreq
from scipy import signal as sg
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

from tqdm.notebook import tqdm
from collections import defaultdict
import itertools

import matplotlib.pyplot as plt
import seaborn as sns
#from IPython.display import display
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
plt.rcParams['figure.figsize'] = (14, 8)

import sys
#print("SYS.PATH: ", sys.path[:3])
#sys.path.insert(0, r"C:\Users\User\[[Python]]\[AlexeyT]\PAC_PROJECT")

sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
plt.rcParams['figure.figsize'] = (14, 8)
from utility_functions import *
from lfp_class import LFP
from pac_class import MyPAC
from patient_class import Patient

print("Succesfully imported libraries and modules\n")

def main():
    
    # 1) Load .pkl file -> Patient

    pickle_filepath = r"H:\Alexey_Timchenko\PAC\Patient1\Patient1.pkl"
    p1 = load_patient_from_pickle(pickle_filepath)

    print(f"Loaded {p1.name}")

    print("Conditions: \n") 
    print(*[s for s in p1.conditions if '180sec' in s], sep='\n')
    print("Placements: \n") 
    print(*p1.placements, sep='\n')

    # 2) Preprocess MyPAC objects to leave only fit_surrogates method to estimate
    conditions = ["1Day OFF Rest 180sec", "1Day OFF RH Move 180sec", "1Day OFF LH Move 180sec", 
              "1Day ON Rest 180sec", "1Day ON RH Move 180sec", "1Day ON LH Move 180sec", 
              "5Day OFF Rest 180sec", "5Day OFF RH Move 180sec", "5Day OFF LH Move 180sec", 
              "5Day ON Rest 180sec", "5Day ON RH Move 180sec", "5Day ON LH Move 180sec"]
    conditions_rest = ["1Day OFF Rest 180sec", "1Day ON Rest 180sec", "5Day OFF Rest 180sec", "5Day ON Rest 180sec"] # "5Day OFF Rest 180sec", "5Day ON Rest 180sec"]
    placements = p1.placements

    # first diagonal ones:

    # DIAGONAL ELEMENTS (ONE ELECTRODE PAC)
    for condition in conditions:
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
                
                lfp_phase = p1.lfp[condition][placement_phase]
                lfp_amplitude = p1.lfp[condition][placement_amplitude]
                
                # checking if file already exists
                pac_filename = create_pac_name(lfp_phase, lfp_amplitude) + ".pkl"
                print(pac_filename)
                if os.path.isfile(os.path.join(p1.root_dir, "pac", pac_filename)):
                    print(f"{pac_filename} already exists")
                    continue
                    
                # pac calculation
                t0 = time.perf_counter()  
                pac = MyPAC(beta_params=(5, 48, 1, 2), hfo_params=(40, 500, 20, 0), verbose=True, multiprocess=False, use_numba=True)
                pac.filter_fit_surrogates(lfp_phase, lfp_amplitude, n_surrogates=700, n_splits=1)
                print(f"Surrogate estimation completed in {round(time.perf_counter() - t0)}")
                pac.save(p1.root_dir)

    # NON-DIAGONAL ELEMENTS (INTER-ELECTRODE PAC)
    print("Starting estimating inter-electrode PAC")
    for condition in conditions:
        for placement_phase in placements:
            for placement_amplitude in placements:
                
                # if phase is on the right "R" and ampl is "L" do not calculate PAC
                if placement_phase[0] != placement_amplitude[0]:
                    continue 
                
                lfp_phase = p1.lfp[condition][placement_phase]
                lfp_amplitude = p1.lfp[condition][placement_amplitude]
                
                # checking if file already exists
                pac_filename = create_pac_name(lfp_phase, lfp_amplitude) + ".pkl"
                print(pac_filename)
                if os.path.isfile(os.path.join(p1.root_dir, "pac", pac_filename)):
                    print(f"{pac_filename} already exists")
                    continue
                    
                # pac calculation
                t0 = time.perf_counter()  
                pac = MyPAC(beta_params=(5, 48, 1, 2), hfo_params=(40, 500, 20, 0), verbose=False, multiprocess=False, use_numba=True)
                pac.filter_fit_surrogates(lfp_phase, lfp_amplitude, n_surrogates=700, n_splits=1)
                print(f"Surrogate estimation completed in {round(time.perf_counter() - t0)}")
                pac.save(p1.root_dir)

if __name__ == '__main__':
    main()