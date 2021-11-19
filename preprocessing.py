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
sns.set(context='notebook', style='ticks', palette='bright', font='sans-serif', font_scale=1, color_codes=True, rc=None)
plt.rcParams['figure.figsize'] = (14, 8)
#print("SYS.PATH: ", sys.path[:3])
#sys.path.insert(0, r"C:\Users\User\[[Python]]\[AlexeyT]\PAC_PROJECT")

from utility_functions import *
from lfp_class import LFP
from pac_class import MyPAC
from patient_class import Patient

# %run utility_functions.py
# %run lfp_class.py
# %run pac_class.py
# %run patient_class.py


print("Succesfully imported libraries and modules\n")

def main():

    with open("path_data.txt") as f:
        data_dir = f.readline()
        
    patient_number = input("Enter patient number: ")
    patient_name = "Patient" + patient_number
    
    confirm = input(f"Confirm preprocessing for {patient_name}? y/n: ")
    
    if confirm != 'y':
        print("Exiting...")
        return

    patient_root_dir = os.path.join(data_dir, patient_name)
    #root_dir = r"H:\Alexey_Timchenko\PAC\Patient2"

    patient = Patient(name=patient_name, root_dir=patient_root_dir)
    files = patient.find_bdf_files()
    for filename in files:
        patient.scan_file_annotations(filename, update_file_conditions=True)

    patient.get_preprocessed_lfps(verbose=False)
    patient.display_all_annotations()
    
    if patient_name == 'Patient1':
        
        new_condition_name = "1Day OFF Rest 180sec"
        conditions_to_merge = ["1Day OFF Rest"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "1Day ON Rest 180sec"
        conditions_to_merge = ["1Day ON Rest"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "1Day OFF RH Move 180sec"
        conditions_to_merge = ["1Day OFF " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "1Day OFF LH Move 180sec"
        conditions_to_merge = ["1Day OFF " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "RHLH"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "1Day ON RH Move 180sec"
        conditions_to_merge = ["1Day ON " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "1Day ON LH Move 180sec"
        conditions_to_merge = ["1Day ON " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        ## 5 DAY

        new_condition_name = "5Day OFF Rest 180sec"
        conditions_to_merge = ["5Day OFF Rest"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "5Day ON Rest 180sec"
        conditions_to_merge = ["5Day ON Rest"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "5Day OFF RH Move 180sec"
        conditions_to_merge = ["5Day OFF " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "5Day OFF LH Move 180sec"
        conditions_to_merge = ["5Day OFF " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "5Day ON RH Move 180sec"
        conditions_to_merge = ["5Day ON " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "5Day ON LH Move 180sec"
        conditions_to_merge = ["5Day ON " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)
        
    if patient_name == 'Patient2':
        
        new_condition_name = "1Day OFF Rest 180sec"
        conditions_to_merge = ["1Day OFF Rest"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "1Day ON Rest 180sec"
        conditions_to_merge = ["1Day ON Rest"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "5Day OFF Rest 180sec"
        conditions_to_merge = ["5Day OFF Rest"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "5Day ON Rest 180sec"
        conditions_to_merge = ["5Day ON Rest"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "5Day OFF RH Move 180sec"
        conditions_to_merge = ["5Day OFF " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "5Day OFF LH Move 180sec"
        conditions_to_merge = ["5Day OFF " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "5Day ON RH Move 180sec"
        conditions_to_merge = ["5Day ON " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "5Day ON LH Move 180sec"
        conditions_to_merge = ["5Day ON " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)
        
    if patient_name == 'Patient3':
        
        new_condition_name = "1Day OFF Rest 180sec"
        conditions_to_merge = ["1Day OFF Rest"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "1Day ON Rest 180sec"
        conditions_to_merge = ["1Day ON Rest"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "1Day OFF RH Move 180sec"
        conditions_to_merge = ["1Day OFF " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "1Day OFF LH Move 180sec"
        conditions_to_merge = ["1Day OFF " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "1Day ON RH Move 180sec"
        conditions_to_merge = ["1Day ON " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "1Day ON LH Move 180sec"
        conditions_to_merge = ["1Day ON " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        ## 5 DAY

        new_condition_name = "5Day OFF Rest 180sec"
        conditions_to_merge = ["5Day OFF Rest"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "5Day ON Rest 180sec"
        conditions_to_merge = ["5Day ON Rest (240sec)"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "5Day OFF RH Move 180sec"
        conditions_to_merge = ["5Day OFF " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "5Day OFF LH Move 180sec"
        conditions_to_merge = ["5Day OFF " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "5Day ON RH Move 180sec"
        conditions_to_merge = ["5Day ON " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "5Day ON LH Move 180sec"
        conditions_to_merge = ["5Day ON " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

    if patient_name == 'Patient4':

        # For this patient 1Day without movement-related conditions (not enough data for 180sec)
        new_condition_name = "1Day OFF Rest 180sec"
        conditions_to_merge = ["1Day OFF Rest (2)", "1Day OFF Rest (3)"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "1Day ON Rest 180sec"
        conditions_to_merge = ["1Day ON Rest (1)"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "1Day OFF RH Move 180sec"
        conditions_to_merge = ["1Day OFF " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "1Day OFF LH Move 180sec"
        conditions_to_merge = ["1Day OFF " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "1Day ON RH Move 180sec"
        conditions_to_merge = ["1Day ON " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "1Day ON LH Move 180sec"
        conditions_to_merge = ["1Day ON " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        ## 5 DAY

        new_condition_name = "5Day OFF Rest 180sec"
        conditions_to_merge = ["5Day OFF Rest"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "5Day ON Rest 180sec"
        conditions_to_merge = ["5Day ON Rest", "5Day ON RH (Pass)", "5Day ON LH (Pass)"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        patient.comment("Not enough data for 5Day ON Rest (140 sec). Added RH (Pass) and LH (Pass) to reach 180 sec target")


        new_condition_name = "5Day OFF RH Move 180sec"
        conditions_to_merge = ["5Day OFF " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "5Day OFF LH Move 180sec"
        conditions_to_merge = ["5Day OFF " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "5Day ON RH Move 180sec"
        conditions_to_merge = ["5Day ON " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "5Day ON LH Move 180sec"
        conditions_to_merge = ["5Day ON " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)
        
    if patient_name == 'Patient5':
        
        new_condition_name = "1Day OFF Rest 180sec"
        conditions_to_merge = ["1Day OFF Rest (EyesOpen)", "1Day OFF Rest (EyesClosed)"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "1Day ON Rest 180sec"
        conditions_to_merge = ["1Day ON Rest (EyesOpen)", "1Day ON Rest (EyesClosed)"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "1Day OFF RH Move 180sec"
        conditions_to_merge = ["1Day OFF " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "1Day OFF LH Move 180sec"
        conditions_to_merge = ["1Day OFF " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "1Day ON RH Move 180sec"
        conditions_to_merge = ["1Day ON " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "1Day ON LH Move 180sec"
        conditions_to_merge = ["1Day ON " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        ## 5 DAY

        new_condition_name = "5Day OFF Rest 180sec"
        conditions_to_merge = ["5Day OFF Rest (EyesOpen)", "5Day OFF Rest (EyesClosed)"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "5Day ON Rest 180sec"
        conditions_to_merge = ["5Day ON Rest (EyesOpen)", "5Day ON Rest (EyesClosed)"]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "5Day OFF RH Move 180sec"
        conditions_to_merge = ["5Day OFF " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "5Day OFF LH Move 180sec"
        conditions_to_merge = ["5Day OFF " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


        new_condition_name = "5Day ON RH Move 180sec"
        conditions_to_merge = ["5Day ON " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

        new_condition_name = "5Day ON LH Move 180sec"
        conditions_to_merge = ["5Day ON " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
        patient.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

    print(f"Created the following conditions for {patient_name}:")
    for condition in patient.conditions:
        if "180sec" in condition:
            print(condition)

    patient.save()
    
    
if __name__ == '__main__':
    main()