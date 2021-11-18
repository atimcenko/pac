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

print("Succesfully imported libraries and modules\n")

def main():

    with open("path_data.txt") as f:
        data_dir = f.readline()

    p2_root_dir = os.path.join(data_dir, "Patient2")
    #root_dir = r"H:\Alexey_Timchenko\PAC\Patient2"

    p2 = Patient(name='Patient2', root_dir=p2_root_dir)
    files = p2.find_bdf_files()
    for filename in files:
        p2.scan_file_annotations(filename, update_file_conditions=True)

    p2.get_preprocessed_lfps(verbose=False)
    p2.display_all_annotations()

    # For this patient 1Day without movement-related conditions (not enough data for 180sec)
    new_condition_name = "1Day OFF Rest 180sec"
    conditions_to_merge = ["1Day OFF Rest"]
    p2.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

    new_condition_name = "1Day ON Rest 180sec"
    conditions_to_merge = ["1Day ON Rest"]
    p2.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

    new_condition_name = "5Day OFF Rest 180sec"
    conditions_to_merge = ["5Day OFF Rest"]
    p2.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

    new_condition_name = "5Day ON Rest 180sec"
    conditions_to_merge = ["5Day ON Rest"]
    p2.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


    new_condition_name = "5Day OFF RH Move 180sec"
    conditions_to_merge = ["5Day OFF " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
    p2.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


    new_condition_name = "5Day OFF LH Move 180sec"
    conditions_to_merge = ["5Day OFF " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
    p2.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


    new_condition_name = "5Day ON RH Move 180sec"
    conditions_to_merge = ["5Day ON " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
    p2.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

    new_condition_name = "5Day ON LH Move 180sec"
    conditions_to_merge = ["5Day ON " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
    p2.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

    print("Created the following conditions: ")
    for condition in p2.conditions:
        if "180sec" in condition:
            print(condition)

    p2.save()
    
if __name__ == '__main__':
    main()