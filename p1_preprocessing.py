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

from utility_functions import *
from lfp_class import LFP
from pac_class import MyPAC
from patient_class import Patient

print("Succesfully imported libraries and modules\n")

def main():
    
    p1 = Patient(name='Patient1', root_dir=r"H:\Alexey_Timchenko\PAC\Patient1")
    files = p1.find_bdf_files()
    for filename in files:
        p1.scan_file_annotations(filename, update_file_conditions=True)

    p1.display_all_annotations()
    p1.get_preprocessed_lfps(verbose=False)

    

    # Here WE CREATE NEW CONDITIONS:
    # 1Day OFF Rest / RH / LH movement 180sec
    # 1Day ON Rest / RH / LH movement 180sec

    # Same for 5Day

    sf = 2000

    new_condition_name = "1Day OFF Rest 180sec"
    conditions_to_merge = ["1Day OFF Rest"]
    p1.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)
    
    new_condition_name = "1Day ON Rest 180sec"
    conditions_to_merge = ["1Day ON Rest"]
    p1.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


    new_condition_name = "1Day OFF RH Move 180sec"
    conditions_to_merge = ["1Day OFF " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
    p1.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


    new_condition_name = "1Day OFF LH Move 180sec"
    conditions_to_merge = ["1Day OFF " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "RHLH"]]
    p1.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)
    

    new_condition_name = "1Day ON RH Move 180sec"
    conditions_to_merge = ["1Day ON " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
    p1.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

    new_condition_name = "1Day ON LH Move 180sec"
    conditions_to_merge = ["1Day ON " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
    p1.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


    ## 5 DAY

    new_condition_name = "5Day OFF Rest 180sec"
    conditions_to_merge = ["5Day OFF Rest"]
    p1.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)
    
    new_condition_name = "5Day ON Rest 180sec"
    conditions_to_merge = ["5Day ON Rest"]
    p1.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


    new_condition_name = "5Day OFF RH Move 180sec"
    conditions_to_merge = ["5Day OFF " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
    p1.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)


    new_condition_name = "5Day OFF LH Move 180sec"
    conditions_to_merge = ["5Day OFF " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
    p1.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)
    

    new_condition_name = "5Day ON RH Move 180sec"
    conditions_to_merge = ["5Day ON " + s for s in ["RH (Com)", "RH (NoCom)", "RH (Hold)", "RH (Pass)"]]
    p1.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

    new_condition_name = "5Day ON LH Move 180sec"
    conditions_to_merge = ["5Day ON " + s for s in ["LH (Com)", "LH (NoCom)", "LH (Hold)", "LH (Pass)"]]
    p1.merge_conditions(conditions_to_merge, new_condition_name, total_duration=180)

    """
    prefix = "1Day OFF "
    new_condition = "Rest 180sec"
    for placement in p1.placements:
        lfp = p1.lfp[prefix + "Rest"][placement]
        new_lfp = LFP(lfp.data[:180*sf], patient_name=p1.name, condition=prefix+new_condition, placement=placement)
        p1.add_lfp(new_lfp)

    prefix = "1Day ON "
    new_condition = "Rest 180sec"
    for placement in p1.placements:
        lfp = p1.lfp[prefix + "Rest"][placement]
        new_lfp = LFP(lfp.data, patient_name=p1.name, condition=prefix+new_condition, placement=placement)
        p1.add_lfp(new_lfp)


    prefix = "1Day OFF "
    new_condition = "RH Move 180sec"
    for placement in p1.placements:
        lfp1 = p1.lfp[prefix + "RH (Com)"][placement]
        lfp2 = p1.lfp[prefix + "RH (NoCom)"][placement]
        lfp3 = p1.lfp[prefix + "RH (Hold)"][placement]
        lfp4 = p1.lfp[prefix + "RH (Pass)"][placement]
        new_data = np.concatenate((lfp1.data, lfp2.data, lfp3.data, lfp4.data))
        new_lfp = LFP(new_data, patient_name=p1.name, condition=prefix+new_condition, placement=placement)
        p1.add_lfp(new_lfp)
        
    prefix = "1Day OFF "
    new_condition = "LH Move 180sec"
    for placement in p1.placements:
        lfp1 = p1.lfp[prefix + "LH (Com)"][placement]
        lfp2 = p1.lfp[prefix + "LH (NoCom)"][placement]
        lfp3 = p1.lfp[prefix + "LH (Hold)"][placement]
        lfp4 = p1.lfp[prefix + "RHLH"][placement]
        new_data = np.concatenate((lfp1.data, lfp2.data, lfp3.data, lfp4.data))
        new_lfp = LFP(new_data, patient_name=p1.name, condition=prefix+new_condition, placement=placement)
        p1.add_lfp(new_lfp)
        
    prefix = "1Day ON "
    new_condition = "RH Move 180sec"
    for placement in p1.placements:
        lfp1 = p1.lfp[prefix + "RH (Com)"][placement]
        lfp2 = p1.lfp[prefix + "RH (NoCom)"][placement]
        lfp3 = p1.lfp[prefix + "RH (Hold)"][placement]
        lfp4 = p1.lfp[prefix + "RH (Pass)"][placement]
        new_data = np.concatenate((lfp1.data, lfp2.data, lfp3.data, lfp4.data))
        new_lfp = LFP(new_data, patient_name=p1.name, condition=prefix+new_condition, placement=placement)
        p1.add_lfp(new_lfp)
        
    prefix = "1Day ON "
    new_condition = "LH Move 180sec"
    for placement in p1.placements:
        lfp1 = p1.lfp[prefix + "LH (Com)"][placement]
        lfp2 = p1.lfp[prefix + "LH (NoCom)"][placement]
        lfp3 = p1.lfp[prefix + "LH (Hold)"][placement]
        lfp4 = p1.lfp[prefix + "LH (Pass)"][placement]
        new_data = np.concatenate((lfp1.data, lfp2.data, lfp3.data, lfp4.data))
        new_lfp = LFP(new_data, patient_name=p1.name, condition=prefix+new_condition, placement=placement)
        p1.add_lfp(new_lfp)
        

    #----------5DAY------------

    # in OFF 300 seconds - shrink it to 180 seconds
    prefix = "5Day OFF "
    new_condition = "Rest 180sec"
    for placement in p1.placements:
        lfp = p1.lfp[prefix + "Rest"][placement]
        new_lfp = LFP(lfp.data[:180*sf], patient_name=p1.name, condition=prefix+new_condition, placement=placement)
        p1.add_lfp(new_lfp)
        
    prefix = "5Day ON "
    new_condition = "Rest 180sec"
    for placement in p1.placements:
        lfp = p1.lfp[prefix + "Rest"][placement]
        new_lfp = LFP(lfp.data, patient_name=p1.name, condition=prefix+new_condition, placement=placement)
        p1.add_lfp(new_lfp)


    prefix = "5Day OFF "
    new_condition = "RH Move 180sec"
    for placement in p1.placements:
        lfp1 = p1.lfp[prefix + "RH (Com)"][placement]
        lfp2 = p1.lfp[prefix + "RH (NoCom)"][placement]
        lfp3 = p1.lfp[prefix + "RH (Hold)"][placement]
        lfp4 = p1.lfp[prefix + "RH (Pass)"][placement]
        new_data = np.concatenate((lfp1.data, lfp2.data, lfp3.data, lfp4.data))
        new_lfp = LFP(new_data, patient_name=p1.name, condition=prefix+new_condition, placement=placement)
        p1.add_lfp(new_lfp)
        
    prefix = "5Day OFF "
    new_condition = "LH Move 180sec"
    for placement in p1.placements:
        lfp1 = p1.lfp[prefix + "LH (Com)"][placement]
        lfp2 = p1.lfp[prefix + "LH (NoCom)"][placement]
        lfp3 = p1.lfp[prefix + "LH (Hold)"][placement]
        lfp4 = p1.lfp[prefix + "LH (Pass)"][placement]
        new_data = np.concatenate((lfp1.data, lfp2.data, lfp3.data, lfp4.data))
        new_lfp = LFP(new_data, patient_name=p1.name, condition=prefix+new_condition, placement=placement)
        p1.add_lfp(new_lfp)
        
    prefix = "5Day ON "
    new_condition = "RH Move 180sec"
    for placement in p1.placements:
        lfp1 = p1.lfp[prefix + "RH (Com)"][placement]
        lfp2 = p1.lfp[prefix + "RH (NoCom)"][placement]
        lfp3 = p1.lfp[prefix + "RH (Hold)"][placement]
        lfp4 = p1.lfp[prefix + "RH (Pass)"][placement]
        new_data = np.concatenate((lfp1.data, lfp2.data, lfp3.data, lfp4.data))
        new_lfp = LFP(new_data, patient_name=p1.name, condition=prefix+new_condition, placement=placement)
        p1.add_lfp(new_lfp)
        
    prefix = "5Day ON "
    new_condition = "LH Move 180sec"
    for placement in p1.placements:
        lfp1 = p1.lfp[prefix + "LH (Com)"][placement]
        lfp2 = p1.lfp[prefix + "LH (NoCom)"][placement]
        lfp3 = p1.lfp[prefix + "LH (Hold)"][placement]
        lfp4 = p1.lfp[prefix + "LH (Pass)"][placement]
        new_data = np.concatenate((lfp1.data, lfp2.data, lfp3.data, lfp4.data))
        new_lfp = LFP(new_data, patient_name=p1.name, condition=prefix+new_condition, placement=placement)
        p1.add_lfp(new_lfp)
    """

    print("Conditions: \n", p1.conditions)
    print("Placements: \n", p1.placements)

    # Saving pickle for p1_pac_estimation
    p1_pickle_filepath = p1.save()
    

if __name__ == '__main__':
    main()