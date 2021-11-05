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
#sys.path.insert(0, r"H:\Alexey_Timchenko\PAC\PAC_PROJECT")

sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
plt.rcParams['figure.figsize'] = (14, 8)
from utility_functions import *
from lfp_class import LFP
from pac_class import MyPAC
from patient_class import Patient

##print("Succesfully imported libraries and modules")

def one_pac_surrogate_estimation(pac: MyPAC):
    pac.fit_surrogates(n_surrogates=4, n_splits=3)

def main():
    
    # 1) Load .pkl file -> Patient

    pickle_filepath = r"H:\Alexey_Timchenko\PAC\Patient1\Patient1.pkl"
    p1 = load_patient_from_pickle(pickle_filepath)

    print(f"Loaded {p1.name}")

    # 2) Preprocess MyPAC objects to leave only fit_surrogates method to estimate
    conditions_rest = ["1Day OFF Rest 180sec", "1Day ON Rest 180sec"] # "5Day OFF Rest 180sec", "5Day ON Rest 180sec"]
    placements = ["L2C-3C"] #"R2C-3C"] 

    print("Started PAC preprocessing")
    t0 = time.perf_counter()
    pacs = []
    for condition in conditions_rest:
        for placement in placements:
            lfp = p1.lfp[condition][placement]
            pac = MyPAC(beta_params=(5, 35, 1, 2), hfo_params=(50, 500, 50, 0), verbose=True, multiprocess=True)
            pac.filter_fit_pac(lfp)
            pacs.append(pac)

    print(f"Finished PAC preprocessing in {time.perf_counter() - t0} seconds")

    # 3) Run Multiprocessing test
    """
    t0 = time.perf_counter()

    print("Started Multiprocessing")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = executor.map(one_pac_surrogate_estimation, pacs)

    print("Ended Multiprocessing")

    pacs_new = []
    for future in futures:
        pacs_new.append(future.result())

    for pac in pacs_new:
        pac.save(p1.root_dir)

    print(f"MP Time: {time.perf_counter() - t0}")

    print("Saved files: ", os.listdir(pac.root_dir))
    """
    # 4) Run consecutive processes test

    t0 = time.perf_counter()
    
    for pac in pacs:
        pac.fit_surrogates(n_surrogates=12, n_splits=3)
        #pac.save(p1.root_dir)

    print(f"Total Time: {time.perf_counter() - t0}")


def test():
    pickle_filepath = r"H:\Alexey_Timchenko\PAC\Patient1\Patient1.pkl"
    p1 = load_patient_from_pickle(pickle_filepath)

    print(f"Loaded {p1.name}")
    print(p1.placements)
    condition = "5Day ON Rest 180sec"
    plc_phase = "R1-2A"
    lfp = p1.lfp[condition][plc_phase]
    pac = MyPAC(beta_params=(5, 48, 1, 2), hfo_params=(40, 500, 20, 40), verbose=True, multiprocess=False, use_numba=True)
    pac.filter_fit_surrogates(lfp, lfp, n_surrogates=10, n_splits=1)


if __name__ == '__main__':
    #main()
    test()