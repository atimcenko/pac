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

from fooof import FOOOF
import fooof

import sys
import os
import time
#import pickle
import dill as pickle

import concurrent.futures
from numba import jit, njit, prange

from tqdm.notebook import tqdm
from collections import defaultdict
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from utility_functions import *
from lfp_class import LFP
from pac_class import MyPAC
from patient_class import Patient


def copy_patient(patient: Patient):
    new_patient = Patient(patient.name, patient.root_dir)
    for attr in patient.__dict__.keys():
        new_patient.__dict__[attr] = patient.__dict__[attr]
    return new_patient