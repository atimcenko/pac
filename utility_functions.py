import numpy as np
import pandas as pd

#!pip install pyEDFlib
import pyedflib
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


def extract_phase(beta):
    return np.angle(signal.hilbert(beta), deg = True)


def extract_amplitude(hfo):
    return np.abs(signal.hilbert(hfo))


@njit(fastmath=True)
def calculate_single_PAC_njit(beta_phase, hfo_amplitude):
    
    bins = np.arange(-180, 200, 20)
    nbins = 18
  
    ind = np.searchsorted(bins, beta_phase, side='left')
    count = np.bincount(ind, None)
    sum_value = np.bincount(ind, hfo_amplitude)
    mean_value = sum_value[1:] / count[1:]
    
    pk = mean_value / mean_value.sum()
    qk = np.ones(nbins) / nbins
    KL = (pk * np.log(pk / qk)).sum()
    return KL


@njit(fastmath=True, parallel=True, cache=True)
def calculate_PAC_matrix_njit(beta_matrix, hfo_matrix):
    n_hfo, n_beta = np.shape(beta_matrix)[0], np.shape(beta_matrix)[1]
    pac_matrix = np.zeros((n_hfo, n_beta))
    for i in prange(n_hfo):
        for j in prange(n_beta):
            pac_matrix[i, j] = calculate_single_PAC_njit(beta_matrix[i, j], hfo_matrix[i, j])
    return pac_matrix


@njit(fastmath=True, parallel=False, cache=False)
def calculate_surrogates_njit(beta_matrix, hfo_matrix, n_surrogates, n_splits=3):
    n_hfo, n_beta = np.shape(beta_matrix)[0], np.shape(beta_matrix)[1]
    surrogate_pac_matrices = np.zeros((n_surrogates, n_hfo, n_beta))
    for k in prange(n_surrogates):
        if n_splits == 1:
            shuffled_hfo_matrix = shuffle_hfo_matrix_njit_1split(hfo_matrix, n_splits)
        else:
            shuffled_hfo_matrix = shuffle_hfo_matrix_njit(hfo_matrix, n_splits)
        surrogate_pac_matrices[k, :, :] = calculate_PAC_matrix_njit(beta_matrix, shuffled_hfo_matrix)
        print(k)
    return surrogate_pac_matrices


@njit(fastmath=True)
def shuffle_hfo_matrix_njit(hfo_matrix, n_splits):   
    n_hfo, n_beta, n_times = np.shape(hfo_matrix)
    rand_idx = np.zeros(n_splits + 2)
    rand_idx[1:n_splits+1] = np.sort(np.random.randint(200, n_times-200, n_splits))
    rand_idx[n_splits] = n_times
    shuffled_idx = np.random.permutation(np.arange(n_splits+1)) # e.g. [3, 1, 0, 2] - this is index of list
    
    # make sure shuffled array is not the same as input
    while not np.abs(shuffled_idx - np.arange(n_splits+1)).sum():
        shuffled_idx = np.random.permutation(np.arange(n_splits+1))
        
    j_left = shuffled_idx[0]
    j_right = j_left + 1 
    shuff = hfo_matrix[:, :, rand_idx[j_left]:rand_idx[j_right]]
    for i in range(1, n_splits+1):
        j_left  = shuffled_idx[i]
        j_right = j_left + 1 
        chunk = hfo_matrix[:, :, rand_idx[j_left]:rand_idx[j_right]]
        shuff = np.concatenate((shuff, chunk), axis=2)
    return shuff


@njit(fastmath=True)
def shuffle_hfo_matrix_njit_1split(hfo_matrix, n_splits):
    n_hfo, n_beta, n_times = np.shape(hfo_matrix)

    rand_idx = np.random.randint(200, n_times-200)
    chunk1 = hfo_matrix[:, :, :rand_idx] # len = rand_idx
    chunk2 = hfo_matrix[:, :, rand_idx:] # len = n - rand_idx
    
    #shuff = np.zeros((n_hfo, n_beta, n_times), dtype=np.float64)
    #shuff[:, :, :n_times - rand_idx] = hfo_matrix[:, :, rand_idx:]
    #shuff[:, :, n_times - rand_idx:] = hfo_matrix[:, :, :rand_idx]
        
    shuff = np.concatenate((chunk2, chunk1), axis=2)
    return shuff


def calculate_single_PAC(beta_phase, hfo_amplitude, method='MI', method_params={'bins': None, 'return_PA_distr': False}):
    """
    Calculates single PAC (beta -> hfo)
    """

    if method == 'MI':
        bins, return_PA_distr = method_params['bins'], method_params['return_PA_distr']
        if bins is None:
            bins = np.arange(-180, 200, 20)
        nbins = len(bins) - 1
        # computing average over nbins beta phase bins
        PA_distr = binned_statistic(beta_phase, hfo_amplitude, statistic='mean', bins=bins).statistic
        # normalizing mean amplitude values
        PA_distr = PA_distr/np.sum(PA_distr)
        # reference uniform distribution
        uniform_distr = np.ones(nbins) / nbins
        # yielding MI value using KL-divergence
        MI = entropy(PA_distr, uniform_distr)

        #phases = bins[:-1] + 10 # np.arange(-170, 190, 20)
        #preferred_phase = phases[np.argmax(PA_distr)]
        
        if return_PA_distr:
            return PA_distr
        return MI
        
    if method == 'GLM':
        x1 = np.cos(beta_phase).reshape((-1, 1)) # column vector
        x2 = np.sin(beta_phase).reshape((-1, 1)) # column vector

        X = np.hstack((x1, x2))
        y = hfo_amplitude/np.max(abs(hfo_amplitude))

        model = LinearRegression()

        model.fit(X, y)

        beta1, beta2 = model.coef_
        pac = np.sqrt(beta1 ** 2 + beta2 ** 2)
        return pac
    
    if method == 'MVL':
        n = len(beta_phase)
        mvl = np.abs(np.sum(hfo_amplitude * np.exp(1j * beta_phase)) / n)
        return mvl


def calculate_PAC_matrix(beta_matrix, hfo_matrix, method='MI', method_params={'bins': None, 'return_PA_distr': False}):
    n_hfo, n_beta = np.shape(beta_matrix)[0], np.shape(beta_matrix)[1]
    pac_matrix = np.zeros((n_hfo, n_beta))
    for i in range(n_hfo):
        for j in range(n_beta):
            pac_matrix[i, j] = calculate_single_PAC(beta_matrix[i, j], hfo_matrix[i, j], method=method, method_params=method_params)
    return pac_matrix


def shuffle_hfo_matrix(hfo_matrix, n_splits=3):
    n_hfo, n_beta, n_times = hfo_matrix.shape
    rand_idx = list(np.sort(np.random.randint(low=200, high=n_times-200, size=n_splits)))
    rand_idx.insert(0, 0)
    rand_idx.append(n_times)
    subarrays = []
    for i in range(n_splits+1):
        subarrays.append(hfo_matrix[:, :, rand_idx[i]:rand_idx[i+1]])
    np.random.shuffle(subarrays)
    shuffled_hfo_matrix = np.concatenate(subarrays, axis=2)
    return shuffled_hfo_matrix

def generate_coupled_signal(f_p, f_a, K_p, K_a, xi, timepoints, noise_level=0.1, noise_type='pink', alpha=1):
    x_fp = K_p * np.sin(2 * np.pi * f_p * timepoints)
    A_fa = K_a/2 * ((1 - xi) * np.sin(2 * np.pi * f_p * timepoints) + xi + 1)
    x_fa = A_fa * np.sin(2 * np.pi * f_a * timepoints)
    
    n = len(timepoints)
    
    if noise_type == 'white-gaussian':
        noise = np.random.normal(scale=noise_level, size=n)
    
    if noise_type == 'white-uniform':
        noise = np.random.uniform(low=-noise_level, high=noise_level, size=n)
        
    if noise_type == 'pink':
        noise = np.random.normal(scale=noise_level, size=n)
        
        noise_spectrum = fft(noise)
        freqs = fftfreq(n, timepoints[1] - timepoints[0])

        oneOverF = np.insert((1/(freqs[1:]**alpha)), 0, 0)
        new_spectrum = oneOverF * noise_spectrum                    
        noise = np.abs(ifft(new_spectrum))    
    
    x = x_fp + x_fa + noise
    
    return x


def downsample(signals, signal_headers, q):
    new_signals = []
    q = 8
    for sig in signals:
        new_signals.append(signal.decimate(sig, q))
        
    sf = signal_headers[0]['sample_rate']
    new_signal_headers = signal_headers.copy()
    
    for new_signal_header in new_signal_headers:
        new_signal_header['sample_rate'] = int(sf/q)   
    del(signals)
    return new_signals, new_signal_headers


def load_pac_from_pickle(filepath):
    with open(filepath, 'rb') as f:
        pac = pickle.load(f, pickle.HIGHEST_PROTOCOL)
    return pac


def load_patient_from_pickle(filepath):
    with open(filepath, 'rb') as f:
        patient = pickle.load(f, pickle.HIGHEST_PROTOCOL)
    patient.pac = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    return patient


def create_pac_name(lfp_phase, lfp_amplitude):
    pac_name_components = ['PAC', lfp_phase.patient_name, 
                           lfp_phase.condition, 
                           lfp_phase.placement, 
                           lfp_amplitude.placement, 
                           f"{lfp_phase.duration} sec"]
    name = '_'.join(pac_name_components)
    return name

def retrieve_pac_name(pac_filename):
    """ Returns patient_name, condition, phase_placement, ampl_placement, duration"""
    comps = pac_filename.split("_")
    patient_name, condition = comps[0], comps[1]
    phase_placement, ampl_placement = comps[2], comps[3]
    duration = comps[4]
    return patient_name, condition, phase_placement, ampl_placement, duration
    


def create_condition_name(day, ldopa, movement):
    return " ".join([day, ldopa, movement])


def retrieve_condition_name(condition):
    """
    1Day OFF RH (Com)
    
    """
    name_components = condition.split(" ")
    day = name_components[0]
    ldopa = name_components[1]
    movement = " ".join(name_components[2:])
    
    return day, ldopa, movement

