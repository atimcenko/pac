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
from numba import njit, prange

from tqdm.notebook import tqdm
from collections import defaultdict
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from utility_functions import *
from lfp_class import LFP

class MyPAC:
    
    def __init__(self,
                 beta_params=(5, 45, 1, 2), 
                 hfo_params=(40, 500, 20, 0),
                 method='MI', 
                 method_params={'return_PA_distr': False, 'bins': None}, 
                 verbose=True, multiprocess=False, use_numba=False):

        
        self.beta_params = beta_params # tuple: (start, stop, step, bw - always)
        self.hfo_params = hfo_params # tuple, if bw = 0 -> varying bw = 2 * beta_f
        
        self.method = method # 'MI', 'GLM', 'MVL'
        self.method_params = method_params
        
        f1_beta, f2_beta, step_beta, bw_beta = self.beta_params
        f1_hfo, f2_hfo, step_hfo, bw_hfo = self.hfo_params
    
        self.beta_freqs = np.arange(f1_beta, f2_beta + step_beta, step_beta)
        self.hfo_freqs = np.arange(f1_hfo, f2_hfo + step_hfo, step_hfo)
        
        self.verbose = verbose
        self.multiprocess = multiprocess
        self.use_numba = use_numba
        
        self.lfp_phase = None
        self.lfp_amplitude = None
        
        self.name = ''
        
        self.patient_name = ''
        self.condition = ''
        self.phase_placement = ''
        self.amplitude_placement = ''
        self.duration = ''
        
        self.beta_matrix = None
        self.hfo_matrix = None
        self.pac_matrix = None
        self.surrogates = None
        self.pvalues = None
        
        self.rand_idx = []
        
        
    def _filter(self, lfp_phase:LFP, lfp_amplitude=None, out='hilbert_component', filter_pad=1):
        
        """
        Creates bandpass filtered LFP signal components according to beta_params and hfo_params
        > Writes self.lfp_phase, self.lfp_amplitude
        > Writes self.patients
        > Creates path for "pac" folder
        
        
        """
        t0 = time.time()
        
        if out == 'hilbert_component':
            out_text = "phases and amplitudes"
        if out == 'data':
            out_text = "filtered signals"
        if self.verbose: print("---Filtering out {} components for PAC estimation---".format(out_text))
        if self.verbose: print("Low frequency band filter params: start = {} Hz, stop = {} Hz, step = {} Hz, bw = {} Hz".format(*self.beta_params))
        if self.verbose: print("High frequency band filter params: start = {} Hz, stop = {} Hz, step = {} Hz, bw = {} Hz".format(*self.hfo_params))
        
        if lfp_amplitude is None:
            lfp_amplitude = lfp_phase
            
        if self.verbose: print("PHASE: {}".format(lfp_phase.name))
        if self.verbose: print("AMPLITUDE: {}".format(lfp_amplitude.name))
            
            
        """ SETTING ATTRIBUTES: phase, amplitude, name"""
            
        self.lfp_phase = lfp_phase
        self.lfp_amplitude = lfp_amplitude
        
        self.create_name()
        self.name_to_components()
        
        if self.verbose: print("[UPDATED: self.lfp_phase; self.lfp_amplitude]")
        
        """----------------------PADDING------------------------"""
        if filter_pad != 0:
            
            filter_pad *= lfp_phase.sf
            zero_pad = np.zeros(filter_pad)

            padded_data_beta = np.hstack((zero_pad, lfp_phase.data, zero_pad))
            padded_data_hfo = np.hstack((zero_pad, lfp_amplitude.data, zero_pad))

            lfp_temp_beta = LFP(padded_data_beta, lfp_phase.sf, lfp_phase.name)
            lfp_temp_hfo = LFP(padded_data_hfo, lfp_amplitude.sf, lfp_amplitude.name)
        """-----------------------------------------------------"""
        
        f1_beta, f2_beta, step_beta, bw_beta = self.beta_params
        f1_hfo, f2_hfo, step_hfo, bw_hfo = self.hfo_params
        
        nsteps_beta = int((f2_beta - f1_beta)/step_beta + 1)
        nsteps_hfo = int((f2_hfo - f1_hfo)/step_hfo + 1)
        
        """------filtering low-frequency (beta) components------"""
        
        # initialize output arrays
        beta_components = np.zeros((nsteps_beta, lfp_amplitude.length))
        phases = np.zeros((nsteps_beta, lfp_amplitude.length))
    
        for j in range(nsteps_beta):
            
            center = f1_beta + j * step_beta
            right = center + bw_beta/2
            left = center - bw_beta/2
            
            lfp_component = lfp_temp_beta.bp_filter(left, right, inplace=False, filter_order=2)
            
            if filter_pad != 0:
                beta_components[j] = lfp_component.data[filter_pad:-filter_pad]
            else:
                beta_components[j] = lfp_component.data
                
            if out == 'hilbert_component':
                beta_components[j] = extract_phase(beta_components[j])
                
        # repeat filtered lfp data (or phase of analytical signal) for each beta freq n_hfo steps times
        beta_matrix = np.tile(beta_components, (nsteps_hfo, 1, 1)) # shape = (n_hfo, n_beta, data_length)
        
        assert beta_matrix.shape == (nsteps_hfo, nsteps_beta, lfp_amplitude.length)
        
        """------filtering high-frequency (HFO) components------"""
        
        # constant hfo bandwidth
        if bw_hfo != 0:
            
            # initialize output arrays
            hfo_components = np.zeros((nsteps_hfo, lfp_amplitude.length))
            amplitudes = np.zeros((nsteps_hfo, lfp_amplitude.length))
            
            for i in range(nsteps_hfo):
                
                center = f1_hfo + i * step_hfo
                right = center + bw_hfo/2
                left = center - bw_hfo/2
                lfp_component = lfp_temp_hfo.bp_filter(left, right, inplace=False, filter_order=3)
                
                if filter_pad != 0:
                    hfo_components[i] = lfp_component.data[filter_pad:-filter_pad]
                else:
                    hfo_components[i] = lfp_component.data 
                
                if out == 'hilbert_component':
                    hfo_components[i] = extract_amplitude(hfo_components[i])
                    
            # creating array of shape (n_hfo, n_beta, lfp_length)
            
            hfo_matrix = np.transpose(np.tile(hfo_components, (nsteps_beta, 1, 1)), axes=[1, 0, 2])
            
         
        """------variable HFO filter bandwidth------"""
        if bw_hfo == 0:
            if self.verbose: print("---Variable amplitude filter bandwidth---")
            
            var_bw_list = 2 * np.arange(f1_beta, f2_beta + step_beta, step_beta) + bw_beta//2
            hfo_matrix = np.zeros((nsteps_hfo, nsteps_beta, lfp_amplitude.length))
            
            for j in range(nsteps_beta):
                
                bw_hfo = var_bw_list[j]
                
                for i in range(nsteps_hfo):
 
                    center = f1_hfo + i * step_hfo
                    right = center + bw_hfo/2
                    left = center - bw_hfo/2
                    
                    if left <= 0:
                        left = center - 20
                    
                    lfp_component = lfp_temp_hfo.bp_filter(left, right, inplace=False, filter_order=3)
                    
                    if filter_pad != 0:
                        hfo_matrix[i, j] = lfp_component.data[filter_pad:-filter_pad]
                    else:
                        hfo_matrix[i, j] = lfp_component.data

                    if out == 'hilbert_component':
                        hfo_matrix[i, j] = extract_amplitude(hfo_matrix[i, j])
                
        assert hfo_matrix.shape == (nsteps_hfo, nsteps_beta, lfp_amplitude.length)
        assert hfo_matrix.shape == beta_matrix.shape       
            
        t1 = time.time()
        
        time_elapsed = round(t1 - t0, 2)
        
        self.beta_matrix = beta_matrix
        self.hfo_matrix = hfo_matrix
        
        if self.verbose: print("[UPDATED: self.beta_matrix; self.hfo_matrix]")
        
        if self.verbose: print("---Finished filtering; elapsed time {} sec---".format(time_elapsed))
        
        return beta_matrix, hfo_matrix
    
    
    def fit_surrogates(self, beta_matrix=None, hfo_matrix=None, pac_matrix=None, out='pvalues', n_surrogates=100, n_splits=3):
        """
        beta and hfo matrices must be extracted phase and amplitude (out='hilbert component for self._filter')
        """
        #assert hasattr(self, "beta_matrix"), "No beta_matrix found! Use self._filter first."
        #assert hasattr(self, "hfo_matrix"), "No hfo_matrix! Use self._filter first." 
        if (beta_matrix is None) or (hfo_matrix is None):
            assert hasattr(self, "beta_matrix"), "No beta_matrix! Use self._filter first."
            assert hasattr(self, "hfo_matrix"), "No hfo_matrix! Use self._filter first."
            beta_matrix, hfo_matrix = self.beta_matrix, self.hfo_matrix
        
        if pac_matrix is None:
            assert hasattr(self, "pac_matrix"), "No pac_matrix! Use self.fit_pac first."
            pac_matrix = self.pac_matrix
        
        if self.verbose: print("Creating {} surrogates".format(n_surrogates))
            
        n_hfo, n_beta, n_times = beta_matrix.shape
        surrogate_pac_matrices = np.zeros((n_surrogates, n_hfo, n_beta))

        t0 = time.perf_counter()

        if self.verbose and self.use_numba: print("USING NO-PYTHON")

        if not self.multiprocess:

            if self.verbose: print("NOT USING FUTURES MULTIPROCESSING")
            if self.use_numba:
                surrogate_pac_matrices = calculate_surrogates_njit(beta_matrix, hfo_matrix, n_surrogates, n_splits)
            else:
                for k in tqdm(range(n_surrogates), desc='Fitting surrogates'):
                    # shuffling HFO matrix in n_splits + 1 chunks
                    shuffled_hfo_matrix = shuffle_hfo_matrix(hfo_matrix, n_splits=n_splits)
                    surrogate_pac_matrices[k, :, :] = calculate_PAC_matrix(beta_matrix, shuffled_hfo_matrix, method=self.method, method_params=self.method_params)
            if self.verbose: print(f"SURROGATE ESTIMATION COMPLETE WITHOUT FUTURES MULTIPROSSING IN {time.perf_counter() - t0} seconds")
        
        if self.multiprocess:
            if self.verbose: print("USING MULTIPROCESSING")
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                print(f"Started {executor}")
                futures = []
                if self.use_numba:
                    f_PAC_matrix = calculate_PAC_matrix_njit
                    if n_splits == 1:
                        f_shuffle_hfo = shuffle_hfo_matrix_njit_1split
                    else:
                        f_shuffle_hfo = shuffle_hfo_matrix_njit
                else:
                    f_PAC_matrix = calculate_PAC_matrix
                for k in range(n_surrogates):
                    shuffled_hfo_matrix = f_shuffle_hfo(hfo_matrix, n_splits)
                    futures.append(executor.submit(f_PAC_matrix, beta_matrix, shuffled_hfo_matrix))
                    if self.verbose:
                        if n_surrogates < 25:
                            print(f"{k}: created future object")
                        else:
                            if k % 20 == 0: print(f"{k}: created future object")

                for k in range(n_surrogates):
                    surrogate_pac_matrices[k, :, :] = futures[k].result()
                    if self.verbose:
                        if n_surrogates < 25:
                            print(f"{k}: calculated result")
                        else:
                            if k % 20 == 0: print(f"{k}: calculated result")

            if self.verbose: print(f"SURROGATE ESTIMATION COMPLETE WITH MULTIPROSSING IN {time.perf_counter() - t0} seconds")

        self.surrogates = surrogate_pac_matrices
        
        if self.verbose: print("[UPDATED: self.surrogates]".format(self.method))
        # p-values 
        if self.verbose: print("Calculating p-values...")
        
        pvalues = np.zeros_like(pac_matrix)
        for i in range(n_hfo):
            for j in range(n_beta):
                currentPAC = pac_matrix[i, j]
                distribution = surrogate_pac_matrices[:, i, j]
                pvalues[i, j] = np.sum(distribution >= currentPAC)/len(distribution)
                       
        self.pvalues = pvalues
        
        if self.verbose: print("[UPDATED: self.pvalues]")
            
        if out == 'pvalues':
            return pvalues
        
        if out == 'surrogates':
            return surrogate_pac_matrices
    
    
    def filter_fit_surrogates(self, lfp_phase, 
                              lfp_amplitude=None, 
                              out='pvalues', 
                              n_surrogates=100, 
                              n_splits=3, 
                              filter_pad=1):
        """
        Same as filter_fit_pac but returns surrogate pac matrices
        Out = "pvalues" or "surrogates"
        """
        
        t0 = time.time()
        
        if lfp_amplitude is None:
            lfp_amplitude = lfp_phase
        
        if self.verbose: print("---Creating and estimating PAC on {} surrogates---".format(n_surrogates))
        
        beta_matrix, hfo_matrix = self._filter(lfp_phase, lfp_amplitude, filter_pad=filter_pad)
        pac_matrix = self.fit_pac(beta_matrix, hfo_matrix)
        out_return = self.fit_surrogates(beta_matrix, hfo_matrix, pac_matrix, out=out, n_surrogates=n_surrogates, n_splits=n_splits)
        
        t1 = time.time()
        time_elapsed = round(t1 - t0, 2)
        
        if self.verbose: print("--- {} surrogates PAC estimation completed in {} sec---".format(n_surrogates, time_elapsed))
            
        if self.verbose: print("Returning {}".format(str(out)))   
        return out_return


    def fit_pac(self, beta_matrix=None, hfo_matrix=None):
        
        if self.verbose: print("---Starting PAC estimation using {} method---".format(self.method))

        if (beta_matrix is None) or (hfo_matrix is None):
            assert hasattr(self, "beta_matrix"), "No beta_matrix! Use self._filter first."
            assert hasattr(self, "hfo_matrix"), "No hfo_matrix! Use self._filter first."
            beta_matrix, hfo_matrix = self.beta_matrix, self.hfo_matrix
        
        n_beta = len(self.beta_freqs)
        n_hfo = len(self.hfo_freqs)
        
        if self.use_numba:
            pac_matrix = calculate_PAC_matrix_njit(beta_matrix, hfo_matrix)
        else:
            pac_matrix = calculate_PAC_matrix(beta_matrix, hfo_matrix, method=self.method, method_params=self.method_params)
        
        self.pac_matrix = pac_matrix
                
        if self.verbose: print("[UPDATED: self.pac_matrix]") 
            
        return pac_matrix
    
                
    def filter_fit_pac(self, lfp_phase: LFP, lfp_amplitude=None, filter_pad=1):
        
        t0 = time.time()
        
        if lfp_amplitude is None:
            lfp_amplitude = lfp_phase
        
        if self.verbose: print("PAC between phase: {} and amplitude: {}".format(lfp_phase.name, lfp_amplitude.placement))
        
        self.fit_pac(*self._filter(lfp_phase, lfp_amplitude, filter_pad=filter_pad))
        
        t1 = time.time()
        time_elapsed = round(t1 - t0, 2)
        
        if self.verbose: print("---PAC estimation completed in {} sec---".format(time_elapsed))
      
    
    def comodulogram(self, source=None, pvalues=None, significant=False, correction='None', smooth=True, sigma=1, vmax=None, ax=None, savefig=False):
        
        if source is None:
            pac_matrix = self.pac_matrix.copy()
        else:
            pac_matrix = source.copy()
            
        if significant:
    
            assert hasattr(self, 'pvalues'), "p-values are not yet calculated! Use self.fit_surrogates"
        
            if correction == 'None':
                zero_indeces = self.pvalues > 0.05
                 
            if correction == 'Multiple':
                
                pvalues = self.pvalues.copy().flatten()
                pvalues_shape = self.pvalues.shape
                
                reject, pvalues_corrected, _, _ = multipletests(pvalues) # default Holm-Sidak method
                
                # if we reject null-hypothesis, that is PAC is confirmed to be significant - padding zero everything else
        
                reject = reject.reshape(pvalues_shape)
                pvalues_corrected = pvalues_corrected.reshape(pvalues_shape)
                
                zero_indeces = True ^ reject # XOR inverts boolean array, True if (True ^ False) and False if (True ^ True)
                
            pac_matrix[zero_indeces] = 0     
            
        f1, f2, step, bw = self.beta_params
        xticklabels = np.arange(f1, f2 + step, step)
        f1, f2, step, bw = self.hfo_params
        yticklabels = np.arange(f2, f1 - step, -step)

        # skipping some xticks

        space_idx = (1 - (xticklabels % 5 == 0)).astype('bool')
        xticklabels = xticklabels.astype('str')
        xticklabels[space_idx] = ''

        # skipping some yticks

        space_idx = (1 - (yticklabels % 50 == 0)).astype('bool')
        yticklabels = yticklabels.astype('str')
        yticklabels[space_idx] = ''

        results = pac_matrix

        if smooth: 
            results = gaussian_filter(pac_matrix, sigma)
            
        if vmax is None:
            vmax = np.max(np.max(results))  
            
        plt.title(f"PAC ({self.method}); {self.patient_name} ; {self.condition}; \n [Phase] {self.phase_placement} ->  [Amplitude] {self.amplitude_placement}")
        sns.heatmap(results[::-1, ::], xticklabels=xticklabels, yticklabels=yticklabels, cmap="RdBu_r", vmin=0, vmax=vmax, ax=ax)
        
        if savefig:
            filename = self.name + '.png'
            im_dir = os.path.join(self.root_dir, 'im')
            try:
                os.mkdir(im_dir)
            except OSError:
                pass
            plt.savefig(os.path.join(im_dir, filename))
            
            
    def create_name(self):
        pac_name_components = ['PAC', self.lfp_phase.patient_name, 
                               self.lfp_phase.condition, 
                               self.lfp_phase.placement, 
                               self.lfp_amplitude.placement, 
                               f"{self.lfp_phase.duration} sec"]
        self.name = '_'.join(pac_name_components)
        if self.verbose: print("[UPDATED] self.name: ", self.name)
        return self.name
    
    
    def name_to_components(self):
        components = self.name.split("_")
        self.patient_name = components[1]
        self.condition = components[2]
        self.phase_placement = components[3]
        self.amplitude_placement = components[4]
        self.duration = components[5]
    
    """
    def __getstate__(self):
        attrs = self.__dict__.copy()
        keys_to_del = ['beta_matrix', 'hfo_matrix', 'lfp_phase', 'lfp_amplitude', 'rand_idx']
        if self.verbose: print(f"Pickling {self.name} without {keys_to_del}")
        for key in keys_to_del:
            del attrs[key]
        return attrs
    """
    
    def create_pac_folder(self, patient_dir):
        path = os.path.join(patient_dir, "pac")
        if self.verbose: print(f"Trying creating {path} folder")
        try:
            os.mkdir(path)
        except OSError:
            if self.verbose: 
                print(f" Folder {path} already exists")
        self.root_dir = path
        if self.verbose: print("[UPDATED] self.root_dir")
    
    """
    def save_data(self, patient_dir):
        t = time.time()
        self.create_pac_folder(patient_dir)
        
        c = DataContainerPAC(self)
        filepath = os.path.join(self.root_dir, c.name + ".pkl")
        self.data_filepath = filepath
        if self.verbose: print(f"Saving {self.name} DATA to self.data_filepath: {filepath} ...")
        with open(filepath, 'wb') as output:
            pickle.dump(c, output, pickle.HIGHEST_PROTOCOL)
        if self.verbose: print(f"Done, {time.time() - t} sec")
        if self.verbose: print("Returning filepath for saved file")
        return filepath
    """
    
    def save(self, patient_dir):
        t = time.time()
        self.create_pac_folder(patient_dir) 
        filepath = os.path.join(self.root_dir, self.name + ".pkl")
        self.obj_filepath = filepath
        if self.verbose: print(f"Saving {self.name} object to self.obj_filepath: {filepath} ...")
        keys_to_del = ['beta_matrix', 'hfo_matrix', 'lfp_phase', 'lfp_amplitude', 'rand_idx']
        if self.verbose: print("Setting attributes back to None")
        for key in keys_to_del:
            self.__dict__[key] = None
            print(f"self.{key} = {self.__dict__[key]}")

        with open(filepath, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        if self.verbose: print(f"Done, {time.time() - t} sec")
        file_stats = os.stat(filepath)
        print(f'File size: {file_stats.st_size / (1024 * 1024)} MB')
        if self.verbose: print("Returning filepath for saved file")
        return filepath
    
    """
    def save(self, patient_dir): 
        t = time.time()
        filepath = self.save_object(patient_dir)
        self.save_data(patient_dir)
        if self.verbose: print(f"Done, {time.time() - t} sec")
        if self.verbose: print("Returning filepath for the saved OBJECT")
        return filepath
    """
    
    
class DataContainerPAC():
    
    def __init__(self, pac: MyPAC):
        """
        Initializes Containter object, which aims to store calculated values separately from object itself for saving purposes
        This Container saves:
        - (UPD - NO) beta_matrix
        - (UPD - NO) hfo_matrix
        - pac_matrix
        - surrogates
        - pvalues
        
        The saved object will have the same name as MyPAC with _data suffix.
        """
        
        #self.beta_matrix = pac.beta_matrix
        #self.hfo_matrix = pac.hfo_matrix
        self.pac_matrix = pac.pac_matrix
        self.surrogates = pac.surrogates
        self.pvalues = pac.pvalues
        
        self.name = pac.name + "_data"
        
        