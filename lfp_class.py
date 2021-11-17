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

class LFP:

    def __init__(self, data, sampling_frequency=2000, patient_name='Sample', condition='', placement=''):
        """
        data: bipolar preferably downsampled
        sampling_frequency: sf of the data
        patient_name: name of the patients (as in Patient class)
        condition: e.g. "OFF Rest"
        placement: e.g. "2A-3A Left"
        """
        self.data = data
        self.sf = sampling_frequency
        self.length = len(self.data)
        self.duration = np.round(self.length/self.sf, 1)
        self.patient_name = patient_name
        self.condition = condition
        self.placement = placement
        self.name = self.create_name()
        
        
    def plot(self, begin=0, end=1, ax=None, **plt_kwargs):
        """
        Plots given LFP time series
        begin, end must be in seconds
        if used as subplot ax needs to be specified
        """
        begin = int(begin * self.sf)
        end = int(end * self.sf)
        timepoints = np.arange(begin, end)
        data = self.data[begin:end]
        if ax is None:
            plt.plot(timepoints, data, label=self.name, **plt_kwargs)
        else:
            ax.plot(timepoints, data, label=self.name, **plt_kwargs)

            
    def get_spectra(self, smooth=False, sigma=4):
        n = self.length
        sf = self.sf
        freqs = np.linspace(sf/n, sf, n)
        
        if smooth:
            return freqs, gaussian_filter1d(abs(fft(self.data)), sigma=sigma)
        return freqs, abs(fft(self.data))
 

    def show_spectra(self, show_freqs=[0, 50], log=False, smooth=False, sigma=4):
        
        n = self.length
        sf = self.sf
        
        assert show_freqs[1] <= sf
        
        if show_freqs[0] <= sf/n:
            left = 0
        else:
            m = np.ceil(sf/show_freqs[0])
            left = int(n/m - 1)
            
        if show_freqs[1] == sf:
            right = n - 1
        else:
            m = np.ceil(sf/show_freqs[1])
            right = int(n/m - 1)
            

        x, y = self.get_spectra(smooth=smooth, sigma=sigma)
        if log:
            plt.yscale('log')
        plt.plot(x[left:right], y[left:right], label=self.name)
        plt.title('Спектр сигнала {}'.format(self.name))
        #plt.grid()
        plt.legend()
        return
      
        
    def get_psd(self, smooth=False, sigma=4, welch_kwargs={'window': 'hann', 
                                      'nperseg': None, 
                                      'noverlap': None, 
                                      'nfft': None, 
                                      'detrend': 'constant', 
                                      'scaling': 'density'}):
        if welch_kwargs['nperseg'] is None:
            welch_kwargs['nperseg'] = self.sf * 1
        x, y = sg.welch(self.data, self.sf, **welch_kwargs)
        if smooth:
            y = gaussian_filter1d(y, sigma=sigma)
        return x, y
    
    
    def show_psd(self, show_freqs=[0, 50], log=False, smooth=False, sigma=4, ax=None, welch_kwargs=\
                                    {'window': 'hann', 
                                      'nperseg': None, 
                                      'noverlap': None, 
                                      'nfft': None, 
                                      'detrend': 'constant', 
                                      'scaling': 'density'}):
        
        x, y = self.get_psd(smooth=smooth, sigma=sigma, welch_kwargs=welch_kwargs)
        show_indexes = (x <= show_freqs[1]) * (x >= show_freqs[0])
        
        title = f"PSD \n {self.patient_name}; {self.condition}"
        
        if ax is not None:
            ax.plot(x[show_indexes], y[show_indexes], label=self.placement)
            ax.set_title(title)
            ax.grid(True)
            ax.legend()
            ax.set_xlabel('Hz')
            ax.set_ylabel("$mV^2/Hz$")
            if log:
                ax.set_yscale('log')
                ax.set_ylabel("dB/Hz")
        if ax is None:
            plt.plot(x[show_indexes], y[show_indexes], label=self.placement)
            plt.title(title)
            #plt.grid()
            plt.legend()
            plt.xlabel('Hz')
            plt.ylabel("$mV^2/Hz$")
            if log:
                plt.yscale('log')
                plt.ylabel("dB/Hz")
        return
        

    def bp_filter(self, low_freq, high_freq, inplace=False, filter_order=4, set_name=False):
        Wn1 = low_freq / (self.sf/2) # Нормируем частоты относительно часототы дискретизации
        Wn2 = high_freq / (self.sf/2)
        # b, a = sg.iirfilter(N=filter_order, Wn=[Wn1, Wn2], btype='bandpass', ftype='butter')
        b, a = sg.butter(N=filter_order, Wn=[Wn1, Wn2], btype='band')
        if inplace:
            self.data = sg.filtfilt(b, a, self.data)
        new_name = self.name
        if set_name:
            new_name = self.name + ", bp-filtered at " + str(low_freq) + "-" + str(high_freq) + "Hz"
        return LFP(sg.filtfilt(b, a, self.data), self.sf, new_name)
    
    
    def notch_filter(self, cutoff_freq, Q=50, inplace=False, set_name=False):
        w0 = cutoff_freq/(self.sf/2)
        b, a = sg.iirnotch(w0, Q)
        if inplace:
            self.data = sg.filtfilt(b, a, self.data)
        new_name = self.name
        if set_name:
            new_name = self.name + ", notch-filtered at " + str(cutoff_freq)
        return LFP(sg.filtfilt(b, a, self.data), self.sf, new_name)
    
    
    def lp_filter(self, high_freq, inplace=False, filter_order=4, set_name=False):
        w0 = high_freq/(self.sf/2)
        b, a = sg.iirfilter(N=filter_order, Wn=high_freq, btype='lowpass', fs=self.sf)
        if inplace:
            self.data = sg.filtfilt(b, a, self.data)
        new_name = self.name
        if set_name:
            new_name = self.name + "lp-filtered at " + str(high_freq) + "Hz"
        return LFP(sg.filtfilt(b, a, self.data), self.sf, new_name)
    
    
    def remove_50hz_harmonics(self, Q, inplace=False):
        harmonics = np.arange(50, 950, 50)
        if inplace:
            for i, cutoff_freq in enumerate(harmonics):
                self.notch_filter(cutoff_freq=cutoff_freq, Q=Q*(i+1), inplace=True)
        else:
            sample_LFP = LFP(self.data, self.sf, name='test')
            for i, cutoff_freq in enumerate(harmonics):
                sample_LFP = sample_LFP.notch_filter(cutoff_freq=cutoff_freq, Q=Q*(i+1), inplace=False)
            return sample_LFP
        
        
    def show_filtered(self, f1, f2, filter_order=2, show_freqs=[0, 500], log=True, spectrum_type='psd', smooth=True, sigma=2):
        if spectrum_type == 'psd':
            self.show_psd(show_freqs=show_freqs, log=log, smooth=smooth, sigma=sigma)
            filtered_lfp = LFP(self.bp_filter(f1, f2, inplace=False, filter_order=filter_order), self.sf, 'Filtered LFP')
            filtered_lfp.show_psd(show_freqs=show_freqs, log=log, smooth=smooth, sigma=sigma)
        if spectrum_type == 'fft':
            self.show_fft(show_freqs=show_freqs, log=log, smooth=smooth, sigma=sigma)
            filtered_lfp = LFP(self.bp_filter(f1, f2, inplace=False, filter_order=filter_order), self.sf, 'Filtered LFP')
            filtered_lfp.show_fft(show_freqs=show_freqs, log=log, smooth=smooth, sigma=sigma)
        
        
    def show_signal(self, signal_length, new_figure=True, amplifier=1):
        """Shows plot of signal with length in samples (cut)"""
        time = np.arange(len(self.data))/self.sf
        if new_figure:
            plt.figure(figsize=(18, 4))
        plt.plot(time[:signal_length], self.data[:signal_length] * amplifier)
        
        
    def preprocess(self):
        self.remove_50hz_harmonics(Q=50, inplace=True)
        self.bp_filter(2, 999, inplace=True)
        
        
    def save(self, patient_dir):
        t = time.time() 
        path = os.path.join(patient_dir, "lfp")
        print(f"Creating {path} folder")
        try:
            os.mkdir(path)
        except OSError:
            print(f" Folder {path} already exists")
            
        self.root_dir = path
        print("[UPDATED] self.root_dir")
        
        self.create_name()
        filepath = os.path.join(self.root_dir, self.name + ".pkl")
        self.pickle_filepath = filepath
        print(f"Saving {self.name} object to {filepath} ...")
        with open(filepath, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print(f"Done, {time.time() - t} sec")
        
        
    def create_name(self):
        self.name = f"LFP_{self.patient_name}_{self.condition}_{self.placement}_{self.duration}"
        return self.name