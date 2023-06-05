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

from numba import njit, prange
import concurrent.futures

from tqdm.notebook import tqdm
import itertools

from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from utility_functions import *
from lfp_class import LFP
from pac_class import MyPAC

class Patient:
    
    def __init__(self, name, root_dir, channels=None, sampling_frequency=2000):
        
        print("List of things to make sure before analysis: ")
        print("1) .bdf files are in patient folder (root_dir)")
        print("2) annotation files share the same name as .bdf files but with _annotations.txt suffix")
        print("3) annotations share the same naming principle: e.g. 1Day OFF RH (Com)")
        
        self.name = name
        self.root_dir = root_dir
        
        self.files = set()
        self.file_conditions = defaultdict(dict)
        self.file_annotations = {} # key = file, value = dataframe
        self.conditions = set()
        self.condition_durations = {}
        self.placements = set()
        self.sorted_placements = list() # from 1 to 4, first half - right, then left
        self.comments = set()
        
        self.lfp = defaultdict(dict) # self.lfp[condition][placement]
        self.pac = defaultdict(lambda: defaultdict(lambda: defaultdict(dict))) # pac[condition][placement_phase][placement_ampl]
        
        self.sf=sampling_frequency
        
        if channels is None:
            ch_names = ['R1', 'R2C', 'R2B', 'R2A', 'R3C', 'R3B', 'R3A', 'R4', \
                        'L1', 'L2C', 'L2B', 'L2A', 'L3C', 'L3B', 'L3A', 'L4']
            ch_indexes = range(9, 25)
            self.channels = {ch_idx: ch_name for (ch_idx, ch_name) in zip(ch_indexes, ch_names)}
        else:
            self.channels = channels
        
        self.ch_names = list(self.channels.values())
        self.ch_indexes = list(self.channels.keys())
        
    

    def get_preprocessed_lfps(self, formula='abc-extended', verbose=True):
        """
        MUST update self.file_conditions for each file BEFORE using this method!
        ### Use p.add_file('filename')
        ### p.file_condition('filename', condition_name, condition_duration)

        *** WHAT THIS FUNCTION DOES ***
        > For each file:
        1) get_signals(file) [default channels: 1, 2abc, 3abc, 4]
        2) get_bipolar_signals (using a predefined formulas)
        3) updates placements using formula (e.g. 2A-3A)
        For each placement and condition (of that file):
            a) adds LFP[condition][placement] instance to Patient instance
            b) removes 50Hz harmonics
            c) filters 4-999 Hz using bandpass
        > Finally it merges file_conditions into ONE self.conditions_durations

        We get self.lfps updated: all bipolar LFPs for each condition and placement preprocessed
        AND we get all placements and conditions: self.placements, self.conditions, self.conditions_durations 
        """

        for file in self.files:
            # for each file we have: conditions, condition_durations
            placements, bipolar_signals = self.get_bipolar_signals(*self.get_signals(os.path.join(self.root_dir,file)), formula=formula)
            self.placements = set(placements)
            for placement in tqdm(self.placements, desc='Creating LFPs'):
                data = bipolar_signals[placement] # SHOULD BE A DICT
                for condition in self.file_conditions[file].keys():
                    # reading file_conditions for timestamps
                    begin, end = self.file_conditions[file][condition]
                    sf = self.sf
                    begin *= sf
                    end *= sf
                    if verbose: print("Adding {} {} LFP, {}-{} sec".format(condition, placement, begin/sf, end/sf))
                    self.lfp[condition][placement] = LFP(data[int(begin):int(end)], sf, patient_name=self.name, condition=condition, placement=placement)
                    # preprocessing lfps
                    lfp = self.lfp[condition][placement]
                    lfp.remove_50hz_harmonics(70, inplace=True)
                    lfp.bp_filter(4, 999, inplace=True, filter_order=3)

            del(bipolar_signals)
    # making conditions normal again (union over "file" axis) 
        for file in self.files:
            self.condition_durations.update(self.file_conditions[file])
        self.conditions = set(self.condition_durations.keys())
        
         
    def get_signals(self, file, ch_nrs=None):
        t0 = time.time()
        filepath = os.path.join(self.root_dir, file)
        print(f"Started reading {filepath}")
        if ch_nrs is None:
            ch_nrs = self.ch_indexes
        print(f"Channels: {ch_nrs}")    
        signals, signal_headers, header = pyedflib.highlevel.read_edf(filepath, ch_nrs=self.ch_indexes, verbose=True)
        for i, signal_header in zip(self.ch_indexes, signal_headers):
            print(signal_header['label'])
            
        # IMPLEMENT CHANNEL RENAMING?
        sf = signal_headers[0]['sample_rate']
        print("Sampling frequency: ", sf)
        if sf != self.sf:
            q = int(sf/self.sf)
            print("Downsampling by the factor ", q)
            new_signals, new_signal_headers = downsample(signals, signal_headers, 8)
            print("New sampling frequency: ", sf/q)   
        else:
            new_signals, new_signal_headers = signals, signal_headers      
        t1 = time.time()   
        print("Reading done, {} sec".format(round(t1 - t0, 1)))    
        return new_signals, new_signal_headers   
    
    
    def get_bipolar_signals(self, signals, signal_headers, formula='abc'):
        """
        Returns bipolar_signals:dict, bipolar_signals_names:list
        
        """
       # ch_names = ['R1', 'R2C', 'R2B', 'R2A', 'R3C', 'R3B', 'R3A', 'R4', \
       #             'L1', 'L2C', 'L2B', 'L2A', 'L3C', 'L3B', 'L3A', 'L4']
        if formula == 'abc':
            substraction_pairs = [(1, 4), (2, 5), (3, 6), (9, 12), (10, 13), (11, 14)]
            # 2a-3a, 2b-3b, 2c-3c
        elif formula == 'abc-extended':
            substraction_pairs = [(0, 1), (0, 2), (0, 3), \
                                  (1, 2), (2, 3), (3, 1), \
                                  (1, 4), (2, 5), (3, 6), \
                                  (4, 5), (5, 6), (6, 4), \
                                  (7, 4), (7, 5), (7, 6), \
                                  (8, 9), (8, 10), (8, 11), \
                                  (9, 10), (10, 11), (11, 9), \
                                  (9, 12), (10, 13), (11, 14), \
                                  (12, 13), (13, 14), (14, 12), \
                                  (15, 12), (15, 13), (15, 14)]
        # 1-2a, 1-2b, 1-2c, abc-pairs, all 2x pairs, 3x pairs, 4-3a, 4-3b, 4-3c
        print("Started creating bipolar signals")
        print("")
        # CALCULATION STEP
        bip_sig_names = []
        bipolar_signals = dict()
        for idx_1, idx_2 in substraction_pairs:
            bip_sig_name = f"{self.ch_names[idx_1]}-{self.ch_names[idx_2][1:]}" # e.g. R2A-3A
            bip_sig_names.append(bip_sig_name)
            bipolar_signals[bip_sig_name] = signals[idx_1] - signals[idx_2]
            print(f"{bip_sig_name} created")
            
        self.sorted_placements = bip_sig_names
        return bip_sig_names, bipolar_signals
    
    
    def add_file(self, filename):
        self.files.add(filename)
        
        
    def find_bdf_files(self):
        folder = self.root_dir
        print(f"Looking for .bdf files in {folder}")
        bdf_files = [f for f in os.listdir(folder) if f[-3:] == 'bdf']
        print(f"Found {bdf_files}")
        self.files.update(bdf_files)
        return bdf_files
        
    
    def file_condition(self, file, condition_label, timestamps):
        self.file_conditions[file][condition_label] = timestamps
        
        
    def scan_file_annotations(self, file, update_file_conditions=False):
        annot_file = file[:-4] + "_annotations.txt"
        print("Reading ", annot_file)
        filepath = os.path.join(self.root_dir, annot_file)
        df = pd.read_csv(filepath, header=0, sep=';')
        # annots
        mask = []
        for i in range(len(df)):
            flag = False
            if type(df.loc[i, 'Annotation']) is str:
                if 'Day' in df.loc[i, 'Annotation']:
                    flag = True
            mask.append(flag)
            
        df_annot = df[mask].reset_index(drop=True)
        
        # Creating columns: Day, L-DOPA, Condition (State, Movement)
        df_annot['Day'] = df_annot['Annotation'].apply(lambda s: s.split(" ")[0])
        df_annot['L-DOPA'] = df_annot['Annotation'].apply(lambda s: s.split(" ")[1])
        df_annot['State'] = df_annot['Annotation'].apply(lambda s: s.split(" ")[2:])
        
        print(df_annot)
        
        self.file_annotations[file] = df_annot
        if update_file_conditions:
            self.annotations_to_file_conditions(file, df_annot)
        return df_annot
    
    
    def annotations_to_file_conditions(self, file, df):
        for i in range(len(df)):
            onset = df.loc[i, 'Onset']
            duration = df.loc[i, 'Duration']
            condition = df.loc[i, 'Annotation']
            self.file_condition(file, condition, (onset, onset + duration))
        print("Updated self.file_condition with annotations from file")
    
                                
    def condition(self, condition_label, timestamps):
        """
        label: condition label e.g. "1Day OFF Rest"
        timestamps: (begin, end) in seconds
        """
        self.condition_durations[condition_label] = timestamps
        self.conditions.add(condition_label)
    
    
    def merge_annotations(self):
        self.annotations = pd.concat([self.file_annotations[file] for file in self.files], axis=0)
        return self.annotations

    
    def merge_conditions(self, conditions_to_merge:list, new_condition_name, total_duration=180, remove_merged=False):
        """
        Gets list of condition names as input and for each placement:
        1) Merges them creating and adding the corresponding LFP with new_condition_name for Patient
        2) Makes sure the total duration of LFP is equals total_duration
        3) Deletes unused LFPs (optional). Removes corresponding keys in self.conditions (all conditions_to_merge)

        * Make sure LFPs for conditions_to_merge are computed!
        * Make sure there is data of necessary duration (sum(durations) >= total_duration)
        * If len(conditions_to_merge) == 1: skips the first step
        
        """
        self.conditions.add(new_condition_name)
        for placement in self.placements:
            lfps_to_merge = [self.lfp[condition][placement] for condition in conditions_to_merge]
            data_to_merge = [lfp.data for lfp in lfps_to_merge]
            new_data = np.concatenate(data_to_merge)[:int(total_duration*self.sf)]
            lfp = LFP(new_data, self.sf, self.name, new_condition_name, placement)
            verbose = False
            if placement == 'R2A-3A':
                verbose = True
            self.add_lfp(lfp, verbose=verbose)
        
        # removing unused LFPs
        if remove_merged:
            for condition in conditions_to_merge:
                self.conditions.remove(condition)
                for placement in self.placements:
                    self.remove_lfp(condition, placement)

            
        
    def display_all_annotations(self):
        df = self.merge_annotations()
        days = ["1Day", "5Day"]
        ldopas = ["OFF", "ON"]
        for day in days:
            print(f"----{day}----")
            for ldopa in ldopas:
                print(f"------{ldopa}------")
                mask = (df["Day"] == day) & (df["L-DOPA"] == ldopa)
                print(df[mask].loc[:, ["Onset", "Duration", "State"]])
                
        
    
    def add_lfp(self, lfp: LFP, verbose=True):
        """
        Adds lfp to patient instance lfp[condition][placement]
        Use correct condition and placement in lfp attributes!
        """
        condition, placement = lfp.condition, lfp.placement
        if verbose: print(f"Adding LFP to {self.name} object. \nCondition: {condition} \nPlacement: {placement}")
        self.lfp[condition][placement] = lfp
        if verbose: print("Updating condition")
        self.conditions.add(condition)
        
    
    def remove_lfp(self, condition, placement):
        """
        Removes LFP from the patient.lfp defaultdict to free up space
        """
        self.lfp[condition][placement] = None
        
        
        
    def add_lfp_deprecated(self, data, sf, condition, placement):
        """
        Writes lfp instance into a double indexed dictionary
        from (bipolar) data according to condition (begin, end) duration
        self.lfp[condition][placement]
        
        """
        raise Exception("Deprecated. The only way to do this is to use self.get_preprocessed_lfps()")
        begin, end = self.condition_durations[condition]
        
        begin *= sf
        end *= sf
        print("Adding {} {} LFP, {}-{} sec".format(condition, placement, begin/sf, end/sf))
        lfp = LFP(data[begin:end], sf, patient=self.name, condition=condition, placement=placement)
        self.lfp[condition][placement] = lfp
        
        
    def add_pac(self, pac_obj):
        """
        Adds pac object to triple-nested self.pac dictionary using pac_ojb.lfp_phase and lfp_amplitude
        
        """
        
        lfp_phase, lfp_amplitude = pac_obj.lfp_phase, pac_obj.lfp_amplitude
        
        assert (self.name == lfp_phase.patient_name)
        condition = lfp_phase.condition
        placement_phase = lfp_phase.placement
        placement_amplitude = lfp_amplitude.placement
        
        self.pac[condition][placement_phase][placement_amplitude] = pac_obj
        
        
    def load_pac(self, filepath=None, condition=None, phase_placement=None, ampl_placement=None, duration=None, verbose=True):
        pac_root_dir = os.path.join(self.root_dir, 'pac')
        if filepath is None:
            assert condition is not None, "If filepath is not specified other parameters should be given"
            name_components = [self.name, 
                               condition, 
                               phase_placement, 
                               ampl_placement, 
                               f"{duration} sec"]
            
            filepath = os.path.join(pac_root_dir, '_'.join(name_components) + '.pkl')   
            if verbose: print(f"Reading {filepath}") 
            with open(filepath, 'rb') as _input:
                pac = pickle.load(_input)

            self.pac[condition][phase_placement][ampl_placement] = pac
            
        else:
            # Having filepath read the condition and placements
            if verbose: print("There is a filepath! Great")
            if verbose: print(f"Reading {filepath}") 
            filename = os.path.basename(filepath)
            _, condition, phase_placement, ampl_placement, _ = retrieve_pac_name(filename) # without .pkl
            with open(filepath, 'rb') as _input:
                pac = pickle.load(_input)
            
            self.pac[condition][phase_placement][ampl_placement] = pac
             
        if verbose: print(f"Updated {self.name} pac.[{condition}][{phase_placement}][{ampl_placement}]")   
        return pac
    
    
    def load_all_pacs(self, verbose=True):
        for filename in os.listdir(os.path.join(self.root_dir, 'pac')):
            self.load_pac(os.path.join(self.root_dir, 'pac', filename), verbose=verbose)
            
              
    def comment(self, comment):
        self.comments.add(comment)
        
        
    def info(self):
        attributes = [a for a in dir(self) if not a.startswith('__')]
        for attribute in attributes:
            print(f"{attribute}: {self.__dict__[attribute]}")
            
    
    def __getstate__(self):
        attrs = self.__dict__.copy()
        keys_to_del = ['pac']
        print(f"Pickling {self.name} without {keys_to_del}")
        for key in keys_to_del:
            del attrs[key]
        return attrs   
            
            
    def save(self, filename=None):
        t = time.time()
        if filename is None:
            filename = self.name + ".pkl"
        filepath = os.path.join(self.root_dir, filename)
        self.pickle_filepath = filepath
        print(f"Saving {self.name} object to {filepath} ...")
        with open(filepath, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print(f"Done, {time.time() - t} sec")
        file_stats = os.stat(filepath)
        print(f'File size: {file_stats.st_size / (1024 * 1024)} MB')
        print("Returning filepath for saved file")
        return filepath
    
            