import numpy as np
import pandas as pd

#!pip install pyEDFlib
import pyedflib
#!pip install ipympl

from scipy.fftpack import fft, ifft, fftfreq
from scipy import signal as sg
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from scipy.stats import binned_statistic, entropy, norm
from scipy import stats
from scipy import ndimage

from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from fooof import FOOOF
import fooof

import sys
import os
import time
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
from utils_io import *

from itertools import combinations
import networkx as nx

def compute_matrix_norm(matrix, norm='p', p=2):
    if norm == 'max':
        return np.max(np.max(matrix))
    if norm == 'p':
        return np.sum(np.sum(matrix ** p)) ** (1/p)

def mask_nonclustered(sig_pac, min_cluster_size=3):
    """
    Takes significant pac_matrix 
    Returns pac_matrix with non-clustered (presumably spurious) pac values zeroed out
    Minimal cluster size is set default to 3

    """
    features, n_features = ndimage.label(sig_pac)
    # get cluster size for each feature
    cluster_size = {i: np.sum(features == i) for i in range(1, n_features + 1)}
    # getting the mask to zero-out small (size < min_cluster_size) cluster
    # True if cluster is too small.
    zero_mask = np.zeros_like(features).astype(bool)
    for i_feature in range(1, n_features + 1):
        if cluster_size[i_feature] < min_cluster_size:
            zero_mask[features == i_feature] = True
    sig_pac[zero_mask] = 0      
    return sig_pac

    
def get_sig_pac(pac, significance=0.01, mask_lonely=True, min_cluster_size=3):
    sig_pac = pac.pac_matrix.copy()
    sig_pac[pac.pvalues > significance] = 0
    if mask_lonely:
        sig_pac = mask_nonclustered(sig_pac, min_cluster_size=min_cluster_size)
    return sig_pac


def get_bigraph(patient, 
                condition, 
                signficance=0.01, 
                norm_type='p', 
                p=0.9, 
                norm_threshold=2e-3, 
                cross_placements=None):
    """
    Computes adjacency matrix for a given patient and condition
    - Significant (p = significance (default=0.01))
    - With 0-valued non-clustered PAC values (no non-zero significant neighbours) (See mask_nonclustered function)
    - Using matrix p-norm
    - For cross-PACs the whole matrix is take, for the inter-PACs only the corresponding part (10-35, 160-500)
    Returns networkx directed Graph object. Phase-giving placements are marked as "-p" and amplitude as "-a"
    """
    
    # 1. Initialize placements for bi-graph (default: None)
    if cross_placements is None:
        cross_placements = ["L4-3A", "L4-3B", "L4-3C", "L2A-3A", "L2B-3B", "L2C-3C", "L1-2A", "L1-2B", "L1-2C", 
                            "R4-3A", "R4-3B", "R4-3C", "R2A-3A", "R2B-3B", "R2C-3C", "R1-2A", "R1-2B", "R1-2C"]
        
    n = len(cross_placements)
    adjacency_matrix = np.zeros(shape=(n, n))
    
    # 2. Compute adjacency matrix
    for i in range(n):
        for j in range(n):
            phase = cross_placements[i]
            amplitude = cross_placements[j]
            if not pac_exists(patient, condition, phase, amplitude):
                    continue
            if not pac_exists(patient, condition, phase, amplitude):
                    continue
            pac = patient.pac[condition][phase][amplitude]
            
            # getting only significant PAC + leaving only "clustered" entries
            sig_pac = get_sig_pac(pac, significance=0.01, mask_lonely=True)
            norm = compute_matrix_norm(sig_pac, norm=norm_type, p=p)
            
            if phase != amplitude:
                norm = compute_matrix_norm(sig_pac, norm=norm_type, p=p)
            else:
                beta10 = np.argmin(np.abs(pac.beta_freqs - 10))
                beta35 = np.argmin(np.abs(pac.beta_freqs - 35))
                hfo140 = np.argmin(np.abs(pac.hfo_freqs - 140))
                norm = compute_matrix_norm(sig_pac[hfo140:, beta10:beta35 + 1], norm=norm_type, p=p)
            if norm > norm_threshold:
                adjacency_matrix[i, j] = np.round(norm * 1e3, 2)
    
    # 3. Initialize graph and nodes
    G = nx.DiGraph()

    nodes_left = [node + '-p' for node in cross_placements]
    nodes_right = [node + '-a' for node in cross_placements]
    
    
    G.add_nodes_from(nodes_left, bipartite=0)
    G.add_nodes_from(nodes_right, bipartite=1)
    
    # 4. Add edges from adjacency matrix
    for i in range(n):
        for j in range(n):
            weight = adjacency_matrix[i, j]
            if weight != 0:
                G.add_edge(nodes_left[i], nodes_right[j], weight=weight)
                
    return G

    
def get_power(y, power_type='p', p=2):
    """
    max, mean, p, std
    """
    if power_type == 'max':
        return np.max(y)
    if power_type == 'mean':
        return np.mean(y)
    if power_type == 'p':
        return np.mean(y ** p) ** (1/p)
    if power_type == 'std':
        return np.std(y ** p) ** (1/p)

    
def get_beta_power(lfp: LFP, freqs=(14, 35), power_type='max', p=2):

    # dividing by std
    lfp_data = lfp.data.copy()
    lfp_data = lfp_data / np.std(lfp_data)
    lfp_norm = LFP(lfp_data, lfp.sf, lfp.patient_name, lfp.condition, lfp.placement)
    
    # getting PSD between 2 to 50 Hz
    f, psd = lfp_norm.get_psd()
    
    mask1= (f >= 2) & (f < 50)

    # getting FOOOF peak fit (without aperiodic component)
    fm = FOOOF(peak_width_limits=(1, 12), max_n_peaks=4, peak_threshold=1, verbose=False)
    fm.fit(f[mask1], psd[mask1])
    y_hat = fm._peak_fit

    # applying only to desired freq range (10 - 35) by default
    f0, f1 = freqs
    final_mask = (f[mask1] >= f0) & (f[mask1] <= f1)
    
    n_peaks = fm.peak_params_.shape[0]
    peak_freqs = fm.peak_params_[:, 0]
    peak_heights = fm.peak_params_[:, 1]

    #beta_power = np.max(peak_heights[(peak_freqs >= freqs[0]) & (peak_freqs <= freqs[1])])
    if n_peaks == 0:
        return 0 
    peak_powers = np.zeros(n_peaks)
    for i in range(n_peaks):
        peak_freq = fm.peak_params_[i, 0]
        if (peak_freq >= freqs[0]) and (peak_freq <= freqs[1]):
            peak_powers[i] = fm.peak_params_[i, 1]   
    return np.max(peak_powers)

def get_hfo_power(lfp: LFP, freqs=(160, 500), power_type='max', p=2):
    
    # dividing by std
    lfp_data = lfp.data.copy()
    lfp_data = lfp_data / np.std(lfp_data)
    lfp_norm = LFP(lfp_data, lfp.sf, lfp.patient_name, lfp.condition, lfp.placement)
    
    f, y = lfp_norm.get_psd(smooth=True, sigma=2)

    # masking
    mask = (f > freqs[0]) & (f < freqs[1])
    f_hat, y_hat = f[mask], y[mask]
    
    # baseline correction
    y_hat -= np.min(y_hat)
    
    # smoothing
    sigma=3
    y_hat = gaussian_filter1d(y_hat, sigma)
    
    # calculating power
    hfo_power = get_power(y_hat, power_type, p)
    
    return hfo_power



def get_power_values(patient, verbose=True):
    """
    Calculates power on one patient. 
    Returns to dictionaries with condition as key
    Values: list of power values corresponding to each electrode location
    beta_powers: dict, hfo_powers: dict
    """
    cross_placements = ["L4-3A", "L4-3B", "L4-3C", "L2A-3A", "L2B-3B", "L2C-3C", "L1-2A", "L1-2B", "L1-2C", 
                        "R4-3A", "R4-3B", "R4-3C", "R2A-3A", "R2B-3B", "R2C-3C", "R1-2A", "R1-2B", "R1-2C"]
    
    if verbose: print(f"Estimating spectral power for {patient.name}")
    
    n = len(cross_placements)
    
    conditions = ["1Day OFF Rest 180sec", "1Day ON Rest 180sec", "5Day OFF Rest 180sec", "5Day ON Rest 180sec"]
    
    beta_powers = {cond:[] for cond in conditions if cond in patient.conditions} # condition -> list of powers corresponding to the cross_placement
    hfo_powers = {cond:[] for cond in conditions if cond in patient.conditions}
    
    for condition in beta_powers.keys():
        if "Rest 180sec" not in condition:
            continue
        if verbose: print(condition)
        iterable = enumerate(cross_placements)
        if verbose: iterable = tqdm(enumerate(cross_placements))
        for i, placement in iterable:
            lfp = patient.lfp[condition][placement]
            beta_powers[condition].append(get_beta_power(lfp))
            hfo_powers[condition].append(get_hfo_power(lfp))
            
    return beta_powers, hfo_powers


def get_power_margins(beta_powers, hfo_powers):
    """
    Input: dict of power values of a single patient
    Output: margins: (b1, b2), (h1, h2)
    """
    beta_powers_total = []
    hfo_powers_total = []
    for key in beta_powers.keys():
        beta_powers_total.append(beta_powers[key])
        hfo_powers_total.append(hfo_powers[key])
    
    b1, b2 = np.min(beta_powers_total), np.max(beta_powers_total)
    h1, h2 = np.min(hfo_powers_total), np.max(hfo_powers_total)
    
    return (b1, b2), (h1, h2)


def get_global_power_margins(margins_list):
    bmin, bmax = +np.inf, 0
    hmin, hmax = +np.inf, 0
    for margins in margins_list:
        (b1, b2), (h1, h2) = margins
        if b1 < bmin:
            bmin = b1
        if b2 > bmax:
            bmax = b2
        if h1 < hmin:
            hmin = h1
        if h2 > hmax:
            hmax = h2
    return (bmin, bmax), (hmin, hmax)


def get_node_colors(patient, 
                    condition, 
                    beta_powers, 
                    hfo_powers, 
                    power_margins, 
                    palette=sns.color_palette("magma", as_cmap=True)):
    """
    power_margins: ((b1, b2), (h1, h2))
    """
    # 1. acquire powers
    # 2. convert them to 0 - 1 values using provided margins
    # 3. use color palette to map [0, 1] -> color
    cross_placements = ["L4-3A", "L4-3B", "L4-3C", "L2A-3A", "L2B-3B", "L2C-3C", "L1-2A", "L1-2B", "L1-2C", 
                        "R4-3A", "R4-3B", "R4-3C", "R2A-3A", "R2B-3B", "R2C-3C", "R1-2A", "R1-2B", "R1-2C"]
    
    n = len(cross_placements)
    
    (b1, b2), (h1, h2) = power_margins
    
    colors_left = [0 for _ in range(n)]
    colors_right = [0 for _ in range(n)]
    
    for i, placement in enumerate(cross_placements):
        
        beta_power = beta_powers[condition][i]
        hfo_power = hfo_powers[condition][i]
        
        beta_norm = (beta_power - b1)/(b2 - b1)
        hfo_norm = (hfo_power - h1)/(h2 - h1)
        
        colors_left[i] = palette(beta_norm)
        colors_right[i] = palette(hfo_norm)
        
    return colors_left + colors_right


def draw_bigraph(G, ax, colors=None, weight_divider=8):
    cross_placements = ["L4-3A", "L4-3B", "L4-3C", "L2A-3A", "L2B-3B", "L2C-3C", "L1-2A", "L1-2B", "L1-2C", 
                        "R4-3A", "R4-3B", "R4-3C", "R2A-3A", "R2B-3B", "R2C-3C", "R1-2A", "R1-2B", "R1-2C"]
    n = len(cross_placements)
    # get node names
    nodes_left = [node + '-p' for node in cross_placements]
    nodes_right = [node + '-a' for node in cross_placements]
    
    # get node positions
    y2 = np.array([0.1, 0.18, 0.26, 0.42, 0.5, 0.58, 0.74, 0.82, 0.9])[::-1]
    y1 = -y2[::-1]
    pos_y = np.concatenate((y1, y2)).reshape(-1, 1)
    #pos_y = np.concatenate((- np.linspace(-0.85, -0.1, n // 2), - np.linspace(0.1, 0.85, n // 2))).reshape(-1, 1)
    pos_left = np.hstack((-0.5 * np.ones((n, 1)), pos_y))
    pos_right = np.hstack((0.5 * np.ones((n, 1)), pos_y)) 

    positions_left = {node:position for node, position in zip(nodes_left, pos_left)}
    positions_right = {node:position for node, position in zip(nodes_right, pos_right)}

    pos = {**positions_left, **positions_right}
    
    weights = np.array([G[u][v]['weight'] for u,v in G.edges()])
    
    if colors is None:
        aquamarine = np.array([127, 125, 212]) / 256
        goldenrod = np.array([255, 193, 37]) / 256
        colors = (aquamarine, goldenrod)
        node_color = [colors[0] for _ in range(n)] + [colors[1] for _ in range(n)]
        
    else:
        node_color=colors

    nx.draw_networkx(G, 
                     ax=ax,
                     pos=pos, 
                     arrows=True, 
                     arrowstyle='->', 
                     width=weights / weight_divider, 
                     node_size=500, 
                     node_color=node_color, alpha=0.8)
    
    ax.margins(0.1, 0)