# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:32:13 2021

@author: Ksenia
"""
# 1.
""" extract ON and OFF intervals from bdf and save cropped data to files """

import mne
import pandas as pd

path = 'D:/[Patients]/PostOperatePD/' #path to bdf data 
out_path = 'D:/ksayfulina/parts_extracted_by_condition/' #path to save cropped data

# files_table = pd.read_csv('D:/ksayfulina/table_with_factors.csv')
files_table = pd.read_csv('D:/ksayfulina/table_with_factors_updated_new_only.csv')


subjects = files_table['surname'].drop_duplicates()
subjects = subjects.tolist()


# subject = 'Otroda'
# state = 'ON'
# day = 5


for subject in subjects:
    for state in ['ON','OFF']:
        for day in [1,5]:
            if day==5 and subject=='Otroda':
                continue
            subset=files_table[(files_table["surname"]==subject) & (files_table["state"]==state) & (files_table["day"]==day)]
            filename = subset.iloc[0]['filename']
            
            
            print(filename)
            
            #load raw file
            raw = mne.io.read_raw_bdf(path+filename, eog=None, misc=None, stim_channel='auto',
                                      exclude=(), preload=False, verbose=None)
        
            for eyes_state in subset['eyes'].drop_duplicates().tolist():
                print(eyes_state)
                
                tmin = subset[(subset['eyes']==eyes_state) & (subset['part']=='start')]
                tmin = tmin['time'].iloc[0]
                
                tmax = subset[(subset['eyes']==eyes_state) & (subset['part']=='end')]
                tmax = tmax['time'].iloc[0]

                
                
                raw_part = raw.copy().crop(tmin, tmax, include_tmax=False)
                
                fname_to_save = out_path + subject + '_' + state + '_' + str(day) + '_eyes_' + eyes_state + '.fif'
                
                raw_part.save(fname_to_save, picks=None, 
                              verbose='Error', overwrite=True)
                
        
       
                


########




