import random
import os
import argparse
import csv
import glob
from re import I
import numpy as np
from numpy import squeeze
import pandas as pd
import scipy.io

'''
Converting the cspc2018_challenge score to .csv format for CS230 AFIB project
We'll keep only NSR and AFIB results: 1 = Normal; 2 = AF.
'''

def extract_ecg(mat_id, mat_files_path):
    """
    Take the MATLAB formated file, extract two channels and make it 
    Name the file as sample###.csv. 
    """
    mat_file = mat_id + ".mat"
    get_mat_file = os.path.join(mat_files_path, mat_file) 
    if os.path.exists(get_mat_file):
        mat = scipy.io.loadmat(get_mat_file, squeeze_me = True)
        all_ecg = mat['ECG']['data'].item()
        ch1 = np.transpose(all_ecg[0,:])
        ch2 = np.transpose(all_ecg[1,:])
        ecg_to_save = np.stack((ch1, ch2), axis = 1)
        return ecg_to_save


def convertCPSC_data(mat_files_path = "allTrainingSet", labels_filename = "labels_allCPSC.csv", next_sample_id = 0):
    # references_full_path = os.path.join(reference_base_path, "REFERENCE.CSV")
    saving_location = "/Users/kellybrennan/Documents/Stanford/cs230/aws_bucket/data"
    references = pd.read_csv("REFERENCE.csv")
    print(references.head(n = 10))

    with open(labels_filename, 'w') as csvfile: 
        writer = csv.writer(csvfile)
        for idx, row in references.iterrows():
            if row['First_label'] == 1 and np.isnan(row['Second_label']): 
                label = 0 #NSR
                ecg = extract_ecg(row['Recording'], mat_files_path)
                if np.any(ecg):
                    writer.writerow([label])
                    pd.DataFrame(ecg).to_csv(saving_location + "/sample" + str(next_sample_id) + '.csv', index = False)
                    next_sample_id += 1
            elif row['First_label'] == 2 and np.isnan(row['Second_label']):  
                label = 1 #AFib 
                ecg = extract_ecg(row['Recording'], mat_files_path)
                if np.any(ecg):
                    writer.writerow([label])
                    pd.DataFrame(ecg).to_csv(saving_location + "/sample" + str(next_sample_id) + '.csv', index = False)
                    next_sample_id += 1
            else:
                continue
        csvfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    result = convertCPSC_data()
