import random
import numpy as np
import os
import sys
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import utils 

def zero_pad(data, length):
    extended = np.zeros(length)
    siglength = np.min([length, data.shape[0]])
    extended[:siglength] = data[:siglength]
    return extended

class PhysioNetDataset(Dataset):
    """
    Handles all aspects of the data.
    """

    def __init__(self, data_type):
        """
        Args:
            data_type: (string) 'data' or 'data_small'
        """
        data_dir = os.path.join('../aws_bucket', data_type)
        self.data_dir = data_dir
        self.labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
        self.length = len(self.labels)
    
    def __len__(self):
        return 2*self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # determine channel
        channel = 0
        if idx >= self.length:
            channel = 1
            idx = idx % self.length
        
        # load sample
        sample_name = 'sample' + str(idx) + '.csv'
        csv_name = os.path.join(self.data_dir, sample_name)
        csv_val = (pd.read_csv(csv_name, header=None)).values

        # extend and transform
        ecg = zero_pad(csv_val[:,channel], length = 30000)

        # normalize spectrogram
        sx_norm = (ecg - np.mean(ecg)) / np.std(ecg)
        
        sample = {'sx': sx_norm, 'label': self.labels.iloc[idx, 0]}

        return sample
