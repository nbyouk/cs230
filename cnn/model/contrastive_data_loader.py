import random
from re import I
import numpy as np
import os
import sys
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import utils 


class PhysioNetDataset(Dataset):
    """
    Handles all aspects of the data.
    """
    def __init__(self, data_type):
        """
        Args:
            data_type: (string) 'train', 'train/small', 'dev' or 'test'
        """
        data_dir = os.path.join('../aws_bucket', data_type)
        self.data_dir = data_dir
        self.labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
        self.length = len(self.labels)
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # determine channel
        # channel = 0
        # if idx >= self.length:
            # channel = 1
            # idx = idx % self.length
        
        # load sample
        sample_name = 'sample' + str(idx) + '.csv'
        csv_name = os.path.join(self.data_dir, sample_name)
        csv_val = (pd.read_csv(csv_name, header=None)).values

        # transform
        # sx = utils.spectrogram(np.expand_dims(csv_val[:, channel], axis=0))[2]

        # normalize spectrogram
        # sx_norm = (sx - np.mean(sx)) / np.std(sx)
        
        # sample = {'sx': sx_norm, 'label': self.labels.iloc[idx, 0]}

       #Trying to get both channels
        sx1 = utils.spectrogram(np.expand_dims(csv_val[:,0], axis=0))[2]
        sx2 = utils.spectrogram(np.expand_dims(csv_val[:,1], axis=0))[2]

        # normalize spectrogram
        sx1_norm = (sx1 - np.mean(sx1)) / np.std(sx1) 
        sx2_norm = (sx2 - np.mean(sx2)) / np.std(sx2)

        #Implement contrastive function:
        sample = {'sx': sx1_norm, 'sx2': sx2_norm, 'label': self.labels.iloc[idx, 0]}

        return sample
