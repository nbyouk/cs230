import random
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
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # load sample
        sample_name = 'sample' + str(idx) + '.csv'
        csv_name = os.path.join(self.data_dir, sample_name)
        csv_val = (pd.read_csv(csv_name, header=None)).values

        # transform
        sx = utils.spectrogram(np.expand_dims(dsv_val[:, 0], axis=0))[2]

        # normalize spectrogram
        sx_norm = (sx - np.mean(sx)) / np.std(sx)
        sample = {'sx': sx_norm, 'label': self.labels.iloc[idx, 0]}

        return sample
