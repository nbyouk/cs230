from calendar import c
import json
import logging
import os
from re import I
import shutil
import torch
import numpy as np
from scipy import signal
import scipy.io as sio
import matplotlib.pyplot as plt

import torch


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'af

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def spectrogram(data, fs=500, nperseg=64, noverlap=32):
    f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = np.transpose(Sxx, [0, 2, 1])
    Sxx = np.abs(Sxx)
    mask = Sxx > 0
    Sxx[mask] = np.log(Sxx[mask])
    return f, t, Sxx

def zero_pad(data, length):
    extended = np.zeros(length)
    siglength = np.min([length, data.shape[0]])
    extended[:siglength] = data[:siglength]
    return extended

def plot_loss_acc(epochs, loss, acc):
    """
    Args:
        list of epoch indices
        list of losses: index corresponds with the epoch
        list of accuracy measurements: index corresponds with the epoch
    """
    plt.subplot(1,2,1)
    plt.plot(epochs, loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.subplot(1,2,2)
    plt.plot(epochs, acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

# def obtain_contrastive_loss(ch1, ch2):
#     """ Calculate NCE Loss For Latent Embeddings in Batch 
#     Args:
#         latent_embeddings1 (torch.Tensor): embeddings from model for different perturbations of same instance (BxHxN). Channel 1
#         latent_embeddings1 (torch.Tensor): embeddings from model for different perturbations of same instance (BxHxN). Channel 1 
#     Outputs:
#         loss (torch.Tensor): scalar NCE loss 
#     """
#     #Calculate cosine similarity
#     cos = nn.CosineSimilarity()
#     cosine_similarity = cos(ch1, ch2)

#     #mask out 
#     return loss

# def obtain_contrastive_loss2(ch1, ch2):
#     """ Calculate NCE Loss For Latent Embeddings in Batch 
#     Args:
#         latent_embeddings1 (torch.Tensor): embeddings from model for different perturbations of same instance (BxHxN). Channel 1
#         latent_embeddings1 (torch.Tensor): embeddings from model for different perturbations of same instance (BxHxN). Channel 1 
#     Outputs:
#         loss (torch.Tensor): scalar NCE loss 
#     """
#     #nviews1 = set(range(ch1.shape[2])) #Should be the same shape
#     #view_combinations1 = combinations(nviews1,2)
#     #nviews2 = set(range(ch2.shape[2])) #Should be the same shape
#     #view_combinations2 = combinations(nviews2,2)
#     loss = 0
#     ncombinations = 1

#     norm1_vector = ch1.norm(dim=1).unsqueeze(0)
#     norm2_vector = ch2.norm(dim=1).unsqueeze(0)
#     sim_matrix = torch.mm(ch1, ch2.transpose(0,1))
#     norm_matrix = torch.mm(norm1_vector.transpose(0,1),norm2_vector)
#     temperature = 0.1
#     argument = sim_matrix/(norm_matrix*temperature)
#     sim_matrix_exp = torch.exp(argument)

#     diag_elements = torch.diag(sim_matrix_exp)

#     tri1_sum = torch.sum(sim_matrix_exp,1) # Not too sure about this
#     tri2_sum = torch.sum(sim_matrix_exp,0)

#     loss_diag1 = -torch.mean(torch.log(diag_elements/tri1_sum))
#     loss_diag2 = -torch.mean(torch.log(diag_elements/tri2_sum))

#     #loss_tri1 = -torch.mean(torch.log(sim_matrix_exp/triu_sum))
#     #loss_tri2 = -torch.mean(torch.log(sim_matrix_exp/tril_sum))     

#     loss = loss_diag1 + loss_diag2
#     loss_terms = 2

#     loss = loss/(loss_terms*ncombinations)
#     return loss
