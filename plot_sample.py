import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
from cnn import utils

parser = argparse.ArgumentParser()
parser.add_argument('indices', metavar='N', type=int, nargs='+')
arg = parser.parse_args()

data_dir = 'aws_bucket/data'
labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'), header=None).values

# 10s snippets of data
fig, axs = plt.subplots(2, 2, figsize=(20,16))

sample_name = 'sample' + str(arg.indices[0]) + '.csv'
csv_name = os.path.join(data_dir, sample_name)
sample = (pd.read_csv(csv_name, header=None)).values
label = labels[arg.indices[0]]
axs[0,0].plot(np.arange(0, 7500)/500, utils.zero_pad(sample[:, 0], 7500)[:7500]) #Channel 1
axs[0,0].plot(np.arange(0, 7500)/500, utils.zero_pad(sample[:, 1], 7500)[:7500]) #Channel 2
axs[0,0].set_title('Physionet-MIT, AFIB: ' + str(labels[arg.indices[0], 0]), fontsize=20)

sample_name = 'sample' + str(arg.indices[1]) + '.csv'
csv_name = os.path.join(data_dir, sample_name)
sample = (pd.read_csv(csv_name, header=None)).values
label = labels[arg.indices[1]]
axs[0,1].plot(np.arange(0, 7500)/500, utils.zero_pad(sample[:, 0], 7500)[:7500]) #Channel 1
axs[0,1].plot(np.arange(0, 7500)/500, utils.zero_pad(sample[:, 1], 7500)[:7500]) #Channel 2
axs[0,1].set_title('Physionet-MIT, AFIB: ' + str(labels[arg.indices[1], 0]), fontsize=20)

sample_name = 'sample' + str(arg.indices[2]) + '.csv'
csv_name = os.path.join(data_dir, sample_name)
sample = (pd.read_csv(csv_name, header=None)).values
label = labels[arg.indices[2]]
axs[1,0].plot(np.arange(0, 7500)/500, utils.zero_pad(sample[:, 0], 7500)[:7500]) #Channel 1
axs[1,0].plot(np.arange(0, 7500)/500, utils.zero_pad(sample[:, 1], 7500)[:7500]) #Channel 2
axs[1,0].set_title('CPSC, AFIB: ' + str(labels[arg.indices[2], 0]), fontsize=20)

sample_name = 'sample' + str(arg.indices[3]) + '.csv'
csv_name = os.path.join(data_dir, sample_name)
sample = (pd.read_csv(csv_name, header=None)).values
label = labels[arg.indices[3]]
axs[1,1].plot(np.arange(0, 7500)/500, utils.zero_pad(sample[:, 0], 7500)[:7500]) #Channel 1
axs[1,1].plot(np.arange(0, 7500)/500, utils.zero_pad(sample[:, 1], 7500)[:7500]) #Channel 2
axs[1,1].set_title('CPSC, AFIB: ' + str(labels[arg.indices[3], 0]), fontsize=20)

for ax in axs.flat:
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylabel('Potential (mV)', fontsize=20)

plt.show()

# 10s snippets of data
fig, axs = plt.subplots(2, 2, figsize=(20,16))

sample_name = 'sample' + str(arg.indices[0]) + '.csv'
csv_name = os.path.join(data_dir, sample_name)
sample = (pd.read_csv(csv_name, header=None)).values
label = labels[arg.indices[0]]
y = utils.spectrogram(np.expand_dims(utils.zero_pad(sample[:, 0], 30000), axis=0))[2]
axs[0,0].imshow(np.transpose(y)[:,:240], aspect='auto', cmap='jet', extent=[0, 15, 33, 0]) #Channel 1
axs[0,0].set_title('Physionet-MIT, AFIB: ' + str(labels[arg.indices[0], 0]), fontsize=20)

sample_name = 'sample' + str(arg.indices[1]) + '.csv'
csv_name = os.path.join(data_dir, sample_name)
sample = (pd.read_csv(csv_name, header=None)).values
label = labels[arg.indices[1]]
y = utils.spectrogram(np.expand_dims(utils.zero_pad(sample[:, 0], 30000), axis=0))[2]
axs[0,1].imshow(np.transpose(y)[:,:240], aspect='auto', cmap='jet', extent=[0, 15, 33, 0]) #Channel 1
axs[0,1].set_title('Physionet-MIT, AFIB: ' + str(labels[arg.indices[1], 0]), fontsize=20)

sample_name = 'sample' + str(arg.indices[2]) + '.csv'
csv_name = os.path.join(data_dir, sample_name)
sample = (pd.read_csv(csv_name, header=None)).values[:30000]
label = labels[arg.indices[2]]
y = utils.spectrogram(np.expand_dims(utils.zero_pad(sample[:, 0], 30000), axis=0))[2]
axs[1,0].imshow(np.transpose(y)[:,:240], aspect='auto', cmap='jet', extent=[0, 15, 33, 0]) #Channel 1
axs[1,0].set_title('CPSC, AFIB: ' + str(labels[arg.indices[2], 0]), fontsize=20)

sample_name = 'sample' + str(arg.indices[3]) + '.csv'
csv_name = os.path.join(data_dir, sample_name)
sample = (pd.read_csv(csv_name, header=None)).values[:30000]
label = labels[arg.indices[3]]
y = utils.spectrogram(np.expand_dims(utils.zero_pad(sample[:, 0], 30000), axis=0))[2]
axs[1,1].imshow(np.transpose(y)[:,:240], aspect='auto', cmap='jet', extent=[0, 15, 33, 0]) #Channel 1
axs[1,1].set_title('CPSC, AFIB: ' + str(labels[arg.indices[3], 0]), fontsize=20)

for ax in axs.flat:
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylabel('Frequency (Hz)', fontsize=20)
plt.show()
