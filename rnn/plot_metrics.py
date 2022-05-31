import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import utils

parser = argparse.ArgumentParser()
parser.add_argument('trainfile')
parser.add_argument('valfile')
arg = parser.parse_args()

df_train = pd.read_csv(arg.trainfile)
df_val = pd.read_csv(arg.valfile)
num_batches = len(df_train)
bpe_train = 318
bpe_val = 38
x_train = np.linspace(0, len(df_train) // bpe_train * 10, len(df_train))
x_val = np.linspace(0, len(df_val) // bpe_val, len(df_val))

fig, axs = plt.subplots(2, 1, figsize=(20,16))
loss_train = df_train["loss"].rolling(20, center=True).mean().values
accuracy_train = df_train["accuracy"].rolling(20, center=True).mean().values
accuracy_val = df_val["accuracy"].rolling(38).mean().values
axs[0].plot(x_train, loss_train, label='Train', linewidth=4)
axs[1].plot(x_train, accuracy_train, label='Train', linewidth=4)
axs[1].plot(x_val, accuracy_val, label='Test', linewidth=4)


axs[0].set_title('Train Loss', fontsize=30)
axs[1].set_title('Accuracy', fontsize=30)
axs[1].set_xlabel('Num epochs', fontsize=30)

axs[0].tick_params(axis='both', labelsize=25)
axs[1].tick_params(axis='both', labelsize=25)
axs[1].legend(fontsize=25)
plt.show()
