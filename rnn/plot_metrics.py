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
x = np.linspace(1, 20, 20)

fig, axs = plt.subplots(2, 1, figsize=(20,16))
loss_train = df_train["loss"]
accuracy_train = df_train["accuracy"]
accuracy_val = df_val["accuracy"]
axs[0].plot(x, loss_train, label='Train', linewidth=4)
axs[1].plot(x, accuracy_train, label='Train', linewidth=4)
axs[1].plot(x, accuracy_val, label='Test', linewidth=4)


axs[0].set_title('Train Loss', fontsize=30)
axs[1].set_title('Accuracy', fontsize=30)
axs[1].set_xlabel('Num epochs', fontsize=30)

axs[0].tick_params(axis='both', labelsize=25)
axs[1].tick_params(axis='both', labelsize=25)
axs[1].legend(fontsize=25)
plt.show()
