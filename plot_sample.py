import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--sample', default='0',
                    help="Sample number")
args = parser.parse_args()

idx = int(args.sample)

data_dir = 'aws_bucket/data'
labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'), header = None).values

sample_name = 'sample' + str(idx) + '.csv'
csv_name = os.path.join(data_dir, sample_name)
sample = (pd.read_csv(csv_name, header=None)).values
label = labels[idx]

plt.figure(figsize=(20,16))
plt.plot(np.arange(0, 5000)/500, sample[:5000, 0]) #Channel 1
plt.plot(np.arange(0, 5000)/500, sample[:5000, 1]) #Channel 2
plt.title('AFIB: ' + str(labels[idx, 0]), fontsize=30)
plt.show()
