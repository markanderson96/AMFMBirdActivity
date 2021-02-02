import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf

import yaml
from pathlib import Path

import matplotlib.pyplot as plt

project_dir = Path(__file__).resolve().parents[2]
# load config file
with open(str(project_dir) + '/config/config.yaml') as file:
    config = yaml.safe_load(file)

# dataset = config['general']['dataset']
# features = 'data/features/' + dataset + '/features.csv'

# df = pd.read_csv(features, index_col=None)
# df = df.drop(['Unnamed: 0', 'fileIndex', 'hasBird'], axis=1)
# df = df.dropna()
# df = df.drop_duplicates()

# fig0, ax0 = plt.subplots()
# corr = df.corr()
# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# cmap = sns.diverging_palette(220, 20, as_cmap=True)
# sns.heatmap(data=corr, annot=True, center=0, cmap=cmap, square=True, mask=mask)
# fig0.suptitle("Diagonal Correlation HeatMap")

time_series = np.genfromtxt('data/interim/nips4b/env/nips4b_birds_trainfile016_env_info.csv')
am_info = np.genfromtxt('data/interim/nips4b/env/nips4b_birds_trainfile016_AM_info.csv')
env = np.genfromtxt('data/interim/nips4b/env/nips4b_birds_trainfile016_env.csv')
sound_data, Fs = sf.read('data/processed/nips4b/nips4b_birds_trainfile016.wav')


amDetected = am_info[0:11]
amFreq = am_info[11:22]
amDepth = am_info[22:33]
amProminence = am_info[33:44]

t_s = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
ax.plot(np.linspace(0,6,Fs*6), sound_data)
ax.set_title('Sound Signal')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Amplitude')
ax.grid(True, which='both')
ax.set_xlim(left=0,right=6)

fig, ax = plt.subplots()
ax.plot(np.linspace(0,6,600), env)
ax.set_title('Hilbert Transform of Signal')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Amplitude')
ax.grid(True, which='both')

fig, axs = plt.subplots(4, 3)
fig.suptitle('Time series of extracted envelope information')
fig.tight_layout()
for i in range(0, len(time_series), 100):
    iy = int((i/100) // 4)
    ix = int((i/100) % 4)
    axs[ix, iy].plot(t_s, time_series[i:i + 100])
    axs[ix, iy].grid(True, which='both', axis='both')
    axs[ix, iy].set_title('Time Starting {}'.format(i/200))
    axs[ix, iy].set_xlabel('Time [s]'); axs[ix, iy].set_ylabel('Amplitude')
    axs[ix, iy].set_xlim(left=0, right=1); axs[ix,iy].set_ylim(bottom=0, top=0.15)

fig, ax = plt.subplots()
ax.set_axis_off()
columns = ['amDetected', 'amFreq', 'amDepth', 'amProminence']
rows = ['0.0-1.0', '0.5-1.0', '1.0-2.0', '1.5-2.5', '2.0-3.0', '2.5-3.5',
        '3.0-4.0', '3.5-4.5', '4.0-5.0', '4.5-5.5', '5.0-6.0']
content = np.transpose(np.array([amDetected, amFreq, amDepth, amProminence]))
table = ax.table(cellText=content, rowLabels=rows, colLabels=columns, loc='center')
ax.set_title('Table of Results')

plt.show()