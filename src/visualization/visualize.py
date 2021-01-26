import numpy as np
import pandas as pd
import seaborn as sns

import yaml
from pathlib import Path

import matplotlib.pyplot as plt

project_dir = Path(__file__).resolve().parents[2]
# load config file
with open(str(project_dir) + '/config/config.yaml') as file:
    config = yaml.safe_load(file)

dataset = config['general']['dataset']
features = 'data/features/' + dataset + '/features.csv'

df = pd.read_csv(features, index_col=None)
df = df.drop(['Unnamed: 0', 'fileIndex', 'hasBird'], axis=1)
df = df.dropna()
df = df.drop_duplicates()
df = df[df.pitchMean != 0]

corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(data=corr, annot=True, center=0, cmap=cmap, square=True, mask=mask)
plt.title("Diagonal Correlation HeatMap")
plt.show()

# f, ax = plt.subplots(figsize=(12, 12))
# corr = df.select_dtypes().corr()

# # TO display diagonal matrix instead of full matrix.
# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True

# # Generate a custom diverging colormap.
# cmap = sns.diverging_palette(220, 10, as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio.
# g = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0, annot=True, fmt='.2f',\
# square=True, linewidths=.5, cbar_kws={"shrink": .5})

# # plt.subplots_adjust(top=0.99)
