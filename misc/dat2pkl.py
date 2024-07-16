import pandas as pd
import glob
import numpy as np
import torch
import os
import pickle


path = 'research/data/raw_data/2013_Sv05s_LL/*.dat'
output_dir = 'research/data/prep_data'
output_file = os.path.join(output_dir, '2013_Sv05s_LL.pkl')

files = sorted(glob.glob(path))

df = pd.read_csv(files[0], names=['lon', 'lat', 'amp', 'sid'])
lons = np.unique(df['lon'].values)
lats = np.unique(df['lat'].values)[::-1]

data_4d = np.zeros((len(files), 1, len(lats), len(lons)))

filenames = []

for file_index, file in enumerate(files):
    df = pd.read_csv(file, names=['lon', 'lat', 'amp', 'sid'])

    for ind in range(len(df)):
        lonInd = int(np.where(df['lon'][ind] == lons)[0].item())
        latInd = int(np.where(df['lat'][ind] == lats)[0].item())
        data_4d[file_index, 0, latInd, lonInd] = df['amp'][ind]

    print(file_index, file)

    filename = os.path.basename(file).split('.')[0]
    filenames.append(filename)

data_4d[data_4d > 8] = 8

data_4d = np.pad(data_4d, ((0, 0), (0, 0), (0, 0), (2, 2)), mode='constant')

data_4d = np.delete(data_4d, np.s_[:3], axis=2)
data_4d = np.delete(data_4d, np.s_[-3:], axis=2)

data_min = np.min(data_4d, axis=(0, 2, 3), keepdims=True)
data_max = np.max(data_4d, axis=(0, 2, 3), keepdims=True)
data_4d = (data_4d - data_min) / (data_max - data_min)

data_dict = {'images': data_4d, 'labels': filenames}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_file, 'wb') as f:
    pickle.dump(data_dict, f)
