import os
import csv
import glob
import numpy as np
import torch
import pickle


path = "research/data/raw_data/2013_calc_scenario/*.csv"
output_dir = 'research/data/prep_data'
output_file = os.path.join(output_dir, '2013_calc_scenario.pkl')

files = sorted(glob.glob(path))

max_data = np.genfromtxt("research/data/raw_data/2013_calc_scenario/TF111_h01-add-ase_nm-vr2_fm1_sd1.csv", delimiter=",")
lat = np.unique(max_data[:,0])[::-1]
lon = np.unique(max_data[:,1])

data_4d = np.zeros((len(files), max_data.shape[1]-3, len(lat), len(lon)), dtype=np.float32)
filenames = []

for file_index, file in enumerate(files):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = [row + [0.]*(max_data.shape[1]-len(row)) for row in reader]
    data = np.array(data, dtype=float)

    data[:,6] = data[:,6] * 10**data[:,7]
    data = np.delete(data, 7, 1)

    for row in range(data.shape[0]):
        lat_index = np.where(lat == data[row,0])[0][0]
        lon_index = np.where(lon == data[row,1])[0][0]
        data_4d[file_index, :data.shape[1]-2, lat_index, lon_index] = data[row, 2:]

    print(file_index, file)

    filename = os.path.basename(file).split('.')[0]
    filenames.append(filename)

channels_to_remove = [6,7,8,9,10,11]
data_4d = np.delete(data_4d, channels_to_remove, 1)

data_4d = np.pad(data_4d, ((0, 0), (0, 0), (0, 12), (0, 13)), mode='constant')
# data_4d = np.pad(data_4d, ((0, 0), (0, 0), (721, 187), (481, 556)), mode='constant')

data_min = np.min(data_4d, axis=(0, 2, 3), keepdims=True)
data_max = np.max(data_4d, axis=(0, 2, 3), keepdims=True)
data_4d = (data_4d - data_min) / (data_max - data_min)

data_dict = {"images": data_4d, "labels": filenames}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_file, 'wb') as f:
    pickle.dump(data_dict, f)
