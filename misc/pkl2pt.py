import os
import pickle
import torch


x_data_path = 'research/data/prep_data/2013_calc_scenario.pkl'
y_data_path = 'research/data/prep_data/2013_Sv05s_LL.pkl'
labels_dict_path = 'research/data/labels_dictionary.pkl'
cv_data_dir = 'research/data/cv_data'

with open(x_data_path, 'rb') as f:
    x_data = pickle.load(f)
with open(y_data_path, 'rb') as f:
    y_data = pickle.load(f)
with open(labels_dict_path, 'rb') as f:
    labels_dict = pickle.load(f)

x_images = x_data['images']
x_labels = x_data['labels']

y_images = y_data['images']
y_labels = y_data['labels']

x_labels = [label.replace("sd1", "Sv5") for label in x_labels]
y_labels = [label.replace("sd1", "Sv5") for label in y_labels]

for i in range(10):
    dir_name = os.path.join(cv_data_dir, f'crossValidation_all{i}')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

for i in range(10):
    dir_name = os.path.join(cv_data_dir, f'crossValidation_all{i}')
    for phase in ['train', 'valid', 'test']:
        phase_labels = labels_dict[f'quakeData-all-crossVaridation{i}'][phase]  
        
        x_indexes = [index for index, label in enumerate(x_labels) if label in phase_labels]
        y_indexes = [index for index, label in enumerate(y_labels) if label in phase_labels]
        
        x_phase_images = torch.tensor(x_images[x_indexes])
        y_phase_images = torch.tensor(y_images[y_indexes])
        
        torch.save(x_phase_images, os.path.join(dir_name, 'x_' + phase + '.pt'), pickle_protocol=4)
        torch.save(y_phase_images, os.path.join(dir_name, 'y_' + phase + '.pt'), pickle_protocol=4)
