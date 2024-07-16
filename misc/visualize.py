import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import argparse


def save_images(data_path, save_dir, mask_path=None, apply_mask=0, show_colorbar=1):
    os.makedirs(save_dir, exist_ok=True)
    data_4d = torch.load(data_path).numpy()

    mask = None
    if apply_mask == 0 and mask_path is not None:
        mask = torch.from_numpy(np.array(Image.open(mask_path))).float() / 255.0

    for i in range(data_4d.shape[0]):
        data = data_4d[i, 0]

        if apply_mask == 0 and mask is not None:
            masked_data = np.where(mask == 1, data, np.nan)
            cmap = plt.get_cmap('Reds')
            cmap.set_bad(color='gray')
            plt.imshow(masked_data, cmap=cmap, interpolation='nearest', vmin=0, vmax=1.0)
        else:
            plt.imshow(data, cmap='Reds', interpolation='nearest', vmin=0, vmax=1.0)

        if show_colorbar != 0:
            cbar = plt.colorbar()
            cbar.set_label('velocity response spectra [m/s]', rotation=90, labelpad=20, fontsize=14)
            cbar.ax.yaxis.set_label_position('right')
            cbar.ax.invert_yaxis()
            cbar.ax.tick_params(labelsize=14)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        filename = f'{i+1:03}.pdf'
        
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
        plt.close()

def save_scatter_plots(true_data_path, pred_data_path, mask_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    true_data = torch.load(true_data_path).numpy()
    pred_data = torch.load(pred_data_path).numpy()
    mask = torch.from_numpy(np.array(Image.open(mask_path))).float() / 255.0

    for i in range(true_data.shape[0]):
        true_batch = true_data[i, 0]
        pred_batch = pred_data[i, 0]

        true_values = true_batch[mask == 1]
        pred_values = pred_batch[mask == 1]

        plt.figure(figsize=(6, 6))
        plt.scatter(true_values, pred_values, s=1)
        plt.xlabel('true', fontsize=14)
        plt.ylabel('predicted', fontsize=14)
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        plt.xticks(np.arange(0, 1.2, 0.2), fontsize=14)
        plt.yticks(np.arange(0, 1.2, 0.2), fontsize=14)

        filename = f'scatter_{i+1:03}.png'

        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('apply_mask', type=int, nargs='?', default=0, help='Apply mask or not')
    parser.add_argument('show_colorbar', type=int, nargs='?', default=0, help='Show colorbar or not')
    args = parser.parse_args()

    save_dir = 'research/data/images'
    mask_path = 'research/data/sea.png'
    true_data_path = 'research/data/cv_data/crossValidation_all0/y_test.pt'
    pred_data_path = 'research/data/cv_data/crossValidation_all0/y_pred.pt'

    save_images(pred_data_path, save_dir, mask_path, args.apply_mask, args.show_colorbar)
    save_scatter_plots(true_data_path, pred_data_path, mask_path, save_dir)

if __name__ == '__main__':
    main()
