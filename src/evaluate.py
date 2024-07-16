import os
import torch
import numpy as np
from PIL import Image
from ssimloss import SSIMLoss


class Evaluator:
    def __init__(self, mask):
        self.mask = mask

    def evaluate(self, y_pred, y_test):
        extended_mask = self.mask.expand_as(y_test)
        y_test_unmasked = y_test[extended_mask == 1]
        y_pred_unmasked = y_pred[extended_mask == 1]

        mse_loss = torch.nn.functional.mse_loss(y_test_unmasked, y_pred_unmasked)

        pixel_range = 1.0
        psnr = 20 * torch.log10(pixel_range / torch.sqrt(mse_loss))
        psnr = psnr.cpu().detach().numpy()

        ssim_loss = SSIMLoss(mask=self.mask)
        ssim = -ssim_loss(y_test, y_pred)
        ssim = ssim.cpu().detach().numpy()

        return psnr, ssim

def load_data(pred_paths, truth_paths):
    all_y_pred = []
    all_y_test = []

    for pred_path, truth_path in zip(pred_paths, truth_paths):
        y_pred = torch.load(pred_path).float()
        y_test = torch.load(truth_path).float()
        all_y_pred.append(y_pred)
        all_y_test.append(y_test)
        
    return torch.cat(all_y_pred, dim=0), torch.cat(all_y_test, dim=0)

def main():
    mask_path = 'research/data/sea.png'
    base_truth_dir = 'research/data/cv_data/crossValidation_all{}/y_test.pt'
    base_pred_dir = 'research/data/cv_data/crossValidation_all{}/y_pred.pt'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mask = torch.from_numpy(np.array(Image.open(mask_path))).float().to(device) / 255.0

    pred_paths = [base_pred_dir.format(i) for i in range(10)]
    truth_paths = [base_truth_dir.format(i) for i in range(10)]

    y_pred, y_test = load_data(pred_paths, truth_paths)
    y_pred = y_pred.to(device)
    y_test = y_test.to(device)

    evaluator = Evaluator(mask)
    psnr, ssim = evaluator.evaluate(y_pred, y_test)

    print(f'Combined PSNR: {psnr}')
    print(f'Combined SSIM: {ssim}')

if __name__ == '__main__':
    main()
