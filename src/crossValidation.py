import os
import time
import datetime
import shutil
import yaml
import random
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from train import Trainer
from inference import Inferencer
from evaluate import Evaluator

def torch_fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def save_loss_graph(trainer, file_path):
    plt.figure()
    plt.plot([abs(loss) for loss in trainer.losses], label='Training Loss')
    plt.plot([abs(val_loss) for val_loss in trainer.val_losses], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.xlim(0, 300)
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(file_path)
    plt.close()

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='research/config.yaml', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        params = yaml.safe_load(stream)

    seed = params['seed']  
    torch_fix_seed(seed)

    base_dir = "research/data/result"
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    experiment_dir = os.path.join(base_dir, f"exp_{current_time}")
    os.makedirs(experiment_dir, exist_ok=True)

    loss_graph_dir = os.path.join(experiment_dir, 'loss_graph')
    pred_data_dir = os.path.join(experiment_dir, 'pred_data')
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoint')
    os.makedirs(loss_graph_dir, exist_ok=True)
    os.makedirs(pred_data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(experiment_dir, 'config.yaml'))

    pred_paths = []
    truth_paths = []
    training_times = []
    inference_times = []
    best_epochs = []
    psnr_results = []
    ssim_results = []

    for i in range(10):
        dir_name = 'crossValidation_all{}'.format(i)
        print(f"Starting cross validation {i}")
        
        params['data_path'] = os.path.join('research/data/cv_data', dir_name)
        
        trainer = Trainer(params)
        trainer.train()
        training_times.append(trainer.time)

        inferencer = Inferencer(params)
        inferencer.model.load_state_dict(trainer.best_model)
        inferencer.model.eval()
        inferencer.inference()
        inference_times.append(inferencer.time)
        best_epochs.append(trainer.best_epoch)

        psnr_results.append(inferencer.psnr)
        ssim_results.append(inferencer.ssim)

        loss_graph_path = os.path.join(loss_graph_dir, f'cv{i}_loss.png')
        save_loss_graph(trainer, loss_graph_path)
        
        pred_file = f'cv{i}_pred.pt'
        pred_paths.append(os.path.join(pred_data_dir, pred_file))
        shutil.copy(os.path.join(params['data_path'], 'y_pred.pt'), os.path.join(pred_data_dir, pred_file))

        truth_file = f'cv{i}_truth.pt'
        truth_paths.append(os.path.join(params['data_path'], 'y_test.pt'))
        shutil.copy(os.path.join(params['data_path'], 'y_test.pt'), os.path.join(pred_data_dir, truth_file))

        model_path = os.path.join(checkpoint_dir, f'cv{i}_model.pth')
        torch.save(trainer.best_model, model_path)

        del trainer
        del inferencer
        torch.cuda.empty_cache()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mask_path = 'research/data/sea.png'
    mask = torch.from_numpy(np.array(Image.open(mask_path))).float().to(device) / 255.0

    y_pred, y_test = load_data(pred_paths, truth_paths)
    y_pred = y_pred.to(device)
    y_test = y_test.to(device)

    evaluator = Evaluator(mask)
    psnr, ssim = evaluator.evaluate(y_pred, y_test)

    formatted_results = ", ".join(f"({psnr_result:.6f}, {ssim_result:.6f})" for psnr_result, ssim_result in zip(psnr_results, ssim_results))
    print(f"Individual Results: {formatted_results}")
    print(f'Combined PSNR: {psnr}')
    print(f'Combined SSIM: {ssim}')

    with open(os.path.join(experiment_dir, 'result.txt'), 'w') as f:
        f.write(f"Individual Results: {formatted_results}\n")
        f.write(f'Combined PSNR: {psnr}\n')
        f.write(f'Combined SSIM: {ssim}\n')
        f.write('\n')
        f.write("Individual Results: " + ",".join(map(str, best_epochs)) + "\n")
        f.write(f"Average Best Epoch: {np.mean(best_epochs)}\n")
        f.write('\n')
        f.write(f"Total Training time: {sum(training_times):.2f} seconds\n")
        f.write(f"Total Inference time: {sum(inference_times):.4f} seconds\n")
        f.write(f"Avarage Inference time: {sum(inference_times)/0.36:.4f} seconds\n")

if __name__ == '__main__':
    main()
