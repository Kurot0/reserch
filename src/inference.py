import torch
import importlib
import numpy as np
import time
from PIL import Image
from evaluate import Evaluator


class Inferencer:
    def __init__(self, params):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        data_path = params['data_path']
        underground_data_path = params['underground_data_path']

        self.x_test = torch.load(data_path + '/x_test.pt').float().to(self.device)
        self.y_test = torch.load(data_path + '/y_test.pt').float().to(self.device)
        self.y_pred_path = data_path + '/y_pred.pt'

        underground_channels = params['underground_channels']
        self.underground_data = torch.load(underground_data_path).float().to(self.device)[:, underground_channels, :, :]
        min_val = torch.min(self.underground_data)
        max_val = torch.max(self.underground_data)
        self.underground_data = (self.underground_data - min_val) / (max_val - min_val)

        self.batch_size = params['batch_size']
        test_dataset = torch.utils.data.TensorDataset(self.x_test)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        model_module = importlib.import_module(params['model_module'])
        model_class = getattr(model_module, params['model_class'])

        self.model = model_class(**params).to(self.device)

        self.psnr = None
        self.ssim = None
        self.time = 0

        self.mask = torch.from_numpy(np.array(Image.open(params['mask_path']))).float().to(self.device) / 255.0
        self.evaluator = Evaluator(self.mask)

    def inference(self):
        self.model.eval()
        all_outputs = []
        start_time = time.time()
        with torch.no_grad():
            for inputs in self.test_loader:
                inputs = inputs[0].to(self.device)
                underground_data = self.underground_data.to(self.device)
                outputs = self.model(inputs, underground_data)
                all_outputs.append(outputs)
        end_time = time.time()
        self.time = end_time - start_time

        y_pred = torch.cat(all_outputs, dim=0)
        torch.save(y_pred.cpu(), self.y_pred_path)

        self.psnr, self.ssim = self.evaluator.evaluate(y_pred, self.y_test)
