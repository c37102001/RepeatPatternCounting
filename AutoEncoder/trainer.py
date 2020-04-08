import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from utils import same_seeds
from torch.optim.lr_scheduler import StepLR
import math
import os
import json
from tqdm import tqdm
from ipdb import set_trace as pdb

class Trainer:
    def __init__(self, arch, model, lr, batch_size, wd):
        self.arch = arch
        if not os.path.exists(f'{arch}/ckpt/'):
            os.makedirs(f'{arch}/ckpt/')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.batch_size = batch_size
        self.model = model
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.opt, step_size=50, gamma=0.1)
        self.history = {'train_loss':[], 'valid_loss':[]}
        self.min_loss = math.inf
        same_seeds(73)

    def run_epoch(self, epoch, img_dataset, training, desc):
        self.model.train(training)
        shuffle = training
        img_dataloader = DataLoader(img_dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=2)

        trange = tqdm(img_dataloader, total=len(img_dataloader), desc=desc)
        total_loss = 0
        for i, (img, _) in enumerate(trange):
            img = img.to(self.device)

            encoded, o_img = self.model(img)
            loss = self.criterion(o_img, img)
            
            if training:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            total_loss += loss.item()
            trange.set_postfix(loss=total_loss/(i+1))

        if training:
            self.history['train_loss'].append(total_loss/len(trange))
            # self.scheduler.step()
        elif desc == '[Pattern Valid]':
            self.history['valid_loss'].append(total_loss/len(trange))
            if loss < self.min_loss:
                torch.save(self.model.state_dict(), f'{self.arch}/ckpt/model.ckpt')
        
        self.save_hist()

    def save_hist(self):
        with open(f'{self.arch}/history.json', 'w') as f:
            json.dump(self.history, f, indent=4)