'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2021-06-27 21:19:51
'''
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from datasets.DataOCTMulti import *
from model.Losses import *
from model.classifer_base import classifer_base, classifer_base2
from model.network1 import *
from model.network2 import *
from model.network3 import *
from utils import utils
from utils.logger import Logger
from utils.Metrics import *

#%% Settings
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_rate', default=0.5, type=float, help="gpu free rate")
parser.add_argument('--gpu_gap', default=10, type=int, help="check free gpu interval")
parser.add_argument('--log_cols', default=140, type=int, help="num of columns for log")

parser.add_argument('--epochs', default=50, type=int, help="nums of epoch")
parser.add_argument('--inchannels', default=3, type=int, help="nums of input channels")
parser.add_argument('--img_size', default=320, type=int, help="image size")
parser.add_argument('--n_class', default=11, type=int, help="output classes")
parser.add_argument('--label_names', default=['1','2','3','4','5','6','7','8','9','10','11'], nargs='+', help="output class names")
parser.add_argument('--plot_save_dir', default='../plots', type=str, help="plot_save_dir")
parser.add_argument('--datatrain', default='train', type=str, help="select all data or part data train")
parser.add_argument('--batch_size', default=8, type=int, help="batch size")
parser.add_argument('--workers', default=4, type=int, help="num of workers")

parser.add_argument('--model', default='resnet50', type=str, help="training model")
parser.add_argument('--en_pretrained', default=True, type=bool, help="whether load pretrained model")
parser.add_argument('--learning_rate', default=2e-4, type=float, help="initial learning rate for Adam")
parser.add_argument('--alpha', default=2, type=float, help="focal loss weigth")
parser.add_argument('--gamma_n', default=2, type=float, help="ASL loss weigth")
parser.add_argument('--ASL_clip', default=0.05, type=float, help="ASL loss clip")
parser.add_argument('--val_step', default=5, type=int, help="the frequency of evaluation")

parser.add_argument('--savename', default='Result', type=str, help="output folder name")

Flags, _ = parser.parse_known_args()
utils.ShowFlags(Flags)
os.environ['CUDA_VISIBLE_DEVICES'] = utils.SearchFreeGPU(Flags.gpu_gap, Flags.gpu_rate)
torch.cuda.empty_cache()

#==============================================================================
# Dataset
#==============================================================================
data_dir = os.path.join('..', 'data_OCT_BV')
dataset_train = DatasetOCTMultiCLA(data_dir, Flags, mode=Flags.datatrain)
dataloader_train = DataLoader(dataset_train, batch_size=Flags.batch_size, num_workers=Flags.workers, shuffle=True)

dataset_val = DatasetOCTMultiCLA(data_dir, Flags, mode=Flags.datatrain.replace('train', 'val'))
dataloader_val = DataLoader(dataset_val, batch_size=Flags.batch_size, num_workers=Flags.workers, shuffle=True)

utils.log('Load Data successfully')

#==============================================================================
# Logger
#==============================================================================
logger = Logger(Flags)
utils.log('Setup Logger successfully')

#==============================================================================
# load model, optimizer, Losses
#==============================================================================
# model = classifer_base(Flags.model, Flags.en_pretrained, Flags.n_class, Flags.inchannels)
model = classifer_base2(Flags)
model = model.cuda() if torch.cuda.is_available() else model
print('load model {}'.format(Flags.model))
# summary(model, [(3, 320, 320), (3, 320, 320)])
optimizer = torch.optim.Adam(model.parameters(), lr=Flags.learning_rate)
criterion = {
    'BCE': nn.BCELoss(),
    'FOCAL': FocalLoss(gamma=2, alpha=[Flags.alpha, 1]),
    'ASL': AsymmetricLossOptimized(gamma_neg=Flags.gamma_n, gamma_pos=1, clip=Flags.ASL_clip),
}
metrics = OCTMultiMetrics()
utils.log('Build Model successfully')

#==============================================================================
# Train model
#==============================================================================
for epoch in range(Flags.epochs + 1):
    ############################################################
    # Train Period
    ############################################################
    model.train()
    pbar = tqdm(dataloader_train, ncols=Flags.log_cols)
    pbar.set_description('Epoch {:2d}'.format(epoch))

    log_Train, temp = {}, {}
    all_true, all_pred = [], []
    for n_step, batch_data in enumerate(pbar):
        # get data
        x_batch1 = batch_data[0]
        x_batch2 = batch_data[1]
        y_batch = batch_data[2]
        if torch.cuda.is_available():
            x_data1 = x_batch1.cuda()
            x_data2 = x_batch2.cuda()
            y_true = y_batch.cuda()
        optimizer.zero_grad()

        # forward
        y_pred = model(x_data1, x_data2)["main_out"]
        # y_pred = torch.softmax(y_pred, dim=1)
        y_pred = torch.sigmoid(y_pred)

        # backward
        loss_bce = criterion['BCE'](y_pred, y_true)
        loss = loss_bce
        loss.backward()
        optimizer.step()

        # metric
        y_true = y_true.cpu().data.numpy()
        y_pred = y_pred.cpu().data.numpy()
        all_true.append(y_true)
        all_pred.append(y_pred)

        # log
        temp['Loss'] = loss.item()
        log_Train = utils.MergeLog(log_Train, temp, n_step)
        pbar.set_postfix(log_Train)
    all_true, all_pred = np.vstack(all_true), np.vstack(all_pred)
    acc, auc = metrics.ClassifyScore(all_true, all_pred)
    log_Train['Acc'], log_Train['AUC'] = acc, auc
    pbar.set_postfix(log_Train)
    logger.write_tensorboard('1.Train', log_Train, epoch)

    if not (epoch % Flags.val_step == 0):
        continue

    ############################################################
    # Test Period
    ############################################################
    print('*' * Flags.log_cols)
    with torch.no_grad():
        model.eval()
        pbar = tqdm(dataloader_val, ncols=Flags.log_cols)
        pbar.set_description('Val')
        log_Test, temp = {}, {}
        all_true, all_pred = [], []
        for n_step, batch_data in enumerate(pbar):
            # get data
            x_batch1 = batch_data[0]
            x_batch2 = batch_data[1]
            y_batch = batch_data[2]
            if torch.cuda.is_available():
                x_data1 = x_batch1.cuda()
                x_data2 = x_batch2.cuda()
                y_true = y_batch.cuda()

            # forward
            y_pred = model(x_data1, x_data2)["main_out"]
            y_pred = torch.sigmoid(y_pred)

            loss_bce = criterion['BCE'](y_pred, y_true)
            loss = loss_bce
            temp['Loss'] = loss.item()

            # metric
            y_true = y_true.cpu().data.numpy()
            y_pred = y_pred.cpu().data.numpy()
            all_true.append(y_true)
            all_pred.append(y_pred)

            log_Test = utils.MergeLog(log_Test, temp, n_step)
            pbar.set_postfix(log_Test)
        # log
        all_true, all_pred = np.vstack(all_true), np.vstack(all_pred)
        acc, auc = metrics.ClassifyScore(all_true, all_pred)
        log_Test['Loss'] = loss.item()
        log_Test['Acc'] = acc
        log_Test['AUC'] = auc
        print(log_Test)
        logger.write_tensorboard('2.Val', log_Test, epoch)
        logger.save_model(model, 'Ep{}_AUC_{:.4f}'.format(epoch, auc))
    print('*' * Flags.log_cols)
