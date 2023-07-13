'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2021-05-09 14:05:16
'''
#%%
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from datasets.DataOCTMulti import *
from model.classifer_base import classifer_base, classifer_base2
from model.network1 import *
from model.network2 import *
from utils import utils
from utils.logger import Logger
from utils.Metrics import *
import pandas as pd
from sklearn.metrics import *

os.environ['CUDA_VISIBLE_DEVICES'] = utils.SearchFreeGPU(10, 0.5)
torch.cuda.empty_cache()

utils.log('start inference')

def AUC_ROC(y_true, y_pred):
    try: 
        roc = roc_auc_score(y_true, y_pred)
    except ValueError: 
        pass 
    return roc

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='T0612-2149-densenet121-ALL-b8-320-bv-data', type=str, help="get center type")
parser.add_argument('--img_size', default=320, type=int, help="image size")
parser.add_argument('--mode', default='test', type=str, help="get center type")
parser.add_argument('--network', default='resnet18-MMESA-b4-ASL', type=str, help="get center type")
parser.add_argument('--label_names', default=['MH','SP','ED','MEM','PVD','ILM','PD','IO','RNL','RPE','CA'], nargs='+', help="output class names")
parser.add_argument('--plot_save_dir', default='../plots', type=str, help="plot_save_dir")

Flags, _ = parser.parse_known_args()
# load model
pypath = os.path.abspath(__file__)
path, _ = os.path.split(pypath)
weightname = os.path.join(path, '..', Flags.model, 'Model', 'Model_Ep60_AUC_0.9252.pkl')
model = torch.load(weightname)
model.eval()
metrics = OCTMultiMetrics()

# load data
data_dir = os.path.join('..', 'data_OCT_BV')
dataset_val = DatasetOCTMultiCLA(data_dir, Flags, mode=Flags.mode)
dataloader_val = DataLoader(dataset_val, batch_size=4)

# save config
_, name = os.path.split(weightname)
name, _ = os.path.splitext(name)
save_dir = os.path.join(path, '..', 'ModelResults', Flags.network + name)
print(save_dir)
utils.checkpath(save_dir)

pred_dir = os.path.join(path, '..', Flags.model, 'Result', Flags.model+'MultiCla.csv')
with open(pred_dir, "a") as f:
    w = csv.writer(f)
    w.writerow(['ID']+['1']+['2']+['3']+['4']+['5']+['6']+['7']+['8']+['9']+['10']+['11'])
#%% test
pbar = tqdm(dataloader_val, ncols=60)
all_true = []
all_pred = []
to_tensor = transforms.ToTensor()
with torch.no_grad():
    for n_step, batch_data in enumerate(pbar):
        # get data
        x_batch1 = batch_data[0]
        x_batch2 = batch_data[1]
        y_batch = batch_data[2]
        data_name = batch_data[3]
        if torch.cuda.is_available():
            x_data1 = x_batch1.cuda()
            x_data2 = x_batch2.cuda()
            y_true = y_batch.cuda()

        y_pred = model(x_data1, x_data2)["main_out"]
        y_pred = torch.sigmoid(y_pred)

        y_true = y_true.cpu().data.numpy()
        y_pred = y_pred.cpu().data.numpy()
        all_true.append(y_true)
        all_pred.append(y_pred)

        for batchidx in range(x_data1.shape[0]):
            name_, _ = os.path.splitext(data_name[batchidx])

            with open(pred_dir, "a") as f:
                w = csv.writer(f)
                rows1 = [name_]
                rows2 = [name_]                
                for classidx in range(0,11):
                    # rows1 = rows1+[float(y_true[batchidx, classidx])]
                    rows2 = rows2+[float(y_pred[batchidx, classidx])]
                # w.writerow(rows1)
                w.writerow(rows2)
                # w.writerow([name_]+[float(y_true[batchidx,1])]+[float(y_pred[batchidx])])

    # metric
    all_true, all_pred = np.vstack(all_true), np.vstack(all_pred)
    acc, pre, sen, spe, f1 = metrics.ClassifyMetric(all_true, all_pred, 'middle')
    roc = metrics.AUC2(all_true, all_pred)
    mean_roc, class_roc = get_roc_score_mutliply(all_true, all_pred)
    mAP, class_AP = get_map_score_mutliply(all_true, all_pred)
    # with open(pred_dir, "a") as f:
    #     w = csv.writer(f)
    #     w.writerow(['mean AUC']+[float(roc)])

print(f"mean AUC class = {mean_roc}")
print(f"mean mAP class = {mAP}")
print(f"mean AUC = {roc}")
print(f"AUC of each class = {class_roc}")
print(f"AP of each class = {class_AP}")
# print(f"mean Acc = {acc}")
# print(f"mean Precision = {pre}")
# print(f"mean Sensitivity = {sen}")
# print(f"mean Specificity = {spe}")
# print(f"mean F1_score = {f1}")

# %%
