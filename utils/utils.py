import argparse
import datetime
import os
import random
import sys
import time
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance


def SearchFreeGPU(interval=60, threshold=0.5):
    while True:
        qargs = ['index', 'memory.free', 'memory.total']
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
        results = os.popen(cmd).readlines()
        gpus = pd.DataFrame(np.zeros([len(results), 3]), columns=qargs)
        for i, line in enumerate(results):
            info = line.strip().split(',')
            gpus.loc[i, 'index'] = info[0]
            gpus.loc[i, 'memory.free'] = float(info[1].upper().strip().replace('MIB', ''))
            gpus.loc[i, 'memory.total'] = float(info[2].upper().strip().replace('MIB', ''))
            gpus.loc[i, 'Freerate'] = gpus.loc[i, 'memory.free'] / gpus.loc[i, 'memory.total']

        maxrate = gpus.loc[:, "Freerate"].max()
        index = gpus.loc[:, "Freerate"].idxmax()
        if maxrate > threshold:
            print('GPU index is: {}'.format(index))
            return str(index)
        else:
            print('Searching Free GPU...')
            time.sleep(interval)


def adjust_learning_rate(optimizer, epoch, config):
    lr = config.learning_rate * (1 - epoch / config.epochs)**0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def dic2txt(filename, dic):
    fp = open(filename, 'w')
    for key in dic:
        fp.writelines(key + ':\t' + str(dic[key]) + '\n')
    fp.close()
    return True


def MergeLog(logs, temp, n_step):
    for key in temp:
        if np.isnan(temp[key]):
            temp[key] = 0

        if key in logs:
            logs[key] = (logs[key] * n_step + temp[key]) / (n_step + 1)
        else:
            logs[key] = (0.0 * n_step + temp[key]) / (n_step + 1)

    return logs


def ShowFlags(FLAGS):
    log('Argparse Settings')
    for i in vars(FLAGS):
        if len(i) < 8:
            print(i + '\t\t------------  ' + str(vars(FLAGS)[i]))
        else:
            print(i + '\t------------  ' + str(vars(FLAGS)[i]))
    print()

    return FLAGS


def checkpath(path):
    try:
        os.makedirs(path)
        # print('creat ' + path)
    except OSError:
        pass


def log(text):
    """
    log status with time label
    """
    print()
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line1 = '=' * 10 + '  ' + nowTime + '  ' + '=' * 10
    length = len(line1)
    leftnum = int((length - 4 - len(text)) / 2)
    rightnum = length - 4 - len(text) - leftnum
    line2 = '*' * leftnum + ' ' * 2 + text + ' ' * 2 + '*' * rightnum
    print(line1)
    print(line2)
    print('=' * len(line1))


def array2image(arr):
    if (np.max(arr) <= 1):
        image = Image.fromarray((arr * 255).astype(np.uint8))
    else:
        image = Image.fromarray((arr).astype(np.uint8))

    return image


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames

def AUC(pred, label, roc_image=None):
    assert len(pred.shape) == 1 and len(label.shape) == 1
    num_data = pred.shape[0]
    thresholds = [i / 100.0 for i in reversed(range(0, 101))]  # 阈值
    tpr = []
    fpr = []
    for th in thresholds:
        p = pred > th   # true or flase
        true_positive = torch.nonzero((p == label)*label, as_tuple=False).size()[0]
        flase_positive = torch.nonzero(p, as_tuple=False).size()[0] - true_positive

        all_positive = torch.nonzero(label, as_tuple=False).size()[0]  # 有病样本的数量
        all_negative = num_data - all_positive  # 正常样本的数量

        tpr_ = true_positive / all_positive  # 真正类率， sensitivity, recall
        fpr_ = flase_positive / all_negative  # 假正类率
        tpr.append(tpr_)
        fpr.append(fpr_)
    # 计算AUC
    auc = 0
    for i in range(len(tpr)-1):
        auc += (fpr[i+1] - fpr[i]) * (tpr[i+1] + tpr[i]) / 2
    # 保存ROC曲线图
    if roc_image is not None:
        plt.cla()
        plt.title(f"AUC={auc:.5f}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.plot(np.array(fpr), np.array(tpr))
        plt.savefig(roc_image)

    return auc



def multi_label_AUC(pred, label, class_names, img_save_dir=None, num_fold=None):
    assert len(pred.size()) == 2 and len(label.size()) == 1
    auc = dict()
    m_auc = []
    for i in range(pred.shape[1]):
        pred_i = pred[:, i]
        label_i = (label == i).float()
        save_dir = None if img_save_dir is None else os.path.join(img_save_dir, class_names[i]+str(num_fold)+".png")
        auc_i = AUC(pred_i, label_i, save_dir)
        auc.update({class_names[i]: auc_i})
        m_auc.append(auc_i)
    m_auc = np.array(m_auc).mean()
    auc.update({"mean": m_auc})
    return auc

# 保存打印指标
def save_print_score(auc, file, label_names):
    result_AUC = [auc['mean']] + [auc[key] for key in label_names]
    title = ['mAUC'] + [name + "_AUC" for name in label_names  ]
    with open(file, "a") as f:
        w = csv.writer(f)
        w.writerow(title)
        w.writerow(result_AUC)

    print("\n##############Test Result##############")
    print(f"mean AUC: {auc['mean']}")
    for i in range(len(label_names)):
        print(f"{label_names[i]}: {auc[label_names[i]]}")