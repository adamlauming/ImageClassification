'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2021-01-13 13:34:04
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import *
from skimage import filters
import numpy as np


class OCTMultiMetrics(object):
    def __init__(self):
        pass

    def SegmentationMetrics(self, all_true, all_pred, mode):
        """
        calculate dice 
        y_true: groud truth in torch (-1, 1, H, W)
        y_pred: predict result in torch
        mask: mask for each image (-1, 1, H, W)
        mode: "otsu", calculate dice with otsu threshold
              "f1", calculate dice with f1 threshold
              "None", soft dice
        """
        N, C = all_true.shape[0], all_true.shape[1]
        dice = np.ones([N, C])
        # iou = np.zeros([N, 1])
        for i in range(N):
            y_true = np.squeeze(all_true[i, ...])
            y_pred = np.squeeze(all_pred[i, ...])
            for k in range(C):
                y_true_ = y_true[k, ...].flatten()
                y_pred_ = y_pred[k, ...].flatten()
                if np.sum(y_true_) > 1.0:
                    dice[i, k] = Dice_coefficient(y_true_, y_pred_, mode)

        return np.mean(dice, axis=0)

    def ClassifyMetric(self, all_true, all_pred, mode):
        """
        calculate acc, precision, sensitivity, specificity
        y_true: groud truth in torch (-1, 1, H, W)
        y_pred: predict result in torch
        mask: mask for each image (-1, 1, H, W)
        """
        N, C = all_true.shape[0], all_true.shape[1]
        acc = np.zeros([N, C])
        precsion = np.zeros([N, C])
        sensitivity = np.zeros([N, C])
        specificity = np.zeros([N, C])
        f1_score = np.zeros([N, C])
        for i in range(N):
            y_true_all = np.squeeze(all_true[i, ...])
            y_pred_all = np.squeeze(all_pred[i, ...])
            for k in range(C):
                y_true = y_true_all[k, ...].flatten()
                y_pred = y_pred_all[k, ...].flatten()

                if mode == 'otsu':  # convert to binary map
                    y_pred = threshold_by_otsu(y_pred)
                elif mode == 'middle':
                    y_pred = threshold_by_middle(y_pred)

                acc[i, k], precsion[i, k], sensitivity[i, k], specificity[i, k], f1_score[i, k] = misc_measures(y_true, y_pred)

        return acc.mean(axis=0), precsion.mean(axis=0), sensitivity.mean(axis=0), specificity.mean(axis=0), f1_score.mean(axis=0)

    # def AUC(self, y_true, y_pred):
    #     y_true = y_true.flatten()
    #     y_pred = y_pred.flatten()

    #     pr = AUC_PR(y_true, y_pred)
    #     roc = AUC_ROC(y_true, y_pred)

    #     return pr, roc

    def ClassifyScore(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        acc = accuracy_score(np.round(y_true), np.round(y_pred))
        auc = AUC_ROC(y_true, y_pred)

        return acc, auc

    def AUC2(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        roc = AUC_ROC(y_true, y_pred)

        return roc


def Jaccard_coefficient(y_true, y_pred, mode='middle'):
    """
    mode: otsu, f1, middle, analoy
    """
    if mode == 'otsu':  # convert to binary map
        y_pred = threshold_by_otsu(y_pred)
    elif mode == 'f1':
        y_pred = threshold_by_f1(y_true, y_pred)
    elif mode == 'middle':
        y_pred = threshold_by_middle(y_pred)

    # hard and soft same here
    overlap = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - overlap
    try:
        iou = overlap / union
    except ZeroDivisionError:
        iou = 0.0

    return iou


def Dice_coefficient(y_true, y_pred, mode='middle'):
    """
    mode: otsu, f1, middle, analoy
    """
    if mode == 'otsu':  # convert to binary map
        y_pred = threshold_by_otsu(y_pred)
    elif mode == 'f1':
        y_pred = threshold_by_f1(y_true, y_pred)
    elif mode == 'middle':
        y_pred = threshold_by_middle(y_pred)

    # hard and soft same here
    overlap = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - overlap
    try:
        dice = 2. * overlap / (union + overlap)
    except ZeroDivisionError:
        dice = 0.0

    return dice


# Way0: convert continous value to binary
def threshold_by_middle(y_pred):
    # cut by 0.5
    threshold = 0.5
    y_pred_bin = np.zeros(y_pred.shape)
    y_pred_bin[y_pred >= threshold] = 1

    return y_pred_bin


# Way1: convert continous value to binary
def threshold_by_otsu(y_pred):
    # cut by otsu threshold
    threshold = filters.threshold_otsu(y_pred)
    y_pred_bin = np.zeros(y_pred.shape)
    y_pred_bin[y_pred >= threshold] = 1

    return y_pred_bin


# Way2: convert continous value to binary
def threshold_by_f1(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    best_f1 = -1
    for index in range(len(precision)):
        curr_f1 = 2. * precision[index] * recall[index] / (precision[index] + recall[index])
        if best_f1 < curr_f1:
            best_f1 = curr_f1
            best_threshold = thresholds[index]

    y_pred_bin = np.zeros(y_pred.shape)
    y_pred_bin[y_pred >= best_threshold] = 1

    return y_pred_bin


def misc_measures(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    s1 = cm[0, 1] + cm[1, 1]
    if s1 == 0: s1 = 0.01
    precsion = 1. * cm[1, 1] / s1  
    # precsion = 1. * cm[1, 1] / (cm[0, 1] + cm[1, 1])
    s2 = cm[1, 0] + cm[1, 1]
    if s2 == 0: s2 = 0.01
    sensitivity = 1. * cm[1, 1] / s2
    # sensitivity = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
    s3 = cm[0, 1] + cm[0, 0]
    if s3 == 0: s3 = 0.01
    specificity = 1. * cm[0, 0] / s3
    # specificity = 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
    s4 = precsion + sensitivity
    if s4 == 0: s4 = 0.01
    f1score = 2 * precsion * sensitivity / s4
    # f1score = 2 * precsion * sensitivity / (precsion + sensitivity)

    return acc, precsion, sensitivity, specificity, f1score


def AUC_PR(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    PR = auc(recall, precision)

    return PR


def AUC_ROC(y_true, y_pred):
    try: 
        roc = roc_auc_score(y_true, y_pred)
    except ValueError: 
        pass 

    return roc

def get_roc_score_mutliply(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        try: 
            score = roc_auc_score(y_true[:,i], y_pred[:,i])
            scores.append(score)
        except ValueError: 
            pass 
    avg_score = np.mean(scores)
    return avg_score, scores

def get_map_score_mutliply(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        try: 
            score = average_precision_score(y_true[:,i], y_pred[:,i])
            if np.isnan(score): score = 0
            scores.append(score)
        except ValueError: 
            pass 
    avg_score = np.mean(scores)
    return avg_score, scores