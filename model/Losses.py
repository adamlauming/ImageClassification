import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class WeightedMSE(nn.Module):
    def __init__(self, threshold=0.3, amp=3):
        super(WeightedMSE, self).__init__()
        self.th = threshold
        self.amp = amp * 1.0

    def forward(self, y_pred, y_true):
        region = (y_true > self.th).float()
        loss = (1 + self.amp * region) * torch.pow((y_true - y_pred), 2)
        loss = loss.float().mean()

        return loss


class PositionLoss(nn.Module):
    def __init__(self, h=512, w=512):
        super(PositionLoss, self).__init__()
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        self.xx = torch.tensor(xx[np.newaxis, :, :]).float().cuda()
        self.yy = torch.tensor(yy[np.newaxis, :, :]).float().cuda()
        self.xx = self.xx.unsqueeze(0)
        self.yy = self.yy.unsqueeze(0)

    def forward(self, y_pred, pos_true):
        N = pos_true.shape[0]
        xx = self.xx.repeat(N, 1, 1, 1)
        yy = self.yy.repeat(N, 1, 1, 1)

        pos_pred = Variable(torch.zeros(N, 2), requires_grad=True).float().cuda()
        pos_pred[:, 0] = torch.sum(y_pred * xx, [1, 2, 3]) * torch.pow(torch.sum(y_pred, [1, 2, 3]), -1)
        pos_pred[:, 1] = torch.sum(y_pred * yy, [1, 2, 3]) * torch.pow(torch.sum(y_pred, [1, 2, 3]), -1)

        loss_pos = F.mse_loss(pos_pred, pos_true)

        return loss_pos


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, list): alpha = torch.Tensor(alpha)
        self.alpha = alpha.cuda()
        self.size_average = size_average
        self.smooth = 1e-6

    def forward(self, y_pred, y_true):

        pos = -y_true * ((1.0 - y_pred)**self.gamma) * torch.log(y_pred + self.smooth)
        neg = -(1.0 - y_true) * (y_pred**self.gamma) * torch.log(1.0 - y_pred + self.smooth)
        loss = self.alpha[0] * pos + self.alpha[1] * neg

        if self.size_average:
            loss_focal = loss.mean()
        else:
            loss_focal = loss.sum()

        return loss_focal


class MaskFisherLoss(nn.Module):
    def __init__(self, smooth=0.1):
        super(MaskFisherLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_true, tensor, mask):
        y_true = y_true + mask
        mask0 = (y_true == 1).repeat(1, tensor.shape[1], 1, 1)
        vec0 = torch.masked_select(tensor, mask0)
        vec0 = torch.reshape(vec0, (tensor.shape[1], -1))
        miu0 = torch.mean(vec0, dim=-1, keepdim=True)
        var0 = torch.var(vec0, dim=-1, keepdim=True)
        cov0 = 1.0 / vec0.shape[-1] * torch.mm(vec0 - miu0, torch.transpose(vec0 - miu0, 0, 1))

        mask1 = (y_true == 2).repeat(1, tensor.shape[1], 1, 1)
        vec1 = torch.masked_select(tensor, mask1)
        vec1 = torch.reshape(vec1, (tensor.shape[1], -1))
        miu1 = torch.mean(vec1, dim=-1, keepdim=True)
        var1 = torch.var(vec1, dim=-1, keepdim=True)
        cov1 = 1.0 / vec0.shape[-1] * torch.mm(vec1 - miu1, torch.transpose(vec1 - miu1, 0, 1))

        Sw = torch.det(cov0) + torch.det(cov1)  # 类内离散度
        Sb = torch.dist(miu0, miu1)  # 类间距离
        fisherloss = (Sw + self.smooth) / (Sb + self.smooth)

        return fisherloss


class FisherLoss2(nn.Module):
    def __init__(self, smooth=0.1):
        super(FisherLoss2, self).__init__()
        self.smooth = smooth

    def forward(self, tensor, y_true):
        mask0 = (y_true == 0).repeat(1, tensor.shape[1], 1, 1)
        vec0 = torch.masked_select(tensor, mask0)
        vec0 = torch.reshape(vec0, (tensor.shape[1], -1))
        miu0 = torch.mean(vec0, dim=-1, keepdim=True)
        var0 = torch.var(vec0, dim=-1, keepdim=True)
        cov0 = 1.0 / vec0.shape[-1] * torch.mm(vec0 - miu0, torch.transpose(vec0 - miu0, 0, 1))

        mask1 = (y_true == 1).repeat(1, tensor.shape[1], 1, 1)
        vec1 = torch.masked_select(tensor, mask1)
        vec1 = torch.reshape(vec1, (tensor.shape[1], -1))
        miu1 = torch.mean(vec1, dim=-1, keepdim=True)
        var1 = torch.var(vec1, dim=-1, keepdim=True)
        cov1 = 1.0 / vec0.shape[-1] * torch.mm(vec1 - miu1, torch.transpose(vec1 - miu1, 0, 1))

        Sw = torch.det(cov0) + torch.det(cov1)  # 类内离散度
        Sb = torch.dist(miu0, miu1)  # 类间距离
        fisherloss = (Sw + self.smooth) / (Sb + self.smooth)

        return fisherloss


class BCELoss(nn.Module):
    def __init__(self, weight=None):
        super(BCELoss, self).__init__()
        self.weight = weight

    def forward(self, y_pred, y_true):
        bce = F.binary_cross_entropy(y_pred, y_true, weight=self.weight)

        return bce


class DiceLoss(nn.Module):
    def __init__(self, smooth=0.01):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        intersect = (y_pred * y_true).sum()
        union = torch.sum(y_pred) + torch.sum(y_true)
        Dice = (2 * intersect + self.smooth) / (union + self.smooth)
        dice_loss = 1 - Dice

        return dice_loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        mse = F.mse_loss(y_pred, y_true)

        return mse


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = x
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


"""
# 第一版fisherloss，原理上有错误
class FisherLoss(nn.Module):
    def __init__(self, smooth=1):
        super(FisherLoss, self).__init__()
        self.smooth = smooth

    def forward(self, tensor, y_true):
        mask0 = (y_true == 0).repeat(1, tensor.shape[1], 1, 1)
        vec0 = torch.masked_select(tensor, mask0)
        vec0 = torch.reshape(vec0, (tensor.shape[0], tensor.shape[1], -1))
        miu0 = torch.mean(vec0, dim=-1, keepdim=True)
        var0 = torch.var(vec0, dim=-1, keepdim=True)

        mask1 = (y_true == 1).repeat(1, tensor.shape[1], 1, 1)
        vec1 = torch.masked_select(tensor, mask1)
        vec1 = torch.reshape(vec1, (tensor.shape[0], tensor.shape[1], -1))
        miu1 = torch.mean(vec1, dim=-1, keepdim=True)
        var1 = torch.var(vec1, dim=-1, keepdim=True)

        Sw = torch.sum(var0 + var1)  # 类内离散度
        Sb = torch.dist(miu0, miu1)  # 类间距离
        fisherloss = (Sw + self.smooth) / (Sb + self.smooth)

        return fisherloss
"""