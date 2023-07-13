import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.init_weights import init_weights

#===============================================================================
# Adaptive Channel Attention
#===============================================================================


#===============================================================================
# EADAM
#===============================================================================
class EADAM_V1(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super().__init__()
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.conv_v = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_q = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)

    def forward(self, x_en, x_de):
        V = self.conv_v(x_en)
        K = self.conv_k(x_en)
        Q = self.conv_q(x_de)
        shape = V.shape

        V = V.view(V.shape[0], V.shape[1], -1)
        K = K.view(V.shape[0], K.shape[1], -1)
        Q = Q.view(Q.shape[0], Q.shape[1], -1)

        H = torch.matmul(V, torch.transpose(K, -1, -2))
        H = torch.matmul(torch.sigmoid(H), Q) / (V.shape[1] * V.shape[2])**0.5

        H = H.view(shape)
        attmap = self.conv_out(H)

        return attmap + x_en + x_de


#===============================================================================
# multiscale input Module
#===============================================================================
class MultiInput(nn.Module):
    def __init__(self, in_channels, factor):
        super().__init__()
        self.conv_in = nn.Conv2d(3, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv_out = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.factor = factor

    def forward(self, img, features):
        x = F.interpolate(img, scale_factor=self.factor, mode='bilinear', align_corners=True)
        x = self.conv_in(x)
        x = self.bn(x)
        out = torch.cat([x, features], dim=1)
        out = self.conv_out(out)

        return out


#===============================================================================
# TopHat Module
#===============================================================================
def dilation(x, size):
    maxpool = nn.MaxPool2d((size, size), stride=(1, 1), padding=size // 2)

    return maxpool(x)


def erosion(x, size):
    maxpool = nn.MaxPool2d((size, size), stride=(1, 1), padding=size // 2)

    return -maxpool(-x)


def black_tophat(x, size):
    x_dilate = dilation(x, size)
    x_close = erosion(x_dilate, size)

    return x_close - x


class TopHatBlockV2(nn.Module):
    def __init__(self, in_channels, poolsize=[3, 5, 7]):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_out = nn.Conv2d(in_channels, in_channels, 1)

        self.poolsize = poolsize
        self.conv_1x1 = nn.Conv2d(in_channels * len(poolsize), in_channels, 1)

    def forward(self, x):
        identity = x

        x = self.conv_in(x)

        out = []
        for idx, size in enumerate(self.poolsize):
            out.append(black_tophat(x, size))

        out = torch.cat(out, dim=1)
        out = self.conv_1x1(out)

        return out + identity


#===============================================================================
# PCA Blocks
#===============================================================================
class SAPCABlockV5(nn.Module):
    """
    Spatial attention block
    """

    def __init__(self, in_channels, scale=3):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        gx = self.conv1(x)
        gx_ = gx.view(gx.shape[0], gx.shape[1], -1)
        gx_ = gx_ - torch.mean(gx_, dim=-1, keepdim=True)

        cmat = torch.matmul(gx_, gx_.transpose(-1, -2)) / gx.shape[0]
        w = []
        for i in range(cmat.shape[0]):
            eigval, eigvec = torch.symeig(cmat[i], eigenvectors=True)
            w.append(eigvec[:, -16:].transpose(-1, -2))
        w = torch.stack(w, dim=0)

        z = torch.matmul(w, gx_)  # 这一步可以考虑加softmax
        z = F.softmax(z * self.scale, dim=1)
        y_ = torch.matmul(w.transpose(-1, -2), z)
        y_ = y_.view(gx.shape[0], gx.shape[1], gx.shape[2], gx.shape[3])

        attmap = self.conv2(y_)

        x_en = x + attmap

        return x_en


class SAPCABlockV4(nn.Module):
    """
    Spatial attention block
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                   # norm_layer(in_channels),
                                   )

    def forward(self, x):
        gx = self.conv1(x)
        gx_ = gx.view(gx.shape[0], gx.shape[1], -1)
        gx_ = gx_ - torch.mean(gx_, dim=-1, keepdim=True)

        cmat = torch.matmul(gx_, gx_.transpose(-1, -2)) / gx.shape[0]
        w = []
        for i in range(cmat.shape[0]):
            eigval, eigvec = torch.symeig(cmat[i], eigenvectors=True)
            w.append(eigvec[:, -x.shape[1] // 4:].transpose(-1, -2))
        w = torch.stack(w, dim=0)

        z = torch.matmul(w, gx_)  # 这一步可以考虑加softmax
        y_ = torch.matmul(w.transpose(-1, -2), z)
        y_ = y_.view(gx.shape[0], gx.shape[1], gx.shape[2], gx.shape[3])

        attmap = self.conv2(y_)

        x_en = x + attmap

        return x_en


#================================================================
#================================================================
#================================================================
class SAPCABlockV3(nn.Module):
    """
    # Spatial attention block
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels, 1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        gx = self.conv1(x)
        gx_ = gx.view(gx.shape[0], gx.shape[1], -1)
        gx_ = gx_ - torch.mean(gx_, dim=-1, keepdim=True)

        cmat = torch.matmul(gx_, gx_.transpose(-1, -2)) / gx.shape[0]
        w = []
        for i in range(cmat.shape[0]):
            eigval, eigvec = torch.symeig(cmat[i], eigenvectors=True)
            w.append(eigvec[:, -x.shape[1] // 4:].transpose(-1, -2))
        w = torch.stack(w, dim=0)

        y = torch.matmul(w, gx_)
        y = y.view(y.shape[0], y.shape[1], gx.shape[2], gx.shape[3])

        attmap = self.conv2(y)
        x_en = x + attmap

        return x_en


class SAPCABlockV2(nn.Module):
    """
    #Spatial attention block
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels, 1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        gx = self.prelu(self.conv1(x))
        gx_ = gx.view(gx.shape[0], gx.shape[1], -1)

        cmat = torch.matmul(gx_, gx_.transpose(-1, -2)) / gx.shape[0]
        w = []
        for i in range(cmat.shape[0]):
            eigval, eigvec = torch.symeig(cmat[i], eigenvectors=True)
            w.append(eigvec[:, -x.shape[1] // 4:].transpose(-1, -2))
        w = torch.stack(w, dim=0)

        y = torch.matmul(w, gx_)
        y = torch.sigmoid(y)
        y = y.view(y.shape[0], y.shape[1], gx.shape[2], gx.shape[3])

        attmap = self.conv2(y)
        x_en = x * torch.tanh(attmap)

        return x_en


class SAPCABlock(nn.Module):
    """
    #Spatial attention block
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels, 1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        gx = self.prelu(self.conv1(x))
        gx_ = gx.view(gx.shape[0], gx.shape[1], -1)

        cmat = torch.matmul(gx_, gx_.transpose(-1, -2)) / gx.shape[0]
        w = []
        for i in range(cmat.shape[0]):
            eigval, eigvec = torch.symeig(cmat[i], eigenvectors=True)
            w.append(eigvec[:, -x.shape[1] // 4:].transpose(-1, -2))
        w = torch.stack(w, dim=0)

        y = torch.matmul(w, gx_)
        y = y.view(y.shape[0], y.shape[1], gx.shape[2], gx.shape[3])

        attmap = self.conv2(y)
        x_en = x + attmap

        return x_en


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        return x

#================================================================
# SelfAM
#================================================================
class SelfAM(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(SelfAM, self).__init__()
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
    
        self.conv_v = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_q = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)

    def forward(self, x):
        V = self.conv_v(x)
        K = self.conv_k(x)
        Q = self.conv_q(x)
        shape = V.shape

        V = V.view(V.shape[0], V.shape[1], -1)
        K = K.view(V.shape[0], K.shape[1], -1)
        Q = Q.view(Q.shape[0], Q.shape[1], -1)

        H = torch.matmul(V, torch.transpose(K, -1, -2))
        H = torch.matmul(torch.sigmoid(H), Q) / (V.shape[1] * V.shape[2])**0.5

        H = H.view(shape)
        attmap = self.conv_out(H)

        return attmap + x


#================================================================
# Multi-scale Reception Composition Block (MRCB)
#================================================================
def fusion_pool(x, size):
    maxpool = nn.MaxPool2d((size, size), stride=(1, 1), padding=size // 2)
    avgpool = nn.AvgPool2d((size, size), stride=(1, 1), padding=size // 2)

    return maxpool(x) + avgpool(x)


class MRCB(nn.Module):
    def __init__(self, in_channels, multi_size=[3, 5, 7]):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_out = nn.Conv2d(in_channels, in_channels, 1)
        self.in_channels = in_channels
        self.multi_size = multi_size
        self.conv_1x1 = nn.Conv2d(in_channels * len(multi_size), in_channels, 1)

    def forward(self, x):
        print(x.shape)
        identity = x

        x = self.conv_in(x)

        out = []
        for idx, size in enumerate(self.multi_size):
            out.append(fusion_pool(x, size))

        out = torch.cat(out, dim=1)
        out = self.conv_1x1(out)
        print(out.shape)
        return out + identity


#================================================================
# Multi-stage Self-attention Fusion Module (MSFM)
#================================================================
class SelfAMv2(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(SelfAMv2, self).__init__()
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels[4] // 8
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.pool_s0 = nn.Sequential(
            nn.AvgPool2d(kernel_size=8, stride=8),
            nn.Conv2d(in_channels[0], self.inter_channels, kernel_size=1)
        )
        self.pool_s1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Conv2d(in_channels[1], self.inter_channels, kernel_size=1)
        )
        self.pool_s2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels[2], self.inter_channels, kernel_size=1)
        )
        self.pool_s3 = nn.Sequential(
            nn.Conv2d(in_channels[3], self.inter_channels, kernel_size=1)
        )

        self.conv_v = nn.Conv2d(in_channels[4], self.inter_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels[4], self.inter_channels, kernel_size=1)
        self.conv_q = nn.Conv2d(in_channels[3], self.inter_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(self.inter_channels, in_channels[4], kernel_size=1)

    def forward(self, s0, s1, s2, s3, x):
        # print(s0.shape, s1.shape, s2.shape, s3.shape, x.shape)
        s0_d = self.pool_s0(s0)
        s1_d = self.pool_s1(s1)
        s2_d = self.pool_s2(s2)
        s3_d = self.pool_s3(s3)
        # print(s0_d.shape, s1_d.shape, s2_d.shape, s3_d.shape)
        V = self.conv_v(x)
        K = self.conv_k(x)
        Q = self.conv_q(s0_d+s1_d+s2_d+s3_d)

        shape = V.shape

        V = V.view(V.shape[0], V.shape[1], -1)
        K = K.view(K.shape[0], K.shape[1], -1)
        Q = Q.view(Q.shape[0], Q.shape[1], -1)

        H = torch.matmul(K, torch.transpose(Q, -1, -2))
        H = torch.matmul(torch.softmax(H, dim=1), V) / (V.shape[1] * V.shape[2])**0.5

        H = H.view(shape)
        attmap = self.conv_out(H)

        return attmap + x


#================================================================
# Dense Infomation Correlation Block (DICB)
#================================================================
class DICB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        self.conv_x = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_y = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_cat = nn.Conv2d(in_channels * 2, self.inter_channels, kernel_size=1)
        self.conv_sum = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_mul = nn.Conv2d(10*10, self.inter_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(self.inter_channels, in_channels, 1)


    def forward(self, x, y):
        conv_x = self.conv_x(x)
        conv_y = self.conv_y(y)
        shape = conv_x.shape
        conv_x = conv_x.view(conv_x.shape[0], conv_x.shape[1], -1)
        conv_y = conv_y.view(conv_y.shape[0], conv_y.shape[1], -1)
        mul = torch.matmul(torch.transpose(conv_x, -1, -2), conv_y)
        mul = mul.view(mul.shape[0], mul.shape[1], shape[2], shape[3])

        sum = self.conv_sum(x+y)
        cat = self.conv_cat(torch.cat([x, y], dim=1))
        mul = self.conv_mul(mul)
        out = self.conv_out(sum+cat+mul)
        return out


class LOAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.maxpool = nn.AdaptiveMaxPool1d((1))
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features=in_channels, out_features=round(in_channels / 2))
        self.fc2 = nn.Linear(in_features=round(in_channels / 2), out_features=in_channels)
        self.sigmoid = nn.Sigmoid()


    def forward(self, feat):
        residual = feat
        out = self.conv1(feat)
        out = self.bn1(out)
        out1 = out
        original_out = out
        # For global average pool
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        avg_att = out
  
        # For global maximum pool
        out1 = self.maxpool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        max_att = out1
        att_weight = avg_att + max_att
        att_weight = self.sigmoid(att_weight)
        att_weight = att_weight.view(original_out.size(0), original_out.size(1), 1)
        y = att_weight * original_out

        return att_weight + residual