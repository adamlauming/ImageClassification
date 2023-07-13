'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2021-06-22 15:57:37
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SCM(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(SCM, self).__init__()
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.conv_v_1 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_k_1 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_q_1 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_v_2 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_k_2 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_q_2 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

    def forward(self, x_1, x_2):
        V_1 = self.conv_v_1(x_1)
        K_1 = self.conv_k_1(x_1)
        Q_1 = self.conv_q_1(x_1)
        shape = V_1.shape
        # print(shape)
        V_1 = V_1.view(V_1.shape[0], V_1.shape[1], -1)
        K_1 = K_1.view(K_1.shape[0], K_1.shape[1], -1)
        Q_1 = Q_1.view(Q_1.shape[0], Q_1.shape[1], -1)

        V_2 = self.conv_v_2(x_2)
        K_2 = self.conv_k_2(x_2)
        Q_2 = self.conv_q_2(x_2)    
        V_2 = V_2.view(V_2.shape[0], V_2.shape[1], -1)
        K_2 = K_2.view(K_2.shape[0], K_2.shape[1], -1)
        Q_2 = Q_2.view(Q_2.shape[0], Q_2.shape[1], -1)

        R12 = torch.sigmoid(torch.matmul(torch.transpose(Q_1, -1, -2), K_2))
        R21 = torch.sigmoid(torch.matmul(torch.transpose(Q_2, -1, -2), K_1))
        # print(R12.shape)

        Fu_1 = torch.matmul(R12, torch.transpose(V_2, -1, -2))
        Fu_2 = torch.matmul(R21, torch.transpose(V_1, -1, -2))
        Fu_1 = torch.transpose(Fu_1, -1, -2).view(shape)
        Fu_2 = torch.transpose(Fu_2, -1, -2).view(shape)
        # print(Fu_1.shape)

        y_1 = torch.cat([x_1, Fu_1], dim=1)
        y_2 = torch.cat([x_2, Fu_2], dim=1)
        # print(y_1.shape)
        return y_1, y_2


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