'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2021-06-27 20:31:01
'''
import numpy as np
import torch
import torch.nn as nn
import torchsummary
from torch.nn import functional as F
from torch.nn import init
from torchvision import models

from model.layers.init_weights import init_weights
from model.layers.blocks_n import *
import segmentation_models_pytorch as smp
import model.densenet as densenet

#==============================================================================
# classifer_V40: densenet121 + Local-Object-Aware Module
#==============================================================================
class classifer_V40(nn.Module):
    def __init__(self, Flags, pretrained=True):
        super().__init__()

        filters = (64 * np.array([1, 2, 4, 8, 16])).astype(np.int)
        base = densenet.densenet121(pretrained=pretrained)

        self.in_block = nn.Sequential(
            base.conv1,
            # base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.feat_dims = 1024
        self.loa_out_dim = 128

        self.loa1_pool = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(5, stride=5),
        )
        self.loa1_fc = nn.Linear(16 * 64, self.loa_out_dim)

        self.loa2_pool = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(5, stride=5),
        )
        self.loa2_fc = nn.Linear(32 * 16, self.loa_out_dim)

        self.loa3_pool = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(5, stride=5),
        )
        self.loa3_fc = nn.Linear(64 * 4, self.loa_out_dim)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.feat_dims, 128)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128 * 8, Flags.n_class)

        self.loab = LOAB(in_channels=8)


    def forward(self, x1, x2):
        s0 = self.in_block(x1)
        s1 = self.encoder1(s0)
        loa_1 = self.loa1_fc(self.loa1_pool(s1).view(s1.size(0), -1)).unsqueeze(1)
        s2 = self.encoder2(s1)
        loa_2 = self.loa2_fc(self.loa2_pool(s2).view(s2.size(0), -1)).unsqueeze(1)
        s3 = self.encoder3(s2)
        loa_3 = self.loa3_fc(self.loa3_pool(s3).view(s3.size(0), -1)).unsqueeze(1)
        feat1 = self.encoder4(s3)
        loa_4 = self.avgpool(feat1).view(feat1.size(0), -1)
        loa_4 = self.fc1(loa_4).unsqueeze(1)

        s00 = self.in_block(x2)
        s11 = self.encoder1(s00)
        loa_11 = self.loa1_fc(self.loa1_pool(s11).view(s11.size(0), -1)).unsqueeze(1)
        s22 = self.encoder2(s11)
        loa_22 = self.loa2_fc(self.loa2_pool(s22).view(s22.size(0), -1)).unsqueeze(1)
        s33 = self.encoder3(s22)
        loa_33 = self.loa3_fc(self.loa3_pool(s33).view(s33.size(0), -1)).unsqueeze(1)
        feat2 = self.encoder4(s33)   
        loa_44 = self.avgpool(feat2).view(feat2.size(0), -1)
        loa_44 = self.fc1(loa_44).unsqueeze(1)
        
        feat_sum = torch.cat((loa_1, loa_2, loa_3, loa_4, loa_11, loa_22, loa_33, loa_44), dim=1)
        y = self.loab(feat_sum).view(feat_sum.size(0), -1)
        y = self.fc2(y)

        outputs = dict()
        outputs.update({"main_out": y})
        return outputs

#==============================================================================
# classifer_V41: SelfAMv2
#==============================================================================
class classifer_V41(nn.Module):
    def __init__(self, Flags, pretrained=True):
        super().__init__()

        filters = (64 * np.array([1, 2, 4, 8, 16])).astype(np.int)
        base = densenet.densenet121(pretrained=pretrained)

        self.in_block = nn.Sequential(
            base.conv1,
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.selfam = SelfAMv2(in_channels=filters, inter_channels=512)

        self.feat_dims = 1024

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feat_dims, Flags.n_class)


    def forward(self, x1, x2):
        s0 = self.in_block(x1)
        s1 = self.encoder1(s0)
        s2 = self.encoder2(s1)
        s3 = self.encoder3(s2)
        s4 = self.encoder4(s3)
        feat1 = self.selfam(s0, s1, s2, s3, s4)

        s00 = self.in_block(x2)
        s11 = self.encoder1(s00)
        s22 = self.encoder2(s11)
        s33 = self.encoder3(s22)
        s44 = self.encoder4(s33)    
        feat2 = self.selfam(s00, s11, s22, s33, s44)

        feat = feat1 + feat2   
        feat = self.avgpool(feat).view(feat.size(0), -1)
        y = self.fc(feat)

        outputs = dict()
        outputs.update({"main_out": y})
        return outputs

#==============================================================================
# classifer_V42: Local-Object-Aware Module + self attention
#==============================================================================
class classifer_V42(nn.Module):
    def __init__(self, Flags, pretrained=True):
        super().__init__()

        filters = (64 * np.array([1, 2, 4, 8, 16])).astype(np.int)
        base = densenet.densenet121(pretrained=pretrained)

        self.in_block = nn.Sequential(
            base.conv1,
            # base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.selfam = SelfAMv2(in_channels=filters, inter_channels=512)


        self.feat_dims = 1024
        self.loa_out_dim = 128

        self.loa1_pool = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(5, stride=5),
        )
        self.loa1_fc = nn.Linear(16 * 64, self.loa_out_dim)

        self.loa2_pool = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(5, stride=5),
        )
        self.loa2_fc = nn.Linear(32 * 16, self.loa_out_dim)

        self.loa3_pool = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(5, stride=5),
        )
        self.loa3_fc = nn.Linear(64 * 4, self.loa_out_dim)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.feat_dims, 128)
        self.avgpool_global = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(128 * 8 + self.feat_dims, Flags.n_class)

        self.loab = LOAB(in_channels=8)

        self.gamma = nn.Parameter(torch.FloatTensor(1), requires_grad=True).data.fill_(1).cuda()
        self.beta = nn.Parameter(torch.FloatTensor(1), requires_grad=True).data.fill_(1).cuda()


    def forward(self, x1, x2):
        s0 = self.in_block(x1)
        s1 = self.encoder1(s0)
        loa_1 = self.loa1_fc(self.loa1_pool(s1).view(s1.size(0), -1)).unsqueeze(1)
        s2 = self.encoder2(s1)
        loa_2 = self.loa2_fc(self.loa2_pool(s2).view(s2.size(0), -1)).unsqueeze(1)
        s3 = self.encoder3(s2)
        loa_3 = self.loa3_fc(self.loa3_pool(s3).view(s3.size(0), -1)).unsqueeze(1)
        s4 = self.encoder4(s3)
        feat1 = self.selfam(s0, s1, s2, s3, s4)
        loa_4 = self.avgpool(s4).view(s4.size(0), -1)
        loa_4 = self.fc1(loa_4).unsqueeze(1)

        s00 = self.in_block(x2)
        s11 = self.encoder1(s00)
        loa_11 = self.loa1_fc(self.loa1_pool(s11).view(s11.size(0), -1)).unsqueeze(1)
        s22 = self.encoder2(s11)
        loa_22 = self.loa2_fc(self.loa2_pool(s22).view(s22.size(0), -1)).unsqueeze(1)
        s33 = self.encoder3(s22)
        loa_33 = self.loa3_fc(self.loa3_pool(s33).view(s33.size(0), -1)).unsqueeze(1)
        s44 = self.encoder4(s33)   
        feat2 = self.selfam(s00, s11, s22, s33, s44)
        loa_44 = self.avgpool(s44).view(s44.size(0), -1)
        loa_44 = self.fc1(loa_44).unsqueeze(1)
        
        feat_sum = torch.cat((loa_1, loa_2, loa_3, loa_4, loa_11, loa_22, loa_33, loa_44), dim=1)
        feat = feat1 + feat2
        feat = self.avgpool_global(feat).view(feat.size(0), -1)
        y = self.loab(feat_sum).view(feat_sum.size(0), -1)
        y = torch.cat((feat * self.gamma, y * self.beta), dim=1) 
        y = self.fc2(y)

        outputs = dict()
        outputs.update({"main_out": y})
        return outputs