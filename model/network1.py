import torch.nn as nn
import torch
import torch.nn.functional as F
from model.base_models.resnet import resnet18, resnet34, resnet50, resnet101
from model.base_models.resnest import resnest50, resnest101, resnest200, resnest269
from model.base_models.PyConvResNet.pyconvresnet import pyconvresnet50, pyconvresnet101
from model.base_models.EfficientNet.model import EfficientNet
from model.base_models.resnext import resnext50_32x4d
from model.base_models.densenet import densenet121, densenet169, densenet201
from model.base_models.SEResNet import SEResnet18, CBAMResnet18, SEResnet34, CBAMResnet34
from model.base_models.ViT import ViT
from model.base_models.SwinT import SwinTransformer
import segmentation_models_pytorch as smp
from model.base_models.blocks import *


class classifer_V0(nn.Module):
    def __init__(self, backbone='resnet18', pretrained_base=True, n_class=11, in_channel=3, **kwargs):
        super(classifer_V0, self).__init__()
        self.in_channel = in_channel
        self.net = backbone
        if pretrained_base:
            self.n_class = 1000
        else:
            self.n_class = n_class

        if backbone == "resnet34":
            self.backbone = resnet34(pretrained=pretrained_base, num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [64, 128, 256, 512]
        elif backbone == "resnet18":
            self.backbone = resnet18(pretrained=pretrained_base, num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained_base, num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            self.backbone = resnet101(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnest50':
            self.backbone = resnest50(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnest101':
            self.backbone = resnest101(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnest200':
            self.backbone = resnest200(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnest269':
            self.backbone = resnest269(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'pyconvresnet50':
            self.backbone = pyconvresnet50(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'pyconvresnet101':
            self.backbone = pyconvresnet101(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif "efficientnet" in backbone:
            if pretrained_base:
                self.backbone = EfficientNet.from_pretrained(backbone)
            else:
                self.backbone = EfficientNet.from_name(backbone, in_channels=in_channel, num_classes=n_class)
            self.base_channel = [self.backbone.out_channels]
        elif backbone == 'densenet121':
            self.backbone = densenet121(pretrained=pretrained_base, num_classes=self.n_class)
            self.base_channel = [self.backbone.num_features]
        elif backbone == 'densenet169':
            self.backbone = densenet169(pretrained=pretrained_base, num_classes=self.n_class)
            self.base_channel = [self.backbone.num_features]
        elif backbone == 'densenet201':
            self.backbone = densenet201(pretrained=pretrained_base, num_classes=self.n_class)
            self.base_channel = [self.backbone.num_features]
        elif backbone == 'resnext50':
            self.backbone = resnext50_32x4d()
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'SEResnet18':
            self.backbone = SEResnet18(pretrained=pretrained_base, num_classes=self.n_class)
            self.base_channel = [64, 128, 256, 512]
        elif backbone == 'SEResnet34':
            self.backbone = SEResnet34(pretrained=pretrained_base, num_classes=self.n_class)
            self.base_channel = [64, 128, 256, 512]
        elif backbone == 'CBAMResnet18':
            self.backbone = CBAMResnet18(pretrained=pretrained_base, num_classes=self.n_class)
            self.base_channel = [64, 128, 256, 512]
        elif backbone == 'CBAMResnet34':
            self.backbone = CBAMResnet34(pretrained=pretrained_base, num_classes=self.n_class)
            self.base_channel = [64, 128, 256, 512]
        elif backbone == 'ViT':
            self.backbone = ViT(image_size=320, patch_size=32, num_classes=n_class, dim=1024, depth=6, heads=16, mlp_dim=2048, channels=in_channel)
            self.base_channel = [1024]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.base_channel[-1]),
                nn.Linear(self.base_channel[-1], n_class)
            )
        elif backbone == 'SwinT':
            self.hidden_dim = 96
            self.backbone = SwinTransformer(hidden_dim=self.hidden_dim, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), channels=in_channel, num_classes=n_class, head_dim=32, window_size=10, downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True)
            self.base_channel = [self.hidden_dim * 8]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.base_channel[-1]),
                nn.Linear(self.base_channel[-1], n_class)
            )
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.7)
        self.fc1 = nn.Linear(self.base_channel[-1], n_class)
        # self.fc1 = nn.Linear(self.base_channel[-1], 128)
        # self.relu_fc = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(128, n_class)


        # # 冻结参数
        # for p in self.backbone.parameters():
        #     p.requires_grad = False

        self.selfam = SelfAM(in_channels=self.base_channel[-1], inter_channels=512)

        self.scm = SCM(in_channels=self.base_channel[-1], inter_channels=512)
        self.fc2 = nn.Linear(512+self.base_channel[-1], n_class)
        self.avgpool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_2 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2):
        x1 = self.backbone.extract_features(x1)
        x2 = self.backbone.extract_features(x2)
        xx1, xx2 = self.scm(x1, x2)

        if self.net == 'ViT' or self.net == 'SwinT':
            x = self.mlp_head(x)
        else:
            xx1 = self.avgpool_1(xx1)
            xx2 = self.avgpool_2(xx2)
            x = xx1 + xx2
            x = x.view(x.size(0), -1)
            x = self.drop(x)
            x = self.fc2(x)
        outputs = dict()
        outputs.update({"main_out": x})
        return outputs

    # def forward(self, x1, x2):
    #     x1 = self.backbone.extract_features(x1)
    #     x2 = self.backbone.extract_features(x2)
    #     x1 = self.selfam(x1)
    #     x2 = self.selfam(x2)
    #     x = x1 + x2
    #     if self.net == 'ViT' or self.net == 'SwinT':
    #         x = self.mlp_head(x)
    #     else:
    #         x = self.avgpool(x)
    #         x = x.view(x.size(0), -1)
    #         x = self.drop(x)
    #         x = self.fc1(x)
    #     outputs = dict()
    #     outputs.update({"main_out": x})
    #     return outputs