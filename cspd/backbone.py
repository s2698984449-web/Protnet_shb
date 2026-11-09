"""ResNet18，保留 layer3 / layer4 的空间特征，不做最终全局池化。"""

import logging

import torch
import torch.nn as nn


class SpatialResNet18(nn.Module):
    """ImageNet ResNet18。

    224 输入时:
      layer2: (B, 128, 28, 28)
      layer3: (B, 256, 14, 14)
      layer4: (B, 512, 7, 7)
    forward 仍返回 (layer3, layer4)，兼容旧调用；要 layer2 用 forward_maps。
    """

    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        import torchvision.models as models

        weights = 'IMAGENET1K_V1' if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.frozen = False
        self.train_layer4 = False
        if freeze:
            self.freeze()
        logging.info(
            'SpatialResNet18 ready: pretrained=%s freeze=%s', pretrained, freeze)

    def freeze(self):
        self.frozen = True
        self.train_layer4 = False
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            super().train(False)
        return self

    def forward_maps(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        if self.frozen:
            layer2 = layer2.detach()
        layer3 = self.layer3(layer2)
        if self.frozen:
            layer3 = layer3.detach()
        layer4 = self.layer4(layer3)
        if self.frozen and not self.train_layer4:
            layer4 = layer4.detach()
        return layer2, layer3, layer4

    def forward(self, x):
        _, layer3, layer4 = self.forward_maps(x)
        return layer3, layer4
