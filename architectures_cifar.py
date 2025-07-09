# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import export


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_scale_factor, projection_size, init_method='He', activation='relu'):
        super().__init__()

        mlp_hidden_size = round(mlp_scale_factor * in_channels)
        if activation == 'relu':
            non_linear_layer = nn.ReLU(inplace=True)
        elif activation == 'leaky relu':
            non_linear_layer = nn.LeakyReLU(inplace=True)
        elif activation == 'tanh':
            non_linear_layer = nn.Tanh()
        else:
            raise AssertionError(f'{activation} is not supported yet.')

        self.mlp_head = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            non_linear_layer,
            nn.Linear(mlp_hidden_size, projection_size)
        )
        self.init_weights(init_method)

    def init_weights(self, init_method='He'):
        for _, m in self.mlp_head.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if init_method == 'He':
                    nn.init.kaiming_normal_(m.weight.data)
                elif init_method == 'Xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, val=0)

    def forward(self, x):
        return self.mlp_head(x)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, nc=1, ema=False, mode='ALL'):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        self.ema = ema
        self.mode = mode
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(nc, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        if self.mode == 'ALL' or self.mode == 'CPRM':
            self.projector = nn.Sequential(MLPHead(nChannels[3], mlp_scale_factor=1, projection_size=128, init_method='He', activation='relu'), nn.Sigmoid(),)
        if not self.ema and (self.mode == 'ALL' or self.mode == 'SDTM'):
            self.sdp_head = nn.Sequential(MLPHead(nChannels[3], mlp_scale_factor=1, projection_size=1, init_method='He', activation='relu'), nn.Sigmoid(),)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def extra_feature(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 7)
        out = out.view(-1, self.nChannels)
        return out

    def mapping(self, x):
        x = self.extra_feature(x)
        x = self.projector(x)
        return x

    def sdp(self, x):
        x = self.extra_feature(x)
        x = self.sdp_head(x)
        return x

    def forward(self, x):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        out = self.extra_feature(x)
        logits = self.classifier(out)

        if self.mode == 'MT':
            return {'logits': logits}
        elif self.mode == 'SDTM':
            if self.ema:
                return {'logits': logits}
            else:
                out1 = out.clone()
                prob = self.sdp_head(out1)
                return {'logits': logits, 'prob': prob}
        elif self.mode == 'CPRM':
            out2 = out.clone()
            projection = self.projector(out2)
            return {'logits': logits, 'projection': projection}
        else:
            out2 = out.clone()
            projection = self.projector(out2)
            if self.ema:
                return {'logits': logits, 'projection': projection}
            else:
                out1 = out.clone()
                prob = self.sdp_head(out1)
                return {'logits': logits, 'projection': projection, 'prob': prob}


@export
def wrn_28_2(num_classes=10, ema=False, mode='ALL'):
    model = WideResNet(depth=28, widen_factor=2, num_classes=num_classes, nc=3, ema=ema, mode=mode)
    return model


@export
def wrn_28_8(num_classes=10, ema=False, mode='ALL'):
    model = WideResNet(depth=28, widen_factor=8, num_classes=num_classes, nc=3, ema=ema, mode=mode)
    return model