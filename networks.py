"""
Classes of networks constructed
"""
import torch
import torch.nn as nn
from gdn import GDN
import torch.nn.functional as F


class AlexNet(nn.Module):
    """
    AlexNet implementation used in torchvision.models, can be used with
    either generalised divisive normalisation (GDN) layer or ReLU.
    """
    def __init__(self, num_classes=1000, pretrained=False):
        super(AlexNetGDN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return x


class percept(nn.Module):
    """
    Network that follows the structure of human visual system.
    """
    def __init__(self, dims=3, normalisation='gdn'):
        super(percept, self).__init__()
        if normalisation == 'batch_norm':
            self.gdn1 = nn.BatchNorm2d(dims)
            self.gdn2 = nn.BatchNorm2d(dims)
            self.gdn3 = nn.BatchNorm2d(6)
            self.gdn4 = nn.BatchNorm2d(128)
        elif normalisation == 'gdn':
            self.gdn1 = GDN(dims, apply_independently=True)
            self.gdn2 = GDN(dims)
            self.gdn3 = GDN(6)
            self.gdn4 = GDN(128)
        elif normalisation == 'instance_norm':
            self.gdn1 = nn.InstanceNorm2d(dims)
            self.gdn2 = nn.InstanceNorm2d(dims)
            self.gdn3 = nn.InstanceNorm2d(6)
            self.gdn4 = nn.InstanceNorm2d(128)
        self.conv1 = nn.Conv2d(dims, dims, kernel_size=1, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(dims, 6, kernel_size=5, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(6, 128, kernel_size=5, stride=1, padding=1)

    def forward(self, x):
        x = self.gdn1(x)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.gdn2(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.gdn3(x)
        x = self.conv3(x)
        x = self.gdn4(x)
        return x

    def features(self):
        features = nn.Sequential(
            self.gdn1,
            self.conv1,
            self.maxpool1,
            self.gdn2,
            self.conv2,
            self.maxpool2,
            self.gdn3,
            self.conv3,
            self.gdn4)
        return features


class correlation(nn.Module):
    """
    Module for measuring Pearson Correlation
    """
    def __init__(self):
        super(correlation, self).__init__()

    def forward(self, score, mos):
        score_n = score - torch.mean(score)
        mos_n = mos - torch.mean(mos)
        score_n_norm = torch.sqrt(torch.mean(score_n**2))
        mos_n_norm = torch.sqrt(torch.mean(mos_n**2))
        denom = score_n_norm * mos_n_norm
        return -torch.mean(score_n*mos_n.squeeze(), dim=0) / denom


class Dist2LogitLayer(nn.Module):
    """
    Taken from https://github.com/richzhang/PerceptualSimilarity/blob/master/models/networks_basic.py
    takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True)
    """
    def __init__(self, chn_mid=32,use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()
        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if(use_sigmoid):
            layers += [nn.Sigmoid(),]
        self.model = nn.Sequential(*layers)

    def forward(self,d0,d1,eps=0.1):
        return self.model.forward(torch.cat((d0,d1,d0-d1,d0/(d1+eps),d1/(d0+eps)),dim=1))

