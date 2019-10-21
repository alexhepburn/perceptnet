"""
Classes of networks constructed
"""
import torch
import torch.nn as nn
from gdn import GDN
import torch.nn.functional as F


class AlexNetGDN(nn.Module):
    
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


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )

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


class AutoEncoder(nn.Module):
    '''
    Convolutional autoencoder for encoding the output of percept network

    Parameters
    ----------
    n_channels : int
        Number of channels that are in the output.
    padding : int
        Number of pixels to zero pad.
    '''
    def __init__(self, n_channels, padding):
        '''
        Constructs a ``AutoEncoder`` class object.
        '''
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 16, 3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 64, 3, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, n_channels, 3, stride=2, padding=1),
            nn.Sigmoid())

    def forward(self, images):
        '''
        Forward pass of the network.

        Parameters
        ----------
        images : torch.Tensor
            Input images to be embedded and reconstructed
        
        Returns
        -------
        encoded : torch.Tensor
            The encoded images.
        decoded : torch.Tensor
            The reconstructed images.
        '''
        encoded = self.encoder(images)
        decoded = self.decoder(encoded)

        return encoded, decoded


class correlation(nn.Module):
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
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''
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

