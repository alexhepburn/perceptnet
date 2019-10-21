import torch
import torch.nn as nn
import torch.nn.functional as F


class GDN(nn.Module):
    def __init__(self,
                 n_channels,
                 gamma_init=.1,
                 reparam_offset=2**-18,
                 beta_min=1e-6,
                 apply_independently=False):
        super(GDN, self).__init__()
        self.n_channels = n_channels
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset
        self.beta_min = beta_min
        self.beta_reparam = (self.beta_min + self.reparam_offset**2)**0.5
        self.apply_independently = apply_independently
        if apply_independently:
            self.groups = n_channels
        else:
            self.groups = 1
        self.initialise_params()

    def initialise_params(self):
        gamma_bound = self.reparam_offset
        gamma = torch.eye(self.n_channels, dtype=torch.float)
        gamma = gamma.view(self.n_channels, self.n_channels, 1, 1)
        gamma = torch.sqrt(self.gamma_init*gamma + self.reparam_offset**2)
        gamma = torch.mul(gamma, gamma)
        if self.apply_independently:
            gamma = gamma[:, 0, :, :].unsqueeze(1)
        self.gamma = nn.Parameter(gamma)
        beta = torch.ones((self.n_channels,))
        beta = torch.sqrt(beta + self.reparam_offset**2)
        self.beta = nn.Parameter(beta)

    def forward(self, x):
        """Forward pass of the layer
        Input must be shape: [batch_size, channels, height, width]
        """
        self.inputs = x
        self.gamma.data = torch.clamp(self.gamma.data, min=self.reparam_offset)
        self.beta.data = torch.clamp(self.beta.data, min=self.beta_reparam)
        norm_pool = F.conv2d(torch.mul(x, x), self.gamma, bias=self.beta,
                             groups=self.groups)
        norm_pool = torch.sqrt(norm_pool)
        output = x / norm_pool
        return output

