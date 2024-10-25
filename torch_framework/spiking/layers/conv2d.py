from math import sqrt

import torch
from torch import nn

from torch_framework.spiking.models import LeakyIandF


class Conv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding="same",
                 bias=True,
                 scaling=1/4,
                 mult_factor=3.5/4,
                 threshold=0.5,
                 alpha=0.5,
                 use_default_initialization=True,
                 mean_gain=0,
                 std_gain=1,
                 bias_zeros=False) -> None:
        super().__init__()

        self.core_layer = nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride,
                                    padding,
                                    bias=bias)
        self.scaling = scaling
        self.mult_factor = mult_factor
        self.threshold = threshold
        self.alpha = alpha

        if isinstance(kernel_size, int):
            in_neurons = in_channels * kernel_size
        else:
            in_neurons = in_channels
            for k in kernel_size:
                in_neurons *= k

        if not use_default_initialization:
            mean = mean_gain * threshold / in_neurons
            std = sqrt(std_gain / in_neurons)

            nn.init.uniform_(self.core_layer.weight, mean-std, mean+std)
            if bias:
                if bias_zeros:
                    nn.init.zeros_(self.core_layer.bias)
                else:
                    nn.init.uniform_(self.core_layer.bias, -std, std)

        self.neuron = LeakyIandF.apply

    def forward(self, inputs):
        inputs = inputs.permute(0, 4, 1, 2, 3)
        outs = self.core_layer(inputs.reshape(-1, *inputs.shape[-3:]))
        outs = outs.reshape(*inputs.shape[:-3], -1, *outs.shape[-2:])
        outs = outs.permute(0, 2, 3, 4, 1)

        return self.neuron(outs, self.threshold, self.scaling, self.mult_factor, self.alpha)
