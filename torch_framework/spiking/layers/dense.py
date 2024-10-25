from math import sqrt

import torch
from torch import nn

from torch_framework.spiking.models import LeakyIandF


class Dense(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias,
                 scaling,
                 mult_factor,
                 threshold,
                 alpha,
                 use_default_initialization=True,
                 mean_gain=0,
                 std_gain=1,
                 bias_zeros=False) -> None:
        super().__init__()

        self.core_layer = nn.Linear(in_features, out_features, bias)
        self.scaling = scaling
        self.mult_factor = mult_factor
        self.threshold = threshold
        self.alpha = alpha

        if not use_default_initialization:
            mean = mean_gain * threshold / in_features
            std = sqrt(std_gain / in_features)

            nn.init.uniform_(self.core_layer.weight, mean-std, mean+std)
            if bias:
                if bias_zeros:
                    nn.init.zeros_(self.core_layer.bias)
                else:
                    nn.init.uniform_(self.core_layer.bias, -std, std)

        self.neuron = LeakyIandF.apply

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outs = self.core_layer(inputs.reshape(-1, inputs.shape[-1]))
        outs = outs.reshape(*inputs.shape[:-1], -1)
        outs = outs.permute(0, 2, 1)
        print(f"Dense: {outs.min()}, {outs.max()}, {outs.mean()}, {outs.std()}")

        return self.neuron(outs, self.threshold, self.scaling, self.mult_factor, self.alpha)
