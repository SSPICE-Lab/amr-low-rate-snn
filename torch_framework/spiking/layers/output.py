import torch
from torch import nn


class Output(torch.nn.Module):
    def __init__(self, in_features, out_features, bias: bool = True) -> None:
        super().__init__()

        self.core_layer = nn.Linear(in_features, out_features, bias)

    def forward(self, inputs):
        summed_inputs = 2 * torch.sum(inputs, dim=-1) - inputs[..., -1]
        return self.core_layer(summed_inputs)
