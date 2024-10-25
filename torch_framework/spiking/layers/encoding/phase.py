"""
Module containing the implementation of the `PhaseEncoding` class.
"""

import numpy as np
import torch


class PhaseEncoding(torch.nn.Module):
    """
    Class implementing phase encoding of inputs
    """

    def __init__(self,
                 timesteps: int,
                 n_bits: int = 8,
                 prescale: float = 1 / 128,
                 repeat_scale: float = 1 / 256
        ) -> None:
        """
        Initialize the `PhaseEncoding` class.

        Parameters
        ----------
        timesteps : int
            Number of timesteps to encode to

        n_bits : int, optional
            Number of bits of the input to consider for encoding
            Defaults to `8`.
            Assumes the input would in the range `[0, 1]`.

        prescale : float, optional
            Constant to multiply the encoded vector with
            Defaults to `1 / 128`.

        repeat_scale : float, optional
            Constant to scale repetitions of the encoded vector with
            Defaults to `1 / 256`.
        """

        super().__init__()

        self.timesteps = timesteps
        self.n_bits = n_bits
        self.n_repeats = int(np.ceil(timesteps / n_bits))
        self.register_buffer("bit_array",
                             torch.Tensor(1 << np.arange(n_bits)[::-1]).int(),
                             persistent=False)
        self.prescale = prescale
        self.repeat_scale = repeat_scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = (inputs * (2**self.n_bits-1)).int()
        encoded_input = torch.bitwise_and(inputs[..., None], self.bit_array).float() * self.prescale

        ret = []
        for _ in range(self.n_repeats):
            ret.append(encoded_input)
            encoded_input = encoded_input.clone() * self.repeat_scale
        ret = torch.cat(ret, dim=-1)
        ret.requires_grad = False

        return ret[..., :self.timesteps]

class PhaseTimeSeriesEncoding(torch.nn.Module):
    """
    Class implementing phase encoding of inputs of a time series
    """

    def __init__(self,
                 n_bits: int = 8,
                 initial_scale: float = None,
        ) -> None:
        """
        Initialize the `PhaseEncoding` class.

        Parameters
        ----------
        n_bits : int, optional
            Number of bits of the input to consider for encoding
            Defaults to `8`.
            Assumes the input would in the range `[0, 1]`.

        initial_scale : float, optional
            Initial scale to multiply the encoded vector with
            Defaults to `None`.
            Initial scale is set to `1 / (2**(n_bits-1))` if `None`.
        """

        super().__init__()

        self.n_bits = n_bits
        self.register_buffer("bit_array",
                             torch.Tensor(1 << np.arange(n_bits)[::-1]).int(),
                             persistent=False)

        if initial_scale is None:
            initial_scale = 1 / (2**(n_bits-1))
        self.initial_scale = initial_scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = (inputs * (2**self.n_bits-1)).int()
        encoded_input = torch.bitwise_and(inputs[..., None], self.bit_array).float() * self.initial_scale
        encoded_input = torch.reshape(encoded_input, (*inputs.shape[:-1], -1))
        encoded_input.requires_grad = False

        return encoded_input
