import torch


class Threshold(torch.nn.Module):
    def __init__(self,
                 threshold,
                 alpha):
        super().__init__()

        self.threshold = threshold
        self.alpha = alpha

    def forward(self, inputs):
        if self.training:
            ret = (inputs - self.threshold + self.alpha) / (2 * self.alpha)
            ret = torch.clamp(ret, 0, 1)
            return ret

        return (inputs > self.threshold).float()
