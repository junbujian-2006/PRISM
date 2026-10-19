import torch
from torch import nn
from torch.nn import functional as F


class RDPE(nn.Module):
    """Estimate four rain-degradation probabilities from an RGB BCHW tensor."""

    def __init__(self):
        super().__init__()
        self.input_projection = nn.Conv2d(5, 16, 3, padding=1, bias=True)
        self.local_3 = nn.Conv2d(
            16, 16, 3, padding=1, groups=16, bias=True
        )
        self.local_5 = nn.Conv2d(
            16, 16, 5, padding=2, groups=16, bias=True
        )
        self.scale_router = nn.Conv2d(16, 2, 1, bias=True)
        self.classifier = nn.Conv2d(16, 4, 1, bias=True)
        self.register_buffer(
            "class_centers",
            torch.tensor([0.125, 0.375, 0.625, 0.875]).view(1, 4, 1, 1),
        )

    @staticmethod
    def _reflection_indices(size, padding, device):
        if size == 1:
            return torch.zeros(
                size + 2 * padding, device=device, dtype=torch.long
            )
        positions = torch.arange(
            -padding, size + padding, device=device
        )
        period = 2 * (size - 1)
        reflected = positions.remainder(period)
        return torch.where(reflected < size, reflected, period - reflected)

    @classmethod
    def _reflection_pad(cls, tensor, padding):
        height_indices = cls._reflection_indices(
            tensor.shape[-2], padding, tensor.device
        )
        width_indices = cls._reflection_indices(
            tensor.shape[-1], padding, tensor.device
        )
        return tensor.index_select(-2, height_indices).index_select(
            -1, width_indices
        )

    @classmethod
    def _local_high_frequency(cls, luminance, kernel_size):
        padding = kernel_size // 2
        padded = cls._reflection_pad(luminance, padding)
        low_frequency = F.avg_pool2d(
            padded, kernel_size=kernel_size, stride=1
        )
        return (luminance - low_frequency).abs()

    def forward(self, rainy):
        if rainy.ndim != 4 or rainy.shape[1] != 3:
            raise ValueError("rainy must have shape [B, 3, H, W]")
        luminance = (
            0.299 * rainy[:, 0:1]
            + 0.587 * rainy[:, 1:2]
            + 0.114 * rainy[:, 2:3]
        )
        high_3 = self._local_high_frequency(luminance, 3)
        high_5 = self._local_high_frequency(luminance, 5)
        features = F.silu(
            self.input_projection(torch.cat((rainy, high_3, high_5), dim=1))
        )
        branch_3 = F.silu(self.local_3(features))
        branch_5 = F.silu(self.local_5(features))
        weights = torch.softmax(self.scale_router(features), dim=1)
        fused = (
            features
            + weights[:, 0:1] * branch_3
            + weights[:, 1:2] * branch_5
        )
        logits = self.classifier(fused)
        probabilities = torch.softmax(logits, dim=1)
        score = (probabilities * self.class_centers).sum(dim=1, keepdim=True)
        return logits, probabilities, score
