import torch
from torch import nn
from torch.nn import functional as F


class DSR(nn.Module):
    """Refine spectral magnitudes and phases of a BCHW feature tensor."""

    def __init__(self, channels):
        super().__init__()
        self.expand = nn.Conv2d(channels, 2 * channels, 1, bias=True)
        self.local = nn.Conv2d(
            2 * channels,
            2 * channels,
            3,
            padding=1,
            groups=2 * channels,
            bias=True,
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=True)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        self.phase_conv = nn.Conv2d(channels, channels, 1, bias=True)
        nn.init.dirac_(self.phase_conv.weight)
        nn.init.zeros_(self.phase_conv.bias)

    @staticmethod
    def enforce_boundary_symmetry(magnitude, original_width):
        height = magnitude.shape[-2]
        paired = (-torch.arange(height, device=magnitude.device)) % height
        projected = magnitude.clone()
        boundary_columns = [0]
        if original_width % 2 == 0:
            boundary_columns.append(original_width // 2)
        for column in boundary_columns:
            projected[..., :, column] = 0.5 * (
                magnitude[..., :, column] + magnitude[..., paired, column]
            )
        return projected

    def refine_magnitude(self, magnitude, original_width):
        features = self.local(self.expand(magnitude))
        candidate, gate_logits = features.chunk(2, dim=1)
        delta = self.project(candidate * torch.sigmoid(gate_logits))
        refined = F.relu(magnitude + delta)
        return self.enforce_boundary_symmetry(refined, original_width)

    def forward(self, spatial):
        if spatial.ndim != 4:
            raise ValueError("spatial must have shape [B, C, H, W]")
        height, width = spatial.shape[-2:]
        with torch.autocast(
                device_type=spatial.device.type,
                dtype=torch.bfloat16,
                enabled=False):
            spectrum = torch.fft.rfft2(
                spatial.float(), dim=(-2, -1), norm="backward"
            )
            magnitude = self.refine_magnitude(spectrum.abs(), width)
            phase = self.phase_conv(torch.angle(spectrum))
            refined_spectrum = torch.polar(magnitude, phase)
            reconstructed = torch.fft.irfft2(
                refined_spectrum,
                s=(height, width),
                dim=(-2, -1),
                norm="backward",
            )
        return reconstructed.abs()
