import torch
from einops import rearrange
from torch import nn


class HaarDWT2D(nn.Module):
    """Return approximation, horizontal, vertical, and diagonal Haar subbands."""

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Haar DWT expects a BCHW tensor")
        height, width = x.shape[-2:]
        if height % 2 or width % 2:
            raise ValueError(
                "Haar DWT requires even spatial dimensions, "
                f"got {height}x{width}"
            )

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        c_a = (x00 + x01 + x10 + x11) * 0.5
        c_h = (-x00 - x01 + x10 + x11) * 0.5
        c_v = (-x00 + x01 - x10 + x11) * 0.5
        c_d = (x00 - x01 - x10 + x11) * 0.5
        return c_a, c_h, c_v, c_d


class HaarIDWT2D(nn.Module):
    """Reconstruct a BCHW tensor from four matching Haar subbands."""

    def forward(self, c_a, c_h, c_v, c_d):
        subbands = (c_a, c_h, c_v, c_d)
        if any(subband.ndim != 4 for subband in subbands):
            raise ValueError("Haar IDWT expects four BCHW subbands")
        if any(subband.shape != c_a.shape for subband in subbands[1:]):
            raise ValueError("Haar IDWT subbands must have matching shapes")
        if any(
            subband.dtype != c_a.dtype or subband.device != c_a.device
            for subband in subbands[1:]
        ):
            raise ValueError(
                "Haar IDWT subbands must use the same dtype and device"
            )

        x00 = (c_a - c_h - c_v + c_d) * 0.5
        x01 = (c_a - c_h + c_v - c_d) * 0.5
        x10 = (c_a + c_h - c_v - c_d) * 0.5
        x11 = (c_a + c_h + c_v + c_d) * 0.5
        batch, channels, height, width = c_a.shape
        output = c_a.new_empty(batch, channels, height * 2, width * 2)
        output[:, :, 0::2, 0::2] = x00
        output[:, :, 0::2, 1::2] = x01
        output[:, :, 1::2, 0::2] = x10
        output[:, :, 1::2, 1::2] = x11
        return output


class HWD(nn.Module):
    """Map [B, H*W, C] tokens to [B, (H/2)*(W/2), 2*C]."""

    def __init__(self, n_feat):
        super().__init__()
        if n_feat % 2:
            raise ValueError("HWD input channels must be even")
        self.n_feat = n_feat
        self.conv = nn.Conv2d(
            n_feat,
            n_feat // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.dwt = HaarDWT2D()
        self.projection = nn.Conv2d(
            n_feat * 2,
            n_feat * 2,
            kernel_size=1,
            groups=1,
            bias=False,
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, H, W):
        if x.ndim != 3 or x.shape[1:] != (H * W, self.n_feat):
            raise ValueError(
                "HWD token shape must be "
                f"(B, {H * W}, {self.n_feat}), got {tuple(x.shape)}"
            )
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.conv(x)
        x = torch.cat(self.dwt(x), dim=1)
        x = self.activation(self.projection(x))
        return rearrange(x, "b c h w -> b (h w) c").contiguous()


class HWU(nn.Module):
    """Map [B, H*W, C] tokens to [B, (2*H)*(2*W), C/2]."""

    def __init__(self, n_feat):
        super().__init__()
        if n_feat % 2:
            raise ValueError("HWU input channels must be even")
        self.n_feat = n_feat
        self.conv = nn.Conv2d(
            n_feat,
            n_feat * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.projection = nn.Conv2d(
            n_feat * 2,
            n_feat * 2,
            kernel_size=1,
            groups=1,
            bias=False,
        )
        self.activation = nn.ReLU(inplace=True)
        self.idwt = HaarIDWT2D()

    def forward(self, x, H, W):
        if x.ndim != 3 or x.shape[1:] != (H * W, self.n_feat):
            raise ValueError(
                "HWU token shape must be "
                f"(B, {H * W}, {self.n_feat}), got {tuple(x.shape)}"
            )
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.conv(x)
        x = self.activation(self.projection(x))
        x = self.idwt(*torch.chunk(x, 4, dim=1))
        return rearrange(x, "b c h w -> b (h w) c").contiguous()
