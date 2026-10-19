# PRISM

Selected model components for image deraining:

- **RDPE**: Rain Degradation Prior Estimator.
- **DSR**: Dual-Domain Spectral Refinement.
- **HWD / HWU**: Haar-wavelet downsampling and upsampling.

This partial release does not include the complete network, training code,
datasets, or pretrained weights.

## Installation

Python 3.8 or newer is required. Install PyTorch for your platform, then run:

```bash
pip install .
```

These components run on CPU or CUDA with no custom CUDA extensions.

## Usage

```python
import torch
from prism import RDPE, DSR, HWD, HWU

rainy = torch.rand(1, 3, 128, 128)
logits, probabilities, score = RDPE()(rainy)
# logits/probabilities: [1, 4, 128, 128]; score: [1, 1, 128, 128]

features = torch.rand(1, 32, 128, 128)
refined = DSR(32)(features)
# refined: [1, 32, 128, 128]

tokens = torch.rand(1, 128 * 128, 32)
down = HWD(32)(tokens, 128, 128)
up = HWU(64)(down, 64, 64)
# down: [1, 64*64, 64]; up: [1, 128*128, 32]
```

RDPE expects RGB inputs in `[0, 1]` and returns logits, four-class
probabilities, and a probability-weighted degradation score. The class
centers are `0.125`, `0.375`, `0.625`, and `0.875`.

DSR applies gated magnitude refinement and a learned `1x1` phase convolution
before reconstructing non-negative features with IRFFT and an absolute value.
The phase convolution starts as an identity map, while the magnitude correction
starts at zero. FFT, refinement convolutions, and reconstruction run in FP32
even when autocast is enabled. DSR expects BCHW features and returns the same shape.

HWD and HWU accept flattened spatial tokens in `[B, H*W, C]` order and an even
channel count. HWD also requires even spatial dimensions. HWD and HWU contain
learned convolutions and are not exact inverses of one another.
`HaarDWT2D` and `HaarIDWT2D` expose the fixed transform pair separately.

## Tests

```bash
pip install ".[test]"
python -m pytest -q
```
