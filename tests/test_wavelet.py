import pytest
import torch


@pytest.mark.parametrize("shape", [(1, 1, 2, 2), (2, 3, 8, 10)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_haar_round_trip(shape, dtype):
    from prism import HaarDWT2D, HaarIDWT2D

    image = torch.randn(*shape, dtype=dtype)
    bands = HaarDWT2D()(image)
    reconstructed = HaarIDWT2D()(*bands)

    assert len(bands) == 4
    assert bands[0].shape == (shape[0], shape[1], shape[2] // 2, shape[3] // 2)
    assert reconstructed.shape == image.shape
    assert reconstructed.dtype == dtype
    assert torch.allclose(reconstructed, image, atol=1e-6, rtol=1e-6)


def test_haar_subband_order_and_sign():
    from prism import HaarDWT2D

    image = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    bands = HaarDWT2D()(image)
    assert [band.item() for band in bands] == [5.0, 2.0, 1.0, 0.0]


def test_haar_round_trip_gradient():
    from prism import HaarDWT2D, HaarIDWT2D

    image = torch.randn(1, 2, 4, 6, dtype=torch.float64, requires_grad=True)
    reconstructed = HaarIDWT2D()(*HaarDWT2D()(image))
    reconstructed.square().sum().backward()
    assert torch.allclose(image.grad, 2 * image.detach())


@pytest.mark.parametrize("shape", [(1, 2, 3, 4), (1, 2, 4, 3), (2, 4, 4)])
def test_dwt_rejects_invalid_shape(shape):
    from prism import HaarDWT2D

    with pytest.raises(ValueError):
        HaarDWT2D()(torch.randn(*shape))


def test_idwt_rejects_mismatched_bands():
    from prism import HaarIDWT2D

    band = torch.randn(1, 2, 3, 4)
    with pytest.raises(ValueError, match="shapes"):
        HaarIDWT2D()(band, band, band, band[:, :, :, :2])
    with pytest.raises(ValueError, match="dtype"):
        HaarIDWT2D()(band, band, band, band.double())


def test_wavelet_token_shapes_and_gradients():
    from prism import HWD, HWU

    down = HWD(8)
    up = HWU(16)
    tokens = torch.randn(2, 8 * 10, 8, requires_grad=True)
    smaller = down(tokens, 8, 10)
    restored = up(smaller, 4, 5)

    assert smaller.shape == (2, 4 * 5, 16)
    assert restored.shape == tokens.shape
    restored.square().mean().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    for module in (down, up):
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("name", ["HWD", "HWU"])
def test_wavelet_resize_rejects_odd_channels(name):
    import prism

    with pytest.raises(ValueError, match="even"):
        getattr(prism, name)(3)


@pytest.mark.parametrize("name", ["HWD", "HWU"])
def test_wavelet_resize_rejects_wrong_token_shape(name):
    import prism

    with pytest.raises(ValueError, match="token shape"):
        getattr(prism, name)(8)(torch.randn(1, 10, 8), 4, 4)
