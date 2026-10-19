import pytest
import torch
from torch import nn


def test_dsr_public_import():
    import prism

    assert hasattr(prism, "DSR")


def test_dsr_phase_convolution_has_identity_initialization():
    from prism import DSR

    module = DSR(3)
    assert isinstance(module.phase_conv, nn.Conv2d)
    assert module.phase_conv.kernel_size == (1, 1)
    assert module.phase_conv.in_channels == module.phase_conv.out_channels == 3
    phase = torch.randn(2, 3, 7, 5)
    assert torch.equal(module.phase_conv(phase), phase)
    assert torch.count_nonzero(module.phase_conv.bias).item() == 0


@pytest.mark.parametrize("height,width", [(7, 9), (8, 10), (1, 1)])
def test_dsr_initial_output_matches_absolute_input(height, width):
    from prism import DSR

    image = torch.randn(2, 3, height, width)
    output = DSR(3)(image)
    assert output.shape == image.shape
    assert (output >= 0).all()
    assert torch.allclose(output, image.abs(), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("height,width", [(7, 9), (8, 10)])
def test_dsr_reconstructs_with_learned_phase(height, width):
    from prism import DSR

    torch.manual_seed(100)
    module = DSR(3)
    image = torch.randn(2, 3, height, width)
    with torch.no_grad():
        module.phase_conv.weight.mul_(0.5)
        module.phase_conv.bias.fill_(0.2)
    spectrum = torch.fft.rfft2(image, norm="backward")
    magnitude = module.refine_magnitude(spectrum.abs(), width)
    phase = module.phase_conv(torch.angle(spectrum))
    expected = torch.fft.irfft2(
        torch.polar(magnitude, phase), s=(height, width), norm="backward"
    ).abs()
    output = module(image)
    assert torch.allclose(output, expected)
    assert not torch.allclose(output, image.abs(), rtol=1e-4, atol=1e-5)


def test_dsr_phase_convolution_backpropagates():
    from prism import DSR

    torch.manual_seed(100)
    module = DSR(3)
    image = torch.randn(2, 3, 7, 9, requires_grad=True)
    output = module(image)
    (output * torch.randn_like(output)).mean().backward()
    assert torch.isfinite(image.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(module.phase_conv.weight.grad).item() > 0


@pytest.mark.parametrize("width", [9, 10])
def test_dsr_magnitude_projection_is_symmetric(width):
    from prism import DSR

    module = DSR(2)
    magnitude = torch.rand(1, 2, 7, width // 2 + 1)
    refined = module.enforce_boundary_symmetry(magnitude, width)
    paired = (-torch.arange(7)) % 7
    columns = [0] + ([width // 2] if width % 2 == 0 else [])
    for column in columns:
        assert torch.allclose(refined[..., :, column], refined[..., paired, column])
    interior = [column for column in range(magnitude.shape[-1]) if column not in columns]
    assert torch.equal(refined[..., interior], magnitude[..., interior])


def test_dsr_keeps_fp32_under_autocast():
    from prism import DSR

    module = DSR(3)
    observed = []
    hook = module.phase_conv.register_forward_pre_hook(
        lambda _module, inputs: observed.append(inputs[0].dtype)
    )
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = module(torch.randn(1, 3, 7, 9).bfloat16())
    finally:
        hook.remove()
    assert output.dtype == torch.float32
    assert observed == [torch.float32]


def test_dsr_strict_state_dict_round_trip():
    from prism import DSR

    model = DSR(3)
    restored = DSR(3)
    restored.load_state_dict(model.state_dict(), strict=True)
    image = torch.randn(1, 3, 7, 9)
    assert torch.equal(model(image), restored(image))


def test_dsr_rejects_missing_phase_parameters():
    from prism import DSR

    model = DSR(3)
    state = {key: value for key, value in model.state_dict().items()
             if not key.startswith("phase_conv.")}
    with pytest.raises(RuntimeError, match="phase_conv"):
        model.load_state_dict(state, strict=True)


def test_dsr_rejects_non_bchw_input():
    from prism import DSR

    with pytest.raises(ValueError, match="shape"):
        DSR(3)(torch.randn(3, 7, 9))


def test_dsr_zero_input_has_finite_gradients():
    from prism import DSR

    image = torch.zeros(1, 3, 7, 9, requires_grad=True)
    output = DSR(3)(image)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.isfinite(image.grad).all()
