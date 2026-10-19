import importlib
import importlib.util

import pytest
import torch


def test_rdpe_is_available():
    assert importlib.util.find_spec("prism") is not None
    assert hasattr(importlib.import_module("prism"), "RDPE")


@pytest.mark.parametrize("height,width", [(1, 1), (1, 5), (3, 1), (3, 7), (8, 10)])
def test_rdpe_shapes_probabilities_and_score(height, width):
    from prism import RDPE

    model = RDPE()
    rainy = torch.rand(2, 3, height, width)
    logits, probabilities, score = model(rainy)

    assert logits.shape == (2, 4, height, width)
    assert probabilities.shape == logits.shape
    assert score.shape == (2, 1, height, width)
    assert torch.isfinite(logits).all()
    assert torch.allclose(probabilities.sum(1), torch.ones_like(score[:, 0]))
    assert (probabilities >= 0).all()
    assert (score >= 0.125).all()
    assert (score <= 0.875).all()
    expected = (probabilities * model.class_centers).sum(1, keepdim=True)
    assert torch.equal(score, expected)


def test_rdpe_backward():
    from prism import RDPE

    model = RDPE()
    rainy = torch.rand(2, 3, 5, 7, requires_grad=True)
    logits, probabilities, score = model(rainy)
    (logits.square().mean() + probabilities.square().mean() + score.mean()).backward()

    assert rainy.grad is not None
    assert torch.isfinite(rainy.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("shape", [(3, 5, 7), (1, 1, 5, 7), (1, 4, 5, 7)])
def test_rdpe_rejects_invalid_input(shape):
    from prism import RDPE

    with pytest.raises(ValueError, match="shape"):
        RDPE()(torch.rand(*shape))


def test_rdpe_state_dict_round_trip():
    from prism import RDPE

    model = RDPE()
    restored = RDPE()
    restored.load_state_dict(model.state_dict(), strict=True)
    rainy = torch.rand(1, 3, 5, 7)
    for actual, expected in zip(restored(rainy), model(rainy)):
        assert torch.equal(actual, expected)
