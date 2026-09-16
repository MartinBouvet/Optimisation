import torch

from optimart.models.loss import CompositePruneLoss


def test_composite_loss_finite_and_masked():
    loss_fn = CompositePruneLoss()
    batch, seq = 4, 12
    logits = torch.randn(batch, seq, requires_grad=True)
    y_hard = torch.zeros(batch, seq)
    y_hard[:, 0] = 1.0
    y_hard[:, 3] = 1.0
    y_target = y_hard * 0.8 + 0.1
    lengths = torch.randint(8, 40, (batch, seq)).float()
    mask = torch.ones(batch, seq)
    mask[:, -3:] = 0

    out = loss_fn(logits, y_hard, y_target, lengths, mask)
    assert torch.isfinite(out.total)
    assert 0.0 <= float(out.expected_keep_ratio.detach()) <= 1.0
    assert out.total.requires_grad


def test_coverage_penalizes_dropping_positives():
    loss_fn = CompositePruneLoss(
        lambda_rank=0.0,
        lambda_l1=0.0,
        lambda_budget=0.0,
        lambda_entropy=0.0,
        lambda_coverage=1.0,
        beta_hard=0.0,
    )
    logits_keep = torch.tensor([[8.0, -8.0]])
    logits_drop = torch.tensor([[-8.0, -8.0]])
    y_hard = torch.tensor([[1.0, 0.0]])
    y_target = y_hard.clone()
    lengths = torch.tensor([[20.0, 20.0]])
    mask = torch.ones(1, 2)

    keep = loss_fn(logits_keep, y_hard, y_target, lengths, mask)
    drop = loss_fn(logits_drop, y_hard, y_target, lengths, mask)
    assert float(drop.coverage) > float(keep.coverage)
