import torch

from optimart.data.dataset import JsonlPruneDataset, collate_scorer_batch
from optimart.data.synthetic import build_synthetic_records
from optimart.models.encoder import HashPairTokenizer
from optimart.models.loss import CompositePruneLoss
from optimart.models.scoring import build_scorer


def test_jsonl_roundtrip_and_collate():
    train, _ = build_synthetic_records(4, 2)
    tokenizer = HashPairTokenizer()
    dataset = JsonlPruneDataset(train, tokenizer, max_seq_length=64, max_segments=8)
    assert len(dataset) == 4
    batch = collate_scorer_batch([dataset[0], dataset[1]], pad_id=0)
    assert batch.input_ids.ndim == 3
    assert batch.mask.sum() >= 2
    assert batch.y_hard.max() >= 0.5


def test_tiny_scorer_backward():
    model, tokenizer = build_scorer({"backbone": "tiny", "head_hidden": 32, "dropout": 0.0})
    train, _ = build_synthetic_records(8, 2)
    dataset = JsonlPruneDataset(train, tokenizer, max_seq_length=80, max_segments=8)
    batch = collate_scorer_batch([dataset[i] for i in range(4)], pad_id=0)
    logits = model(batch.input_ids, batch.attention_mask, batch.token_type_ids)
    loss = CompositePruneLoss()(logits, batch.y_hard, batch.y_target, batch.lengths, batch.mask)
    loss.total.backward()
    grads = [p.grad.abs().sum() for p in model.parameters() if p.grad is not None]
    assert logits.shape == batch.y_hard.shape
    assert torch.isfinite(loss.total)
    assert sum(grads) > 0
