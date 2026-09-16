"""Composite training loss for the Optimart relevance scorer.

Mathematical specification
--------------------------
Let q be the query, C = {c_1, …, c_n} the segmented context, ℓ_i the token
length of c_i, and s_i = σ(f_θ(q, c_i)) ∈ (0, 1) the predicted keep
probability (a normalized proxy of I(c_i ; q)).

Gold:
  y_i^h ∈ {0, 1}   extractive / supporting-fact indicator
  y_i^* ∈ [0, 1]   mixed target  α y_i^h + (1-α) σ(PMI_i)

Loss:

  L(θ) = L_sem
       + λ_rank L_rank
       + λ_1    L_1
       + λ_B    L_B
       + λ_cov  L_cov
       + λ_H    H(s)

1. Semantic term (imbalanced relevance)
   L_sem = β L_focal(s, y^h) + (1-β) SmoothL1(s, y^*)

   Focal BCE with logits z_i, p_i = σ(z_i):
   L_focal = − (1/Z) Σ_i m_i α_{y_i} (1 − p_{t,i})^γ log p_{t,i}

2. Pairwise ranking (supporting facts must outscore distractors)
   L_rank = mean_{i∈P, j∈N}  log(1 + exp(−(s_i − s_j) / τ))
   Negatives and positives are subsampled per example.

3. L1 sparsity (force pruning)
   L_1 = (Σ_i m_i s_i) / (Σ_i m_i)

4. Budget hinge — differentiable knapsack capacity
   L̂ = Σ_i s_i ℓ_i ,   L_tot = Σ_i ℓ_i ,   B = ρ L_tot
   L_B = ReLU(L̂ − B) / L_tot

5. Coverage hinge — do not delete oracle-positive tokens
   L̂⁺ = Σ_{i: y_i^h=1} s_i ℓ_i ,   L⁺ = Σ_{i: y_i^h=1} ℓ_i
   L_cov = ReLU(κ L⁺ − L̂⁺) / max(L⁺, 1)

6. Bernoulli entropy — polarize scores toward 0/1 for discrete knapsack
   H(s) = − (1/Z) Σ_i m_i [ s_i log s_i + (1−s_i) log(1−s_i) ]

The coverage term is what keeps L1/budget from collapsing accuracy.
At inference, s_i is frozen and a 0-1 knapsack under budget B is solved.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class PruneLossOutput:
    total: Tensor
    semantic: Tensor
    ranking: Tensor
    l1: Tensor
    budget: Tensor
    coverage: Tensor
    entropy: Tensor
    expected_keep_ratio: Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "loss_semantic": float(self.semantic.detach()),
            "loss_ranking": float(self.ranking.detach()),
            "loss_l1": float(self.l1.detach()),
            "loss_budget": float(self.budget.detach()),
            "loss_coverage": float(self.coverage.detach()),
            "loss_entropy": float(self.entropy.detach()),
            "expected_keep_ratio": float(self.expected_keep_ratio.detach()),
        }


class CompositePruneLoss(nn.Module):
    def __init__(
        self,
        beta_hard: float = 0.7,
        lambda_rank: float = 0.3,
        lambda_l1: float = 0.05,
        lambda_budget: float = 0.2,
        lambda_coverage: float = 0.5,
        lambda_entropy: float = 0.02,
        budget_ratio: float = 0.45,
        coverage_ratio: float = 0.95,
        focal_gamma: float = 2.0,
        pos_weight: float = 6.0,
        rank_tau: float = 0.15,
        max_pos_pairs: int = 8,
        max_neg_pairs: int = 16,
    ) -> None:
        super().__init__()
        if not 0.0 <= beta_hard <= 1.0:
            raise ValueError("beta_hard must be in [0, 1]")
        if not 0.0 < budget_ratio < 1.0:
            raise ValueError("budget_ratio must be in (0, 1)")
        self.beta_hard = beta_hard
        self.lambda_rank = lambda_rank
        self.lambda_l1 = lambda_l1
        self.lambda_budget = lambda_budget
        self.lambda_coverage = lambda_coverage
        self.lambda_entropy = lambda_entropy
        self.budget_ratio = budget_ratio
        self.coverage_ratio = coverage_ratio
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight
        self.rank_tau = rank_tau
        self.max_pos_pairs = max_pos_pairs
        self.max_neg_pairs = max_neg_pairs

    def forward(
        self,
        logits: Tensor,
        y_hard: Tensor,
        y_target: Tensor,
        lengths: Tensor,
        mask: Tensor,
    ) -> PruneLossOutput:
        """Compute the composite loss.

        Args:
            logits: (B, S) raw scores from the encoder head.
            y_hard: (B, S) {0,1} supporting-fact / answer-span labels.
            y_target: (B, S) mixed regression targets in [0, 1].
            lengths: (B, S) token counts ℓ_i.
            mask: (B, S) 1 for real segments, 0 for padding.
        """
        mask_f = mask.float()
        valid = mask_f.sum().clamp_min(1.0)
        probs = torch.sigmoid(logits) * mask_f

        semantic = self._semantic_loss(logits, y_hard, y_target, mask_f, valid)
        ranking = self._pairwise_rank_loss(probs, y_hard, mask)
        l1 = probs.sum() / valid

        token_mass = (probs * lengths).sum(dim=-1)
        token_total = (lengths * mask_f).sum(dim=-1).clamp_min(1.0)
        budget = F.relu(token_mass - self.budget_ratio * token_total) / token_total
        budget = budget.mean()

        pos_mask = (y_hard > 0.5).float() * mask_f
        pos_mass = (probs * lengths * pos_mask).sum(dim=-1)
        pos_total = (lengths * pos_mask).sum(dim=-1).clamp_min(1.0)
        coverage = F.relu(self.coverage_ratio * pos_total - pos_mass) / pos_total
        coverage = coverage.mean()

        entropy = self._bernoulli_entropy(probs, mask_f, valid)
        keep_ratio = (token_mass / token_total).mean()

        total = (
            semantic
            + self.lambda_rank * ranking
            + self.lambda_l1 * l1
            + self.lambda_budget * budget
            + self.lambda_coverage * coverage
            + self.lambda_entropy * entropy
        )
        return PruneLossOutput(
            total=total,
            semantic=semantic,
            ranking=ranking,
            l1=l1,
            budget=budget,
            coverage=coverage,
            entropy=entropy,
            expected_keep_ratio=keep_ratio,
        )

    def _semantic_loss(
        self,
        logits: Tensor,
        y_hard: Tensor,
        y_target: Tensor,
        mask_f: Tensor,
        valid: Tensor,
    ) -> Tensor:
        probs = torch.sigmoid(logits)
        p_t = torch.where(y_hard > 0.5, probs, 1.0 - probs).clamp(1e-7, 1.0 - 1e-7)
        weight = torch.where(y_hard > 0.5, torch.full_like(y_hard, self.pos_weight), torch.ones_like(y_hard))
        focal = (1.0 - p_t).pow(self.focal_gamma)
        bce = -torch.log(p_t)
        focal_bce = (mask_f * weight * focal * bce).sum() / valid.clamp_min(1.0)

        residual = F.smooth_l1_loss(probs, y_target, reduction="none")
        regression = (mask_f * residual).sum() / valid
        return self.beta_hard * focal_bce + (1.0 - self.beta_hard) * regression

    def _pairwise_rank_loss(self, probs: Tensor, y_hard: Tensor, mask: Tensor) -> Tensor:
        losses: list[Tensor] = []
        for b in range(probs.size(0)):
            valid_idx = mask[b].bool()
            scores = probs[b, valid_idx]
            labels = y_hard[b, valid_idx]
            pos = scores[labels > 0.5]
            neg = scores[labels <= 0.5]
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            pos = pos[torch.randperm(pos.numel(), device=pos.device)[: self.max_pos_pairs]]
            neg = neg[torch.randperm(neg.numel(), device=neg.device)[: self.max_neg_pairs]]
            delta = (pos.unsqueeze(1) - neg.unsqueeze(0)) / self.rank_tau
            losses.append(F.softplus(-delta).mean())
        if not losses:
            return probs.sum() * 0.0
        return torch.stack(losses).mean()

    def _bernoulli_entropy(self, probs: Tensor, mask_f: Tensor, valid: Tensor) -> Tensor:
        p = probs.clamp(1e-7, 1.0 - 1e-7)
        ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        return (mask_f * ent).sum() / valid
