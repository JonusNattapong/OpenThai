"""
Loss Functions for Token Classification: Focal Loss & Inverse Class Weighting.
Addresses heavy class imbalance in Thai NER datasets.
"""

from typing import Optional, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    tag_counts: Dict[str, int],
    label2id: Dict[str, int],
    power: float = 0.5,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """
    Compute smooth inverse class frequency weights from label frequency distribution.

    Args:
        tag_counts: Dictionary mapping tag names to frequency counts.
        label2id: Mapping from tag string to integer index.
        power: Damping power (0.5 for square root damping, prevents extreme weights).
        max_weight: Upper bound clip for rare class weights.
    """
    num_classes = len(label2id)
    weights = torch.ones(num_classes, dtype=torch.float32)

    total_count = sum(tag_counts.values())
    if total_count == 0:
        return weights

    median_count = sorted(tag_counts.values())[len(tag_counts) // 2]

    for tag, idx in label2id.items():
        count = tag_counts.get(tag, 1)
        # Smoothed inverse frequency
        w = math.pow(median_count / max(count, 1), power)
        weights[idx] = min(w, max_weight)

    # Normalize weights so mean is 1.0
    weights = weights / weights.mean()
    return weights


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss for Token Classification.
    Downweights well-classified easy examples and focuses gradient on hard/rare entities.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, seq_len, num_classes) or (N, num_classes)
            targets: (batch_size, seq_len) or (N,)
        """
        # Reshape to 2D
        if logits.dim() > 2:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

        # Filter out ignored indices (-100)
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Gather probability of target class
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weighting factor: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - target_probs, self.gamma)

        loss = -focal_weight * target_log_probs

        # Apply class weights if provided
        if self.alpha is not None:
            class_weight = self.alpha[targets]
            loss = loss * class_weight

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
