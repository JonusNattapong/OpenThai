"""
Pure PyTorch Vectorized Conditional Random Field (CRF) for Token Classification.
Supports forward log-likelihood calculation, Viterbi decoding, and optional BIO transition constraints.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class CRF(nn.Module):
    """
    Linear-chain Conditional Random Field.

    Args:
        num_tags: Number of tags.
        batch_first: Whether first dimension represents batch size. Default is True.
    """

    def __init__(self, num_tags: int, batch_first: bool = True) -> None:
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        # transitions[i, j] is the score of transitioning from tag j to tag i
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize transitions with uniform random values."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def enforce_bio_constraints(self, id2label: dict[int, str]) -> None:
        """
        Enforce valid BIO sequence constraints by setting invalid transitions to a large negative score.
        For example:
          - Cannot transition to I-TAG unless from B-TAG or I-TAG of the same entity type.
          - Cannot start a sequence with I-TAG.
        """
        penalty = -10000.0
        with torch.no_grad():
            for i in range(self.num_tags):
                to_tag = id2label.get(i, "")
                if to_tag.startswith("I-"):
                    # Cannot start with I-TAG
                    self.start_transitions[i] = penalty
                    to_ent = to_tag[2:]

                    for j in range(self.num_tags):
                        from_tag = id2label.get(j, "")
                        from_ent = from_tag[2:] if "-" in from_tag else ""
                        # If from_tag is not B-ent or I-ent of the same type, penalize
                        if from_ent != to_ent:
                            self.transitions[i, j] = penalty

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: Optional[torch.ByteTensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute the negative log-likelihood of a sequence of tags given emissions.

        Args:
            emissions: (seq_len, batch_size, num_tags) or (batch_size, seq_len, num_tags)
            tags: (seq_len, batch_size) or (batch_size, seq_len)
            mask: (seq_len, batch_size) or (batch_size, seq_len)
            reduction: 'none', 'sum', 'mean', or 'token_mean'
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        # Numerator: score of the ground truth tag sequence
        numerator = self._compute_score(emissions, tags, mask)
        # Denominator: partition function Z(x) (sum over all possible sequences)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator

        if reduction == "none":
            return -llh
        if reduction == "mean":
            return -llh.mean()
        if reduction == "sum":
            return -llh.sum()
        if reduction == "token_mean":
            return -llh.sum() / mask.float().sum()
        raise ValueError(f"invalid reduction: {reduction}")

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.ByteTensor] = None,
    ) -> List[List[int]]:
        """
        Find the optimal tag sequence using the Viterbi decoding algorithm.
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        return self._viterbi_decode(emissions, mask)

    def _compute_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        seq_len, batch_size = tags.shape
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_len):
            is_valid = mask[i].float()
            trans_score = self.transitions[tags[i], tags[i - 1]]
            emission_score = emissions[i, torch.arange(batch_size), tags[i]]
            score += (trans_score + emission_score) * is_valid

        # Add end transitions for the last valid position of each sequence
        last_valid_indices = mask.long().sum(dim=0) - 1
        last_tags = tags.gather(0, last_valid_indices.unsqueeze(0)).squeeze(0)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len = emissions.size(0)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)

        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        seq_len, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)
            history.append(indices)

        score += self.end_transitions
        _, best_last_tag = score.max(dim=1)
        best_tags_arr = [best_last_tag.tolist()]

        seq_lengths = mask.long().sum(dim=0).tolist()
        best_paths = []

        for b in range(batch_size):
            best_tag = best_last_tag[b].item()
            seq_l = seq_lengths[b]
            best_path = [best_tag]

            for hist in reversed(history[: seq_l - 1]):
                best_tag = hist[b][best_tag].item()
                best_path.append(best_tag)

            best_path.reverse()
            best_paths.append(best_path)

        return best_paths
