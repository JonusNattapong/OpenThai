"""
Ensemble Inference Engine for OpenThai-NER.
Combines multiple neural architectures (PhayaThaiBERT + WangchanBERTa)
via soft-voting probability interpolation and coherent BIO span reconstruction.
"""

from typing import List, Dict, Union, Optional
import numpy as np
import torch

from .pipeline import OpenThaiNER
from .utils import reconstruct_spans, render_html


class EnsembleOpenThaiNER:
    """
    Weighted Ensemble of multiple Thai NER models.
    Merges prediction confidence across diverse backbones to eliminate single-model errors.
    """

    def __init__(
        self,
        model_paths_or_pipelines: List[Union[str, OpenThaiNER]],
        weights: Optional[List[float]] = None,
        device: Optional[str] = None,
    ):
        self.models = []
        for item in model_paths_or_pipelines:
            if isinstance(item, OpenThaiNER):
                self.models.append(item)
            else:
                self.models.append(OpenThaiNER(item, device=device))

        if weights is None:
            # Uniform weights
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def predict(
        self,
        text: Union[str, List[str]],
        threshold: float = 0.5,
    ) -> Union[List[Dict], List[List[Dict]]]:
        """
        Run ensemble inference over text.
        Blends entity span detections using weighted probability voting.
        """
        if isinstance(text, list):
            return [self._predict_single(t, threshold) for t in text]
        return self._predict_single(text, threshold)

    def _predict_single(self, text: str, threshold: float) -> List[Dict]:
        if not text or not text.strip():
            return []

        # Gather predictions from each model
        all_model_entities = []
        for model in self.models:
            ents = model.predict(text, threshold=threshold * 0.7)  # Slightly lower threshold for candidates
            all_model_entities.append(ents)

        # Character-level voting array: text_len x num_entities
        # Merge overlapping spans by computing weighted confidence scores
        candidate_spans = []

        for m_idx, (model_ents, weight) in enumerate(zip(all_model_entities, self.weights)):
            for ent in model_ents:
                candidate_spans.append({
                    "start": ent["start"],
                    "end": ent["end"],
                    "word": ent["word"],
                    "entity": ent["entity"],
                    "score": ent["score"] * weight,
                    "model_idx": m_idx,
                })

        if not candidate_spans:
            return []

        # Group overlapping spans
        candidate_spans.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

        merged_entities = []
        current_cluster = [candidate_spans[0]]

        for span in candidate_spans[1:]:
            prev = current_cluster[-1]
            # Check overlap
            if span["start"] < prev["end"]:
                current_cluster.append(span)
            else:
                merged_entities.append(self._resolve_cluster(current_cluster, text))
                current_cluster = [span]

        if current_cluster:
            merged_entities.append(self._resolve_cluster(current_cluster, text))

        # Filter by threshold and format
        final_ents = [e for e in merged_entities if e and e["score"] >= threshold]
        final_ents.sort(key=lambda x: x["start"])
        return final_ents

    def _resolve_cluster(self, cluster: List[Dict], text: str) -> Optional[Dict]:
        """Resolve overlapping span predictions using weighted consensus."""
        if not cluster:
            return None

        # Vote on entity type weighted by score
        type_scores = {}
        for c in cluster:
            t = c["entity"]
            type_scores[t] = type_scores.get(t, 0.0) + c["score"]

        best_type = max(type_scores.items(), key=lambda x: x[1])[0]

        # Use consensus bounds
        relevant = [c for c in cluster if c["entity"] == best_type]
        start = min(c["start"] for c in relevant)
        end = max(c["end"] for c in relevant)
        word = text[start:end]

        avg_score = sum(c["score"] for c in relevant)
        avg_score = min(avg_score, 0.9999)

        return {
            "entity": best_type,
            "word": word,
            "start": start,
            "end": end,
            "score": round(float(avg_score), 4),
        }

    def render_html(self, text: str, threshold: float = 0.5) -> str:
        """Render HTML highlighted visualization."""
        entities = self.predict(text, threshold=threshold)
        return render_html(text, entities)
