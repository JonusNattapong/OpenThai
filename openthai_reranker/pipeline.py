"""
High-Level Cross-Encoder Reranking Pipeline for Thai Information Retrieval and RAG.
Supports PyTorch (CUDA/MPS/CPU) and ONNX Runtime acceleration.
"""

from typing import List, Dict, Tuple, Union, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .utils import clean_thai_text, format_rerank_results


class OpenThaiReranker:
    """
    Thai Cross-Encoder Re-Ranker for RAG pipelines and semantic search.

    Args:
        model_name_or_path: Pretrained model identifier or path.
        device: 'cuda', 'cpu', or None (auto-detect).
        onnx_path: Path to .onnx model file for fast CPU inference.
        max_length: Maximum sequence length for (query, document) pairs.
    """

    def __init__(
        self,
        model_name_or_path: str = "airesearch/wangchanberta-base-att-spm-uncased",
        device: Optional[str] = None,
        onnx_path: Optional[str] = None,
        max_length: int = 512,
    ):
        self.model_name = model_name_or_path
        self.onnx_path = onnx_path
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if onnx_path:
            import onnxruntime as ort
            self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.device = "cpu"
            self.use_onnx = True
        else:
            self.use_onnx = False
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            # Try loading as sequence classification model with 1 output label
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name_or_path,
                    num_labels=1,
                )
            except Exception:
                # Fallback with ignore_mismatched_sizes if base model has different head
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name_or_path,
                    num_labels=1,
                    ignore_mismatched_sizes=True,
                )

            self.model.to(self.device)
            self.model.eval()

    def compute_score(
        self,
        pairs: List[Union[Tuple[str, str], List[str]]],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> List[float]:
        """
        Compute cross-encoder relevance scores for a list of (query, document) pairs.

        Args:
            pairs: List of (query, document) string tuples.
            batch_size: Mini-batch size for inference.
            normalize: Whether to apply sigmoid activation to bound scores into [0, 1].
        """
        if not pairs:
            return []

        all_scores = []
        clean_pairs = [[clean_thai_text(q), clean_thai_text(d)] for q, d in pairs]

        for i in range(0, len(clean_pairs), batch_size):
            batch = clean_pairs[i : i + batch_size]
            queries = [p[0] for p in batch]
            docs = [p[1] for p in batch]

            inputs = self.tokenizer(
                queries,
                docs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            if self.use_onnx:
                ort_inputs = {
                    "input_ids": inputs["input_ids"].numpy(),
                    "attention_mask": inputs["attention_mask"].numpy(),
                }
                logits = self.session.run(None, ort_inputs)[0]
                logits = torch.from_numpy(logits).view(-1)
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits.view(-1)

            if normalize:
                scores = torch.sigmoid(logits).cpu().tolist()
            else:
                scores = logits.cpu().tolist()

            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)

        return all_scores

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        batch_size: int = 32,
        snippet_len: int = 140,
    ) -> List[Dict[str, Any]]:
        """
        Rerank a candidate list of documents for a given query in descending relevance order.

        Args:
            query: User search query or question.
            documents: List of retrieved candidate text passages.
            top_k: Number of highest-ranked results to return (default: all).
            batch_size: Batch size for scoring.
            snippet_len: Character length of preview snippet in output.
        """
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self.compute_score(pairs, batch_size=batch_size, normalize=True)
        return format_rerank_results(documents, scores, top_k=top_k, snippet_len=snippet_len)
