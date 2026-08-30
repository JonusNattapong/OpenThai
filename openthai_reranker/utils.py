"""
Utilities for Text Preprocessing, Truncation, and Reranking Result Formatting.
"""

from typing import List, Dict, Any


def clean_thai_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    if not text:
        return ""
    # Normalize duplicate whitespace
    cleaned = " ".join(text.strip().split())
    return cleaned


def format_rerank_results(
    documents: List[str],
    scores: List[float],
    top_k: int = None,
    snippet_len: int = 120,
) -> List[Dict[str, Any]]:
    """
    Sort candidate documents by score in descending order and return structured results.
    """
    results = []
    for idx, (doc, score) in enumerate(zip(documents, scores)):
        snippet = doc[:snippet_len] + ("..." if len(doc) > snippet_len else "")
        results.append({
            "original_index": idx,
            "document": doc,
            "snippet": snippet,
            "relevance_score": round(float(score), 4),
        })

    # Sort by relevance score descending
    results.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Re-assign final rank
    for rank, item in enumerate(results, 1):
        item["rank"] = rank

    if top_k is not None:
        results = results[:top_k]

    return results
