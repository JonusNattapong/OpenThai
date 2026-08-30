"""
Utilities for Token Parsing, BIO Span Reconstruction, and Thai Text Alignment.
"""

from typing import List, Dict, Any

# Standard color palette for Thai NER entity types (usable in HTML / Gradio / Notebooks)
ENTITY_COLORS = {
    "PERSON": "#E0F2FE",        # Light blue
    "ORGANIZATION": "#FEF3C7",  # Light yellow
    "LOCATION": "#DCFCE7",      # Light green
    "DATE": "#F3E8FF",          # Light purple
    "TIME": "#FCE7F3",          # Light pink
    "MONEY": "#ECFCCB",         # Light lime
    "PERCENT": "#CFFAFE",       # Light cyan
    "LAW": "#FEE2E2",           # Light red
    "PHONE": "#E2E8F0",         # Light slate
    "EMAIL": "#E0E7FF",         # Light indigo
    "URL": "#EDE9FE",           # Light violet
    "PRODUCT": "#FFEDD5",       # Light orange
    "ID": "#F1F5F9",            # Slate
    "FACILITY": "#D1FAE5",      # Emerald
    "DISEASE": "#FFE4E6",       # Rose
}


def reconstruct_spans(
    text: str,
    tokens: List[str],
    predictions: List[str],
    scores: List[float] = None,
) -> List[Dict[str, Any]]:
    """
    Reconstruct BIO subword predictions into cohesive entity spans with exact
    character start and end offsets in the original raw text.
    """
    if scores is None:
        scores = [1.0] * len(predictions)

    entities = []
    current_entity = None

    # SentencePiece prefix replacement: ' ' -> ' ' or ''
    # Find offsets of reconstructed text in original string
    search_cursor = 0

    for token, tag, score in zip(tokens, predictions, scores):
        # Ignore special tokens
        if token in ("<s>", "</s>", "<pad>", "<unk>", "[CLS]", "[SEP]"):
            continue

        clean_token = token.replace(" ", " ").replace(" ", "")
        if not clean_token:
            clean_token = " "

        # Locate token in original text from current cursor
        token_start = text.find(clean_token, search_cursor)
        if token_start == -1:
            token_start = search_cursor
        token_end = token_start + len(clean_token)
        search_cursor = token_end

        if tag == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        elif tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            entity_type = tag[2:]
            current_entity = {
                "entity": entity_type,
                "word": text[token_start:token_end],
                "start": token_start,
                "end": token_end,
                "score": float(score),
                "_scores": [score],
            }
        elif tag.startswith("I-"):
            entity_type = tag[2:]
            if current_entity and current_entity["entity"] == entity_type:
                # Extend current entity
                current_entity["end"] = token_end
                current_entity["word"] = text[current_entity["start"]:token_end]
                current_entity["_scores"].append(score)
                current_entity["score"] = float(sum(current_entity["_scores"]) / len(current_entity["_scores"]))
            else:
                # Orphan I- tag: treat as new entity
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "entity": entity_type,
                    "word": text[token_start:token_end],
                    "start": token_start,
                    "end": token_end,
                    "score": float(score),
                    "_scores": [score],
                }

    if current_entity:
        entities.append(current_entity)

    # Clean up internal fields
    for ent in entities:
        ent.pop("_scores", None)
        ent["word"] = ent["word"].strip()

    # Filter out empty span strings
    entities = [e for e in entities if e["word"]]
    return entities


def render_html_highlight(text: str, entities: List[Dict[str, Any]]) -> str:
    """Render interactive highlighted HTML snippet for entities."""
    # Sort entities by start index
    entities = sorted(entities, key=lambda x: x["start"])
    html_parts = []
    cursor = 0

    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        label = ent["entity"]
        color = ENTITY_COLORS.get(label, "#F3F4F6")

        # Text before entity
        if start > cursor:
            html_parts.append(text[cursor:start])

        # Entity badge
        entity_text = text[start:end]
        html_parts.append(
            f'<mark style="background-color: {color}; border-radius: 4px; padding: 2px 6px; margin: 0 2px; font-weight: 500;">'
            f'{entity_text} <span style="font-size: 0.75em; opacity: 0.8; text-transform: uppercase; font-weight: 700; color: #374151;">[{label}]</span>'
            f'</mark>'
        )
        cursor = end

    if cursor < len(text):
        html_parts.append(text[cursor:])

    return f'<div style="line-height: 2.2; font-family: sans-serif; font-size: 16px;">{"".join(html_parts)}</div>'
