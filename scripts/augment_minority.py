"""
Minority Entity Data Augmentation Engine for Thai NER.
Identifies low-frequency entity classes (< 60 occurrences) and performs
safe, in-place entity swapping while preserving exact BIO sequence validity.
"""

import json
import random
from collections import defaultdict
from typing import List, Dict, Tuple


def extract_entities_from_sample(tokens: List[str], tags: List[str]) -> List[Dict]:
    """Extract full entity spans from tokens and BIO tags."""
    entities = []
    current_ent = None

    for idx, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if current_ent is not None:
                entities.append(current_ent)
            ent_type = tag[2:]
            current_ent = {"type": ent_type, "start": idx, "end": idx + 1, "tokens": [token]}
        elif tag.startswith("I-"):
            ent_type = tag[2:]
            if current_ent is not None and current_ent["type"] == ent_type:
                current_ent["end"] = idx + 1
                current_ent["tokens"].append(token)
            else:
                if current_ent is not None:
                    entities.append(current_ent)
                current_ent = {"type": ent_type, "start": idx, "end": idx + 1, "tokens": [token]}
        else:
            if current_ent is not None:
                entities.append(current_ent)
                current_ent = None

    if current_ent is not None:
        entities.append(current_ent)

    return entities


def build_entity_lexicon(samples: List[Dict]) -> Dict[str, List[List[str]]]:
    """Build dictionary of unique token sequences for each entity type."""
    lexicon = defaultdict(list)
    seen = defaultdict(set)

    for sample in samples:
        entities = extract_entities_from_sample(sample["tokens"], sample["tags"])
        for ent in entities:
            ent_type = ent["type"]
            tok_seq = tuple(ent["tokens"])
            if tok_seq not in seen[ent_type]:
                seen[ent_type].add(tok_seq)
                lexicon[ent_type].append(ent["tokens"])

    return lexicon


def augment_dataset(
    train_file: str,
    output_file: str,
    target_count: int = 12000,
    seed: int = 42,
):
    random.seed(seed)
    with open(train_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"[Augment] Loaded {len(samples)} original training samples.")

    # 1. Count entity frequencies
    entity_counts = defaultdict(int)
    for sample in samples:
        entities = extract_entities_from_sample(sample["tokens"], sample["tags"])
        for ent in entities:
            entity_counts[ent["type"]] += 1

    # Identify minority entities (frequency < 80)
    minority_types = {ent_type for ent_type, count in entity_counts.items() if count < 80}
    print(f"[Augment] Identified {len(minority_types)} minority entity types (counts < 80):")
    for t in sorted(minority_types)[:10]:
        print(f"  - {t}: {entity_counts[t]} instances")

    # 2. Build lexicon for swapping
    lexicon = build_entity_lexicon(samples)

    # Filter samples that contain at least one minority entity
    minority_samples = []
    for s in samples:
        ents = extract_entities_from_sample(s["tokens"], s["tags"])
        if any(e["type"] in minority_types for e in ents):
            minority_samples.append(s)

    print(f"[Augment] Found {len(minority_samples)} sentences containing minority entities.")

    # 3. Generate augmented instances
    augmented_samples = list(samples)
    needed = target_count - len(samples)

    attempts = 0
    generated = 0

    while generated < needed and attempts < needed * 5:
        attempts += 1
        # Sample with replacement from sentences with minority entities
        base_sample = random.choice(minority_samples if minority_samples else samples)
        orig_tokens = list(base_sample["tokens"])
        orig_tags = list(base_sample["tags"])

        ents = extract_entities_from_sample(orig_tokens, orig_tags)
        # Filter ents to replace
        candidates = [e for e in ents if e["type"] in minority_types or random.random() < 0.3]
        if not candidates:
            continue

        target_ent = random.choice(candidates)
        ent_type = target_ent["type"]
        replacement_pool = lexicon.get(ent_type, [])

        if len(replacement_pool) <= 1:
            continue

        new_tokens = random.choice(replacement_pool)
        if new_tokens == target_ent["tokens"]:
            continue

        # Splice tokens and create new BIO tags
        start_idx = target_ent["start"]
        end_idx = target_ent["end"]

        new_tags = [f"B-{ent_type}"] + [f"I-{ent_type}"] * (len(new_tokens) - 1)

        aug_tokens = orig_tokens[:start_idx] + new_tokens + orig_tokens[end_idx:]
        aug_tags = orig_tags[:start_idx] + new_tags + orig_tags[end_idx:]

        # Integrity check
        if len(aug_tokens) == len(aug_tags) and len(aug_tokens) > 0:
            augmented_samples.append({
                "tokens": aug_tokens,
                "tags": aug_tags,
                "domain": base_sample.get("domain", "augmented"),
            })
            generated += 1

    print(f"[Augment] Successfully generated {generated} new augmented samples.")
    print(f"[Augment] Total augmented dataset size: {len(augmented_samples)} sentences.")

    # Save to output file
    with open(output_file, "w", encoding="utf-8") as f:
        for s in augmented_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[Augment] Saved augmented dataset to {output_file}")


if __name__ == "__main__":
    augment_dataset("data/train.jsonl", "data/train_augmented.jsonl", target_count=12000)
