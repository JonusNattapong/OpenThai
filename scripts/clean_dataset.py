"""
Dataset Cleaner, Normalizer, and Stratified Splitter for OpenThai-NER-Corpus.
"""

import json
import os
import argparse
import random
from collections import Counter, defaultdict
import urllib.request

HF_DATASET_URL = "https://huggingface.co/datasets/JonusNattapong/OpenThai-NER-Corpus/raw/main/ThaiNER.jsonl"

# Comprehensive canonical mapping: resolves short forms, typos, and hyper-fine grained entities into standard schemas
CANONICAL_ENTITY_MAP = {
    # Short forms
    "LOC": "LOCATION",
    "ORG": "ORGANIZATION",
    "PER": "PERSON",
    "DAT": "DATE",
    "DTAE": "DATE",
    "DATA": "DATE",
    "DIS": "DISEASE",
    "ACC": "ACCOUNT",
    "ACCOUNT_NO": "ACCOUNT",
    "CARD_LAST4": "ACCOUNT",
    "PHONE_NO": "PHONE",
    "TEL": "PHONE",
    "MAIL": "EMAIL",
    "URL_LINK": "URL",
    "URL_PATH": "URL",
    "LINK": "URL",
    "LINK_URL": "URL",
    
    # Subtypes to canonical parents
    "DEPARTMENT": "ORGANIZATION",
    "TEAM": "ORGANIZATION",
    "STUDENT_ID": "ID",
    "IDCARD": "ID",
    "PASSPORT": "ID",
    "CASE_NO": "ID",
    "VEHICLE_NO": "ID",
    "LAW_NO": "LAW",
    "PAYMENT": "MONEY",
    "DURATION": "TIME",
    "ADDRESS": "LOCATION",
    "PORT": "LOCATION",
    "TECH": "TECHNOLOGY",
    "WORK": "PRODUCT",
    "WORK_OF_ART": "PRODUCT",
}


def download_dataset_if_needed(target_path: str):
    """Download ThaiNER.jsonl from Hugging Face if not already present."""
    if os.path.exists(target_path):
        print(f"[Info] Found existing dataset at: {target_path}")
        return

    print(f"[Info] Downloading dataset from {HF_DATASET_URL}...")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    urllib.request.urlretrieve(HF_DATASET_URL, target_path)
    print(f"[Info] Download complete: {target_path}")


def normalize_tag(tag: str, canonical_map: bool = True) -> str:
    """Normalize a single tag to standard canonical entity format."""
    tag = tag.strip()
    if tag == "O" or not tag:
        return "O"
    
    if "-" in tag:
        prefix, entity_type = tag.split("-", 1)
        prefix = prefix.upper()
        entity_type = entity_type.upper()
    else:
        prefix = "B"
        entity_type = tag.upper()

    if canonical_map and entity_type in CANONICAL_ENTITY_MAP:
        entity_type = CANONICAL_ENTITY_MAP[entity_type]

    return f"{prefix}-{entity_type}"


def fix_bio_sequence(tags: list[str], canonical_map: bool = True) -> list[str]:
    """Fix invalid BIO sequence transitions and normalize tag names."""
    fixed_tags = []
    prev_entity_type = None

    for tag in tags:
        norm_tag = normalize_tag(tag, canonical_map=canonical_map)
        if norm_tag == "O":
            fixed_tags.append("O")
            prev_entity_type = None
        elif norm_tag.startswith("B-"):
            entity_type = norm_tag[2:]
            fixed_tags.append(norm_tag)
            prev_entity_type = entity_type
        elif norm_tag.startswith("I-"):
            entity_type = norm_tag[2:]
            # If I-TAG has no preceding matching B-TAG/I-TAG, promote to B-TAG
            if prev_entity_type != entity_type:
                fixed_tags.append(f"B-{entity_type}")
            else:
                fixed_tags.append(norm_tag)
            prev_entity_type = entity_type
        else:
            fixed_tags.append("O")
            prev_entity_type = None

    return fixed_tags


def clean_and_validate(input_file: str, canonical_map: bool = True):
    """Clean samples, fix typos, validate alignment, and collect statistics."""
    valid_samples = []
    tag_counter = Counter()
    domain_samples = defaultdict(list)
    skipped_count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                skipped_count += 1
                continue

            tokens = sample.get("tokens", [])
            tags = sample.get("tags", [])
            domain = sample.get("domain", "unknown").strip()
            sample_id = sample.get("id", f"sample_{line_idx}")

            if not tokens or not tags or len(tokens) != len(tags):
                skipped_count += 1
                continue

            cleaned_tags = fix_bio_sequence(tags, canonical_map=canonical_map)
            for t in cleaned_tags:
                tag_counter[t] += 1

            clean_sample = {
                "id": sample_id,
                "domain": domain,
                "tokens": tokens,
                "tags": cleaned_tags,
            }
            valid_samples.append(clean_sample)
            domain_samples[domain].append(clean_sample)

    print(f"[Done] Processed {len(valid_samples)} valid samples (Skipped: {skipped_count})")
    print(f"[Info] Unique domains: {len(domain_samples)}")
    return valid_samples, domain_samples, tag_counter


def stratified_split(domain_samples: dict, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Perform domain-stratified train/val/test split."""
    random.seed(seed)
    train_data, val_data, test_data = [], [], []

    for domain, samples in domain_samples.items():
        shuffled = list(samples)
        random.shuffle(shuffled)
        n = len(shuffled)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # Ensure at least 1 in train if n >= 1
        if n_train == 0 and n > 0:
            n_train = 1

        train_domain = shuffled[:n_train]
        val_domain = shuffled[n_train:n_train + n_val]
        test_domain = shuffled[n_train + n_val:]

        train_data.extend(train_domain)
        val_data.extend(val_domain)
        test_data.extend(test_domain)

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser(description="Clean and split OpenThai-NER-Corpus")
    parser.add_argument("--input", default="data/ThaiNER.jsonl", help="Input raw JSONL file")
    parser.add_argument("--output_dir", default="data", help="Output directory for splits")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_canonical", action="store_true", help="Disable synonym & canonical tag mapping")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    download_dataset_if_needed(args.input)

    use_canonical = not args.no_canonical
    samples, domain_samples, tag_counts = clean_and_validate(args.input, canonical_map=use_canonical)
    train_data, val_data, test_data = stratified_split(
        domain_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Build unique canonical label mapping
    all_tags = sorted(list(tag_counts.keys()))
    if "O" in all_tags:
        all_tags.remove("O")
        all_tags = ["O"] + all_tags

    label2id = {tag: i for i, tag in enumerate(all_tags)}
    id2label = {i: tag for i, tag in enumerate(all_tags)}

    # Save splits
    for name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        out_path = os.path.join(args.output_dir, f"{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for s in split_data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[Saved] {out_path} ({len(split_data)} samples)")

    # Save label mappings
    mapping_path = os.path.join(args.output_dir, "label_map.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label, "tag_counts": dict(tag_counts)}, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {mapping_path} (Total unique labels: {len(label2id)})")


if __name__ == "__main__":
    main()
