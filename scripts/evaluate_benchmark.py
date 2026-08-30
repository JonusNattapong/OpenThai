"""
Benchmark Evaluation Script for OpenThai-NER.
Evaluates model on data/test.jsonl using seqeval (Precision, Recall, F1, Accuracy).
"""

import json
import argparse
from tqdm import tqdm
from openthai_ner import OpenThaiNER

try:
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
except ImportError:
    raise ImportError("Please install seqeval: pip install seqeval")


def evaluate_test_set(model_path: str, test_file: str = "data/test.jsonl", max_samples: int = None):
    print(f"[Info] Loading model from {model_path}...")
    ner = OpenThaiNER(model_name_or_path=model_path)

    print(f"[Info] Loading test data from {test_file}...")
    samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]

    print(f"[Info] Running evaluation on {len(samples)} samples...")
    true_labels = []
    pred_labels = []

    for s in tqdm(samples, desc="Evaluating"):
        tokens = s["tokens"]
        gold_tags = s["tags"]
        raw_text = "".join(tokens)

        extracted = ner.predict(raw_text)

        # Map back to token-level predictions for seqeval comparison
        curr_pos = 0
        token_preds = []
        for tok in tokens:
            tok_start = raw_text.find(tok, curr_pos)
            if tok_start == -1:
                tok_start = curr_pos
            tok_end = tok_start + len(tok)
            curr_pos = tok_end

            # Check if this token falls inside any predicted entity
            matched_tag = "O"
            for ent in extracted:
                if tok_start >= ent["start"] and tok_end <= ent["end"]:
                    if tok_start == ent["start"]:
                        matched_tag = f"B-{ent['entity']}"
                    else:
                        matched_tag = f"I-{ent['entity']}"
                    break
            token_preds.append(matched_tag)

        true_labels.append(gold_tags)
        pred_labels.append(token_preds)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT (seqeval):")
    print("=" * 50)
    print(classification_report(true_labels, pred_labels))

    f1 = f1_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels)
    rec = recall_score(true_labels, pred_labels)
    acc = accuracy_score(true_labels, pred_labels)

    print(f"Overall Metrics:")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  Accuracy:   {acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenThai-NER on test set")
    parser.add_argument("--model", default="JonusNattapong/OpenThai-NER", help="Model checkpoint")
    parser.add_argument("--test_file", default="data/test.jsonl", help="Path to test.jsonl")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of test samples")
    args = parser.parse_args()

    evaluate_test_set(args.model, args.test_file, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
