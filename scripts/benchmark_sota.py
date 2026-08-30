"""
Benchmark Evaluation & SOTA Comparison Suite for Thai NER.
Evaluates and benchmarks multiple Thai NER models side-by-side using seqeval.
"""

import json
import time
import argparse
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

try:
    from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
except ImportError:
    raise ImportError("Please install seqeval: pip install seqeval")


def load_test_samples(filepath: str, max_samples: int = 100) -> List[Dict]:
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if max_samples and len(samples) >= max_samples:
                    break
    return samples


def evaluate_model_pipeline(
    model_id_or_path: str,
    samples: List[Dict],
    device: str = "cpu",
    desc: str = "Model",
) -> Dict:
    print(f"\n[Benchmarking] {desc} ({model_id_or_path})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        model = AutoModelForTokenClassification.from_pretrained(model_id_or_path)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"  Failed to load {model_id_or_path}: {e}")
        return None

    id2label = model.config.id2label

    true_tags_all = []
    pred_tags_all = []
    total_tokens = 0

    start_time = time.time()

    for sample in samples:
        tokens = sample["tokens"]
        gold_tags = sample["tags"]
        raw_text = "".join(tokens)
        total_tokens += len(tokens)

        inputs = tokenizer(raw_text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        pred_ids = torch.argmax(outputs.logits, dim=-1)[0].tolist()
        pred_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

        # Map back to character offsets
        pred_entities = []
        cursor = 0
        current_ent = None

        for t, pid in zip(pred_tokens, pred_ids):
            if t in ("<s>", "</s>", "<pad>", "<unk>"):
                continue
            clean_t = t.replace(" ", " ").replace(" ", "")
            if not clean_t:
                clean_t = " "

            t_start = raw_text.find(clean_t, cursor)
            if t_start == -1:
                t_start = cursor
            t_end = t_start + len(clean_t)
            cursor = t_end

            tag = id2label.get(pid, id2label.get(str(pid), "O"))
            if tag != "O":
                pred_entities.append({"start": t_start, "end": t_end, "tag": tag})

        # Match back to original word tokens
        word_cursor = 0
        sample_pred_tags = []
        for word in tokens:
            w_start = raw_text.find(word, word_cursor)
            if w_start == -1:
                w_start = word_cursor
            w_end = w_start + len(word)
            word_cursor = w_end

            matched = "O"
            for ent in pred_entities:
                if ent["start"] <= w_start and ent["end"] >= w_end:
                    matched = ent["tag"]
                    break
            sample_pred_tags.append(matched)

        true_tags_all.append(gold_tags)
        pred_tags_all.append(sample_pred_tags)

    elapsed = time.time() - start_time
    ms_per_sent = (elapsed / len(samples)) * 1000
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

    f1 = f1_score(true_tags_all, pred_tags_all)
    prec = precision_score(true_tags_all, pred_tags_all)
    rec = recall_score(true_tags_all, pred_tags_all)
    acc = accuracy_score(true_tags_all, pred_tags_all)

    return {
        "name": desc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "accuracy": acc,
        "latency_ms": ms_per_sent,
        "throughput": tokens_per_sec,
    }


def print_leaderboard(results: List[Dict]):
    print("\n" + "=" * 75)
    print("🏆 THAI NER SOTA BENCHMARK LEADERBOARD")
    print("=" * 75)
    print(f"{'Model Name':<32} | {'F1':>7} | {'Precision':>9} | {'Recall':>7} | {'Speed (tok/s)':>13}")
    print("-" * 75)
    for r in sorted(results, key=lambda x: x["f1"], reverse=True):
        print(f"{r['name']:<32} | {r['f1']*100:>6.2f}% | {r['precision']*100:>8.2f}% | {r['recall']*100:>6.2f}% | {r['throughput']:>11.1f}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Thai NER SOTA Benchmark Suite")
    parser.add_argument("--test_file", default="data/test.jsonl", help="Test dataset path")
    parser.add_argument("--samples", type=int, default=100, help="Number of test sentences")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry_run", action="store_true", help="Run quick dry run on 5 samples")
    args = parser.parse_args()

    num_samples = 5 if args.dry_run else args.samples
    samples = load_test_samples(args.test_file, max_samples=num_samples)
    print(f"[Info] Loaded {len(samples)} test samples from {args.test_file}")

    models_to_benchmark = [
        ("JonusNattapong/OpenThai-NER", "OpenThai-NER (Baseline)"),
        ("models/openthai-ner-final", "OpenThai-NER (Final Release)"),
        ("pythainlp/thainer-corpus-v2-base-model", "PyThaiNLP ThaiNER-v2"),
    ]

    results = []
    for model_path, display_name in models_to_benchmark:
        res = evaluate_model_pipeline(model_path, samples, device=args.device, desc=display_name)
        if res:
            results.append(res)

    if results:
        print_leaderboard(results)


if __name__ == "__main__":
    main()
