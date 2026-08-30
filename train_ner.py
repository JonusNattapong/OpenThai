"""
Self-Contained Production Training Pipeline for OpenThai-NER.
Auto-installs dependencies, auto-prepares cleaned dataset if missing,
fixes eval_loss NaN, ensures proper subword label alignment, and computes strict Entity-F1 via seqeval.
"""

import os
import sys
import json
import argparse
import subprocess
import random
import urllib.request
from collections import Counter, defaultdict

# Auto-install necessary dependencies on remote/fresh VMs (e.g. Colab)
def ensure_dependencies():
    packages = []
    try:
        import sentencepiece
    except ImportError:
        packages.append("sentencepiece")
    try:
        import seqeval
    except ImportError:
        packages.append("seqeval")
    try:
        import evaluate
    except ImportError:
        packages.append("evaluate")

    if packages:
        print(f"[Setup] Installing required packages: {packages}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)

ensure_dependencies()

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    set_seed,
)

try:
    import evaluate
    METRIC_BACKEND = "evaluate"
except ImportError:
    try:
        from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
        METRIC_BACKEND = "seqeval_direct"
    except ImportError:
        METRIC_BACKEND = None


HF_CORPUS_URL = "https://huggingface.co/datasets/JonusNattapong/OpenThai-NER-Corpus/raw/main/ThaiNER.jsonl"

CANONICAL_ENTITY_MAP = {
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


def auto_prepare_dataset(data_dir: str):
    """Auto-download and prepare clean dataset splits if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, "ThaiNER.jsonl")
    label_map_path = os.path.join(data_dir, "label_map.json")

    if not os.path.exists(raw_path):
        print(f"[Info] Downloading OpenThai-NER-Corpus from {HF_CORPUS_URL}...")
        urllib.request.urlretrieve(HF_CORPUS_URL, raw_path)
        print("[Info] Download complete.")

    print("[Info] Cleaning and formatting dataset...")
    valid_samples = []
    tag_counter = Counter()
    domain_samples = defaultdict(list)

    with open(raw_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except Exception:
                continue

            tokens = sample.get("tokens", [])
            tags = sample.get("tags", [])
            domain = sample.get("domain", "unknown").strip()

            if not tokens or not tags or len(tokens) != len(tags):
                continue

            fixed_tags = []
            prev_type = None
            for tag in tags:
                tag = tag.strip()
                if tag == "O" or not tag:
                    fixed_tags.append("O")
                    prev_type = None
                else:
                    prefix = "B" if not "-" in tag else tag.split("-")[0].upper()
                    ent = tag.split("-")[-1].upper()
                    ent = CANONICAL_ENTITY_MAP.get(ent, ent)
                    norm_tag = f"{prefix}-{ent}"

                    if norm_tag.startswith("I-") and prev_type != ent:
                        norm_tag = f"B-{ent}"
                    fixed_tags.append(norm_tag)
                    prev_type = ent

            for t in fixed_tags:
                tag_counter[t] += 1

            clean_sample = {
                "id": sample.get("id", f"s_{idx}"),
                "domain": domain,
                "tokens": tokens,
                "tags": fixed_tags,
            }
            valid_samples.append(clean_sample)
            domain_samples[domain].append(clean_sample)

    print(f"[Done] Valid samples extracted: {len(valid_samples)}")

    # Stratified split
    train_data, val_data, test_data = [], [], []
    random.seed(42)
    for dom, s_list in domain_samples.items():
        shuffled = list(s_list)
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * 0.8)) if n > 0 else 0
        n_val = int(n * 0.1)

        train_data.extend(shuffled[:n_train])
        val_data.extend(shuffled[n_train:n_train + n_val])
        test_data.extend(shuffled[n_train + n_val:])

    for name, split_set in [("train", train_data), ("val", val_data), ("test", test_data)]:
        with open(os.path.join(data_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for item in split_set:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    all_tags = sorted(list(tag_counter.keys()))
    if "O" in all_tags:
        all_tags.remove("O")
        all_tags = ["O"] + all_tags

    label2id = {tag: i for i, tag in enumerate(all_tags)}
    id2label = {i: tag for i, tag in enumerate(all_tags)}

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label, "tag_counts": dict(tag_counter)}, f, ensure_ascii=False, indent=2)

    print(f"[Done] Dataset prepared successfully in {data_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train OpenThai-NER Model")
    parser.add_argument("--model_name", default="Pavarissy/phayathaibert-thainer", help="Base model checkpoint")
    parser.add_argument("--data_dir", default="data", help="Directory containing train.jsonl, val.jsonl, test.jsonl")
    parser.add_argument("--output_dir", default="models/openthai-ner-final", help="Directory to save model")
    parser.add_argument("--batch_size", type=int, default=16, help="Per device train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Per device eval batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Total training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU training")
    parser.add_argument("--label_all_subwords", action="store_true", help="Assign I-label to subsequent subwords instead of -100")
    parser.add_argument("--use_crf", action="store_true", help="Enable Linear-Chain CRF Layer with BIO constraints")
    parser.add_argument("--loss_type", default="ce", choices=["ce", "focal", "crf"], help="Loss function type (ce, focal, crf)")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    parser.add_argument("--use_augmented", action="store_true", help="Use data/train_augmented.jsonl instead of train.jsonl")
    parser.add_argument("--llrd", action="store_true", help="Enable Layer-wise Learning Rate Decay (LLRD)")
    parser.add_argument("--llrd_decay", type=float, default=0.8, help="Layer-wise LR decay factor (default: 0.8)")
    return parser.parse_args()


def tokenize_and_align_labels(examples, tokenizer, label2id, max_length=256, label_all_subwords=False):
    """Align word-level BIO tags with SentencePiece subwords and mask special tokens with -100."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    labels = []
    for i, tags in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        prev_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                tag = tags[word_idx] if word_idx < len(tags) else "O"
                label_ids.append(label2id.get(tag, label2id.get("O", 0)))
            else:
                if label_all_subwords:
                    tag = tags[word_idx] if word_idx < len(tags) else "O"
                    if tag.startswith("B-"):
                        tag = "I-" + tag[2:]
                    label_ids.append(label2id.get(tag, label2id.get("O", 0)))
                else:
                    label_ids.append(-100)
            prev_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def build_compute_metrics(id2label):
    """Build seqeval metric calculation function."""
    if METRIC_BACKEND == "evaluate":
        seqeval_metric = evaluate.load("seqeval")

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            true_predictions = [
                [id2label[p_idx] for (p_idx, l_idx) in zip(prediction, label) if l_idx != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [id2label[l_idx] for (p_idx, l_idx) in zip(prediction, label) if l_idx != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
        return compute_metrics

    elif METRIC_BACKEND == "seqeval_direct":
        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            true_predictions = [
                [id2label[p_idx] for (p_idx, l_idx) in zip(prediction, label) if l_idx != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [id2label[l_idx] for (p_idx, l_idx) in zip(prediction, label) if l_idx != -100]
                for prediction, label in zip(predictions, labels)
            ]

            return {
                "precision": precision_score(true_labels, true_predictions),
                "recall": recall_score(true_labels, true_predictions),
                "f1": f1_score(true_labels, true_predictions),
                "accuracy": accuracy_score(true_labels, true_predictions),
            }
def create_optimizer_with_llrd(model, base_lr=2e-5, decay_rate=0.8, weight_decay=0.01, crf_lr=5e-4):
    """
    Creates an AdamW optimizer with Layer-wise Learning Rate Decay (LLRD).
    Lower layers receive smaller learning rates to retain general Thai representations,
    while classifier and CRF layers receive higher learning rates.
    """
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = []

    backbone = getattr(model, "roberta", getattr(model, "camembert", None))
    if backbone is not None and hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        num_layers = len(backbone.encoder.layer)

        # 1. Embeddings
        embed_lr = base_lr * (decay_rate ** (num_layers + 1))
        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in backbone.embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": embed_lr,
            },
            {
                "params": [p for n, p in backbone.embeddings.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": embed_lr,
            },
        ])

        # 2. Transformer layers (0 to num_layers - 1)
        for layer_idx, layer in enumerate(backbone.encoder.layer):
            layer_lr = base_lr * (decay_rate ** (num_layers - layer_idx))
            optimizer_grouped_parameters.extend([
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                    "lr": layer_lr,
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": layer_lr,
                },
            ])

        # 3. Classifier head
        head_params = [p for n, p in model.named_parameters() if "classifier" in n or "dropout" in n]
        if head_params:
            optimizer_grouped_parameters.append({
                "params": head_params,
                "weight_decay": weight_decay,
                "lr": base_lr * 2.0,
            })

        # 4. CRF layer
        crf_params = [p for n, p in model.named_parameters() if "crf" in n]
        if crf_params:
            optimizer_grouped_parameters.append({
                "params": crf_params,
                "weight_decay": 0.0,
                "lr": crf_lr,
            })

        return torch.optim.AdamW(optimizer_grouped_parameters)
    else:
        return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)


def main():
    args = parse_args()
    set_seed(args.seed)

    label_map_file = os.path.join(args.data_dir, "label_map.json")
    if not os.path.exists(label_map_file):
        print(f"[Info] {label_map_file} not found. Preparing dataset automatically...")
        auto_prepare_dataset(args.data_dir)

    with open(label_map_file, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
        label2id = mapping_data["label2id"]
        id2label = {int(k): v for k, v in mapping_data["id2label"].items()}

    print(f"[Info] Loaded {len(label2id)} unique labels from {label_map_file}")

    train_filename = "train_augmented.jsonl" if args.use_augmented and os.path.exists(os.path.join(args.data_dir, "train_augmented.jsonl")) else "train.jsonl"
    print(f"[Info] Training with dataset: {train_filename}")
    data_files = {
        "train": os.path.join(args.data_dir, train_filename),
        "validation": os.path.join(args.data_dir, "val.jsonl"),
        "test": os.path.join(args.data_dir, "test.jsonl"),
    }
    raw_datasets = load_dataset("json", data_files=data_files)
    print(f"[Info] Dataset splits: train={len(raw_datasets['train'])}, val={len(raw_datasets['validation'])}, test={len(raw_datasets['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.use_crf or args.loss_type in ("focal", "crf"):
        from openthai_ner.model_crf import OpenThaiNERWithCRF
        from openthai_ner.losses import compute_class_weights

        tag_counts = mapping_data.get("tag_counts", {})
        class_weights = compute_class_weights(tag_counts, label2id) if args.loss_type in ("focal", "ce") else None
        if class_weights is not None:
            class_weights = class_weights.to("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

        print(f"[Model] Initializing OpenThaiNERWithCRF (use_crf={args.use_crf}, loss_type={args.loss_type})...")
        model = OpenThaiNERWithCRF.from_backbone(
            args.model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            use_crf=args.use_crf,
            loss_type=args.loss_type,
            class_weights=class_weights,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    tokenized_datasets = raw_datasets.map(
        lambda x: tokenize_and_align_labels(
            x,
            tokenizer,
            label2id,
            max_length=args.max_length,
            label_all_subwords=args.label_all_subwords,
        ),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, pad_to_multiple_of=8)

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    import inspect
    valid_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    eval_arg = "eval_strategy" if "eval_strategy" in valid_params else "evaluation_strategy"
    t_args = {
        "output_dir": args.output_dir,
        eval_arg: "epoch",
        "save_strategy": "epoch",
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "logging_dir": f"{args.output_dir}/logs",
        "logging_steps": 50,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1" if METRIC_BACKEND else "accuracy",
        "greater_is_better": True,
        "fp16": False,
        "bf16": use_bf16,
        "max_grad_norm": 1.0,
        "dataloader_num_workers": 0,
        "report_to": "none",
    }
    if "warmup_ratio" in valid_params:
        t_args["warmup_ratio"] = args.warmup_ratio
    elif "warmup_steps" in valid_params:
        t_args["warmup_steps"] = 100

    # Filter out any unexpected parameter
    final_kwargs = {k: v for k, v in t_args.items() if k in valid_params}
    training_args = TrainingArguments(**final_kwargs)

    compute_metrics = build_compute_metrics(id2label)

    custom_optimizers = (None, None)
    if args.llrd:
        print(f"[LLRD] Initializing Layer-wise Learning Rate Decay (base_lr={args.learning_rate}, decay={args.llrd_decay})...")
        opt = create_optimizer_with_llrd(
            model,
            base_lr=args.learning_rate,
            decay_rate=args.llrd_decay,
            weight_decay=args.weight_decay,
        )
        custom_optimizers = (opt, None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=custom_optimizers,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n[Info] Starting model training on Colab GPU...")
    trainer.train()

    print("\n[Info] Running evaluation on test set...")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print("Test Results:")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    print(f"\n[Done] Training complete! Model and artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
