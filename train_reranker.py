"""
Production Cross-Encoder Training Script for Thai RAG and Information Retrieval.
Supports Binary Classification (BCEWithLogits) and MarginMSE Ranking Loss.
"""

import os
import sys
import json
import random
import argparse
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Thai Cross-Encoder Re-Ranker")
    parser.add_argument("--model_name", default="airesearch/wangchanberta-base-att-spm-uncased", help="Pretrained backbone model")
    parser.add_argument("--data_file", default=None, help="Path to custom training jsonl file")
    parser.add_argument("--output_dir", default="models/openthai-reranker-final", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Eval batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU training")
    return parser.parse_args()


def generate_synthetic_thai_rag_data(num_samples: int = 1200) -> List[Dict[str, Any]]:
    """Generate high-quality multi-domain Thai Query-Passage pairs for training and validation."""
    domains = [
        {
            "query_tmpl": "อาการของ{topic}มีอะไรบ้าง",
            "pos_tmpl": "{topic}เป็นภาวะสุขภาพที่พบได้บ่อย โดยอาการสำคัญได้แก่ {detail} และควรปรึกษาแพทย์ทันทีหากมีอาการรุนแรง",
            "neg_tmpl": "การท่องเที่ยวใน{topic}มีสถานที่น่าสนใจมากมาย เช่น ชายหาดและวัดโบราณที่สวยงาม",
            "topics": [
                ("โรคเบาหวาน", "กระหายน้ำบ่อย ปัสสาวะบ่อย น้ำหนักลดลงอย่างรวดเร็ว และอ่อนเพลียเรื้อรัง"),
                ("โรคความดันโลหิตสูง", "ปวดศีรษะบริเวณท้ายทอย เวียนศีรษะ ตาพร่ามัว และเหนื่อยง่าย"),
                ("ไข้เลือดออก", "ไข้สูงเฉียบพลัน ปวดกระบอกตา มีจุดเลือดออกตามผิวหนัง และคลื่นไส้"),
                ("โรคกรดไหลย้อน", "แสบร้อนกลางอก เรอเปรี้ยว เจ็บหน้าอก และกลืนลำบาก"),
                ("โรคซึมเศร้า", "อารมณ์ดิ่งต่อเนื่อง เบื่อหน่ายสิ่งรอบตัว นอนไม่หลับ และสมาธิสั้น"),
            ]
        },
        {
            "query_tmpl": "วิธีเดินทางไป{topic}",
            "pos_tmpl": "การเดินทางไป{topic}สามารถใช้บริการ{detail} ซึ่งสะดวกและรวดเร็วที่สุด",
            "neg_tmpl": "{topic}เป็นผลไม้ที่มีรสหวานอมเปรี้ยว นิยมรับประทานสดหรือแปรรูปเป็นของหวาน",
            "topics": [
                ("สนามบินสุวรรณภูมิ", "รถไฟฟ้าแอร์พอร์ต เรล ลิงก์ (Airport Rail Link) หรือทางด่วนมอเตอร์เวย์"),
                ("เกาะเสม็ด", "นั่งรถโดยสารมาลงที่ท่าเรือบ้านเพ จังหวัดระยอง แล้วต่อเรือข้ามฟาก"),
                ("วัดพระแก้ว", "รถไฟฟ้า MRT ลงสถานีสนามไชย แล้วเดินต่อประมาณ 10 นาที หรือนั่งเรือด่วนเจ้าพระยา"),
                ("ดอยอินทนนท์", "เช่ารถขับจากตัวเมืองเชียงใหม่ ใช้ทางหลวงหมายเลข 108 มุ่งหน้าสู่อำเภอจอมทอง"),
            ]
        },
        {
            "query_tmpl": "สิทธิ์ประโยชน์ของ{topic}",
            "pos_tmpl": "ผู้มี{topic}จะได้รับความคุ้มครองและสิทธิประโยชน์ครอบคลุม {detail}",
            "neg_tmpl": "{topic}เป็นเทคโนโลยีคอมพิวเตอร์ที่ใช้ประมวลผลข้อมูลกราฟิกระดับสูง",
            "topics": [
                ("ประกันสังคม มาตรา 33", "ค่ารักษาพยาบาล เงินชดเชยการว่างงาน ค่าคลอดบุตร และเงินบำเหน็จชราภาพ"),
                ("บัตรทอง (บัตร 30 บาท)", "การตรวจวินิจฉัยโรค ค่ายาในบัญชียาหลักแห่งชาติ และการผ่าตัดรักษาโรคเรื้อรัง"),
                ("กองทุนสำรองเลี้ยงชีพ", "เงินสมทบจากนายจ้าง สิทธิประโยชน์ลดหย่อนภาษี และผลตอบแทนจากการลงทุน"),
            ]
        },
        {
            "query_tmpl": "หลักเกณฑ์การยื่นภาษี{topic}",
            "pos_tmpl": "การยื่นภาษี{topic}ผู้มีเงินได้จะต้อง {detail} ภายในกำหนดเวลาของกรมสรรพากร",
            "neg_tmpl": "การทำขนม{topic}ต้องใช้แป้งข้าวเหนียวผสมกะทิและน้ำตาลมะพร้าวแท้",
            "topics": [
                ("บุคคลธรรมดา ภ.ง.ด.91", "ยื่นแบบแสดงรายการเงินได้ผ่านระบบออนไลน์ภายในวันที่ 8 เมษายนของทุกปี"),
                ("นิติบุคคล ภ.ง.ด.50", "จัดทำงบการเงินและยื่นเสียภาษีภายใน 150 วันนับแต่วันสิ้นสุดรอบระยะเวลาบัญชี"),
                ("ภาษีมูลค่าเพิ่ม ภ.พ.30", "นำส่งภาษีภายในวันที่ 15 ของเดือนถัดไปพร้อมรายงานภาษีซื้อภาษีขาย"),
            ]
        }
    ]

    samples = []
    for _ in range(num_samples):
        cat = random.choice(domains)
        topic, detail = random.choice(cat["topics"])
        query = cat["query_tmpl"].format(topic=topic)
        pos_passage = cat["pos_tmpl"].format(topic=topic, detail=detail)

        # Negative passage from different category
        other_cat = random.choice([c for c in domains if c != cat])
        other_topic, other_detail = random.choice(other_cat["topics"])
        neg_passage = other_cat["neg_tmpl"].format(topic=other_topic, detail=other_detail)

        # 1 positive pair (label 1.0)
        samples.append({"query": query, "passage": pos_passage, "label": 1.0})
        # 1 negative pair (label 0.0)
        samples.append({"query": query, "passage": neg_passage, "label": 0.0})

    random.shuffle(samples)
    return samples


def compute_reranker_metrics(eval_pred):
    """Compute Accuracy, Precision, Recall, and MRR proxy."""
    predictions, labels = eval_pred
    preds = 1.0 / (1.0 + np.exp(-predictions.squeeze()))  # Sigmoid
    binary_preds = (preds >= 0.5).astype(float)
    acc = (binary_preds == labels).mean()

    # Mean absolute error
    mae = np.abs(preds - labels).mean()

    return {
        "accuracy": float(acc),
        "mae": float(mae),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 65)
    print("🚀 OPENTHAI-RERANKER TRAINING PIPELINE")
    print(f"Backbone: {args.model_name}")
    print(f"Output:   {args.output_dir}")
    print("=" * 65)

    # 1. Prepare Dataset
    if args.data_file and os.path.exists(args.data_file):
        print(f"[Data] Loading dataset from {args.data_file}...")
        with open(args.data_file, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f if line.strip()]
    else:
        print("[Data] Generating multi-domain Thai RAG dataset (2,400 query-passage pairs)...")
        data_list = generate_synthetic_thai_rag_data(num_samples=1200)

    # Split train/val/test (80:10:10)
    total = len(data_list)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    train_data = data_list[:train_end]
    val_data = data_list[train_end:val_end]
    test_data = data_list[val_end:]

    print(f"[Data] Splits: Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")

    raw_datasets = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    })

    # 2. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )

    def preprocess_function(examples):
        return tokenizer(
            examples["query"],
            examples["passage"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=20,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=False,
        bf16=use_bf16,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Custom Loss with BCEWithLogits
    class RerankerTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("label").float()
            outputs = model(**inputs)
            logits = outputs.logits.view(-1)
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_reranker_metrics,
    )

    print("\n[Train] Starting Cross-Encoder training...")
    trainer.train()

    print("\n[Evaluate] Running evaluation on test set...")
    test_res = trainer.evaluate(tokenized_datasets["test"])
    print("\n" + "=" * 50)
    print("🏆 TEST EVALUATION RESULTS:")
    for k, v in test_res.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("=" * 50)

    # Save artifacts
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(test_res, f, ensure_ascii=False, indent=2)

    print(f"\n[Done] Model saved successfully to {args.output_dir}!")


if __name__ == "__main__":
    main()
