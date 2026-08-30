# OpenThai-NER

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-JonusNattapong%2FOpenThai--NER-blue)](https://huggingface.co/JonusNattapong/OpenThai-NER)
[![Dataset](https://img.shields.io/badge/%F0%9F%A7%A0%20Dataset-OpenThai--NER--Corpus-green)](https://huggingface.co/datasets/JonusNattapong/OpenThai-NER-Corpus)
[![License: CC BY 3.0](https://img.shields.io/badge/License-CC%20BY%203.0-lightgrey.svg)](https://creativecommons.org/licenses/by/3.0/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)

OpenThai-NER is a production-grade Named Entity Recognition (NER) library and model for the Thai language. It builds upon `Pavarissy/phayathaibert-thainer` and fine-tunes on the multi-domain `OpenThai-NER-Corpus` with subword span reconstruction, numerical stability fixes, an optional linear-chain CRF decoding layer, and INT8 ONNX export for low-latency CPU deployment.

---

## Key Features

- **Numerical Stability**: Eliminates training instability (`eval_loss: NaN`) by replacing mixed-precision `fp16` with `bf16`/`fp32`, applying gradient clipping (`max_grad_norm=1.0`), and strictly masking padding/special tokens (`-100`) in `DataCollatorForTokenClassification`.
- **Dataset Quality**: Normalized 8,601 valid samples across 407 domains. Resolved syntax typos (`DTAE` &rarr; `DATE`) and consolidated synonym tags (`LOC` &rarr; `LOCATION`, `ORG` &rarr; `ORGANIZATION`, `PER` &rarr; `PERSON`).
- **Subword Span Reconstruction**: Automatically aggregates SentencePiece subword fragments into complete Thai word spans with exact character start and end offsets.
- **Decoding Architecture**: Implements a pure PyTorch linear-chain CRF layer with Viterbi decoding and BIO transition constraints to prevent structural tag violations.
- **Class Imbalance Mitigation**: Implements Focal Loss with smoothed inverse class frequency weights to improve recall on rare entity categories.
- **Hardware Acceleration**: Provides dynamic INT8 quantization via ONNX Runtime, reducing latency by 3.8&times; on CPU environments.

---

## Benchmark & Evaluation

Evaluated on `data/test.jsonl` (1,092 test sequences across 407 domains) using `seqeval` strict span-level matching:

<p align="center">
  <img src="assets/benchmark.svg" alt="OpenThai-NER Benchmark Leaderboard" width="100%">
</p>

### Metric Summary

| Metric | OpenThai-NER (Final) | WangchanBERTa Base | PyThaiNLP ThaiNER-v2 | PhayaThaiBERT Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Strict Span F1** | **79.28%** | 78.10% | 76.40% | 71.20% |
| **Precision** | **78.87%** | 78.20% | 75.90% | 70.80% |
| **Recall** | **79.70%** | 78.00% | 76.90% | 71.60% |
| **Token Accuracy** | **90.76%** | 89.90% | 88.50% | 85.20% |
| **Validation Loss** | **0.3697** | 0.3850 | N/A | *NaN (Unstable)* |
| **CPU Latency (INT8)** | **11.2 ms/seq** | 18.5 ms/seq | 15.1 ms/seq | 42.6 ms/seq |

---

## Directory Structure

```text
OpenThai/
├── openthai_ner/                 # Core Python package
│   ├── __init__.py               # Package entrypoint
│   ├── pipeline.py               # Inference pipeline & span reconstruction
│   ├── crf.py                    # Linear-chain CRF with Viterbi decoding
│   ├── losses.py                 # Focal Loss & class weighting
│   ├── model_crf.py              # Combined Transformer + CRF model class
│   └── utils.py                  # Offset alignment & HTML rendering
├── scripts/
│   ├── clean_dataset.py          # Dataset cleaner & stratified split
│   ├── export_onnx.py            # ONNX export & INT8 quantization
│   ├── evaluate_benchmark.py     # Evaluation script (seqeval)
│   ├── benchmark_sota.py         # Multi-model comparative benchmark
│   └── build_package.py          # Packaging script for PyPI release
├── notebooks/
│   └── OpenThai_NER_FineTuning_Final.ipynb  # Google Colab GPU training notebook
├── space_deploy/                 # Standalone Hugging Face Space application
│   ├── app.py
│   ├── README.md
│   └── requirements.txt
├── data/                         # Processed data splits
│   ├── train.jsonl               # 6,792 training sequences
│   ├── val.jsonl                 # 717 validation sequences
│   ├── test.jsonl                # 1,092 test sequences
│   └── label_map.json            # Canonical 128-tag label mapping
├── train_ner.py                  # Self-contained training script
├── app.py                        # Local Gradio web demo
├── pyproject.toml                # Build configuration
└── README.md
```

---

## Installation

### From Source
```bash
git clone https://github.com/JonusNattapong/OpenThai.git
cd OpenThai
pip install -r requirements.txt
```

### Install as Package
```bash
pip install .
# Or directly via Git:
pip install git+https://github.com/JonusNattapong/OpenThai.git
```

---

## Quickstart

### Basic Inference

```python
from openthai_ner import OpenThaiNER

ner = OpenThaiNER("JonusNattapong/OpenThai-NER")

text = "นายสมชาย เข็มกลัด เดินทางไปประชุมที่กระทรวงการคลัง ถนนพระราม 6 ในวันที่ 15 มกราคม"
entities = ner.predict(text, threshold=0.5)

for ent in entities:
    print(f"[{ent['entity']}] '{ent['word']}' (Span: {ent['start']}:{ent['end']}, Score: {ent['score']:.4f})")
```

**Output:**
```text
[PERSON] 'นายสมชาย เข็มกลัด' (Span: 0:17, Score: 0.9812)
[ORGANIZATION] 'กระทรวงการคลัง' (Span: 36:50, Score: 0.9924)
[LOCATION] 'ถนนพระราม 6' (Span: 51:62, Score: 0.9540)
[DATE] 'วันที่ 15 มกราคม' (Span: 66:82, Score: 0.9715)
```

### HTML Rendering (Notebooks & Web)

```python
html_output = ner.render_html(text)
# In Jupyter Notebook:
# from IPython.display import HTML; display(HTML(html_output))
```

### Fast Inference with ONNX Runtime

```python
from openthai_ner import OpenThaiNER

ner_onnx = OpenThaiNER(
    "JonusNattapong/OpenThai-NER",
    onnx_path="models/onnx/openthai_ner_quantized.onnx"
)
results = ner_onnx.predict("ธนาคารแห่งประเทศไทย ประกาศปรับลดอัตราดอกเบี้ย")
```

---

## 🎯 OpenThai-ReRanker (Cross-Encoder for RAG)

`openthai_reranker` provides high-precision cross-encoder scoring for Retrieval-Augmented Generation (RAG) and semantic search pipelines.

### Reranking Candidates in RAG

```python
from openthai_reranker import OpenThaiReranker

reranker = OpenThaiReranker("airesearch/wangchanberta-base-att-spm-uncased")

query = "อาการสำคัญของโรคเบาหวานมีอะไรบ้าง"
documents = [
    "การเดินทางไปเกาะเสม็ดสามารถขึ้นเรือข้ามฟากได้ที่ท่าเรือบ้านเพ",
    "โรคเบาหวานเป็นภาวะที่มีน้ำตาลในเลือดสูง ผู้ป่วยมักปัสสาวะบ่อย กระหายน้ำ และน้ำหนักลดลงรวดเร็ว",
    "กรมสรรพากรกำหนดเวลายื่นภาษีเงินได้บุคคลธรรมดาภายในวันที่ 8 เมษายน",
    "อาการทั่วไปของเบาหวานชนิดที่ 2 คือแผลหายช้า ชาตามปลายมือปลายเท้า และอ่อนเพลีย"
]

# Rerank and extract top 2 most relevant passages
ranked = reranker.rerank(query, documents, top_k=2)

for item in ranked:
    print(f"Rank {item['rank']}: [Score {item['relevance_score']:.4f}] {item['snippet']}")
```

### Training Cross-Encoder Re-Ranker
```bash
python train_reranker.py \
  --model_name airesearch/wangchanberta-base-att-spm-uncased \
  --epochs 3 \
  --batch_size 16 \
  --output_dir models/openthai-reranker-final
```

---

## Training (NER)

The training script is self-contained and handles dataset downloading, subword alignment, and metric logging automatically.

### Run on Local GPU or Colab
```bash
python train_ner.py \
  --model_name Pavarissy/phayathaibert-thainer \
  --data_dir data \
  --output_dir models/openthai-ner-final \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5
```

### Advanced Options (CRF & Focal Loss)
```bash
# Train with Linear-Chain CRF Layer
python train_ner.py --use_crf --output_dir models/openthai-ner-crf

# Train with Focal Loss for class imbalance
python train_ner.py --loss_type focal --focal_gamma 2.0
```

### Google Colab Execution
Run directly via the Google Colab CLI:
```bash
colab run --gpu T4 train_ner.py
```
Or open [`notebooks/OpenThai_NER_FineTuning_Final.ipynb`](notebooks/OpenThai_NER_FineTuning_Final.ipynb) in Google Colab.

---

## Model Export & Optimization

### ONNX Export & INT8 Quantization
```bash
python scripts/export_onnx.py \
  --model JonusNattapong/OpenThai-NER \
  --output_dir models/onnx
```

### Multi-Model Benchmark
```bash
python scripts/benchmark_sota.py --test_file data/test.jsonl --samples 100
```

---

## Web Demo

Run the interactive Gradio demo locally:
```bash
python app.py
```
To deploy directly to Hugging Face Spaces, push the contents of [`space_deploy/`](space_deploy/) to your Space repository.

---

## Supported Entities

The canonical schema covers 128 BIO labels across the following core categories:
- **Agents:** `PERSON`, `ORGANIZATION`
- **Locations & Facilities:** `LOCATION`, `FACILITY`
- **Temporal & Quantities:** `DATE`, `TIME`, `MONEY`, `PERCENT`
- **Identifiers & Contact:** `ID`, `ACCOUNT`, `PHONE`, `EMAIL`, `URL`
- **Specialized Domains:** `LAW`, `PRODUCT`, `DISEASE`, `TECHNOLOGY`

---

## Citation

```bibtex
@software{openthai_ner2026,
  author = {Nattapong Tapachoom},
  title = {OpenThai-NER: Production-Ready Thai Named Entity Recognition},
  url = {https://github.com/JonusNattapong/OpenThai},
  version = {0.1.0},
  year = {2026}
}
```

## License

This project is released under the [Creative Commons Attribution 3.0 Unported (CC BY 3.0)](https://creativecommons.org/licenses/by/3.0/) license.
