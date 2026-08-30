"""
Upload Model Card, Benchmark Assets, and Configuration to Hugging Face Hub.
"""

import os
from huggingface_hub import HfApi, HfFolder

REPO_ID = "JonusNattapong/OpenThai-NER"
HF_TOKEN = os.environ.get("HF_TOKEN") or HfFolder.get_token()

api = HfApi(token=HF_TOKEN)

# 1. Prepare Model Card with HF Frontmatter
with open("README.md", "r", encoding="utf-8") as f:
    readme_body = f.read()

hf_frontmatter = """---
language:
- th
license: cc-by-3.0
pipeline_tag: token-classification
tags:
- ner
- thai
- token-classification
- transformers
- pytorch
- phayathaibert
- phayathaibert-thainer
base_model: Pavarissy/phayathaibert-thainer
datasets:
- JonusNattapong/OpenThai-NER-Corpus
metrics:
- name: precision
  type: precision
  value: 0.7887
- name: recall
  type: recall
  value: 0.7970
- name: f1
  type: f1
  value: 0.7928
- name: accuracy
  type: accuracy
  value: 0.9076
model-index:
- name: OpenThai-NER
  results:
  - task:
      type: token-classification
      name: Named Entity Recognition
    dataset:
      name: OpenThai-NER-Corpus
      type: JonusNattapong/OpenThai-NER-Corpus
    metrics:
    - name: Strict Span F1
      type: f1
      value: 0.7928
    - name: Precision
      type: precision
      value: 0.7887
    - name: Recall
      type: recall
      value: 0.7970
    - name: Token Accuracy
      type: accuracy
      value: 0.9076
---
"""

hf_readme = hf_frontmatter + readme_body

temp_readme_path = "models/openthai-ner-final/HF_README.md"
with open(temp_readme_path, "w", encoding="utf-8") as f:
    f.write(hf_readme)

print(f"[Upload] Uploading assets/benchmark.svg to {REPO_ID}...")
api.upload_file(
    path_or_fileobj="assets/benchmark.svg",
    path_in_repo="assets/benchmark.svg",
    repo_id=REPO_ID,
    repo_type="model",
)
print("[Upload] assets/benchmark.svg uploaded successfully.")

print(f"[Upload] Uploading Model Card README.md to {REPO_ID}...")
api.upload_file(
    path_or_fileobj=temp_readme_path,
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
)
print("[Upload] README.md uploaded successfully.")

print(f"[Upload] Uploading test_results.json to {REPO_ID}...")
api.upload_file(
    path_or_fileobj="models/openthai-ner-final/test_results.json",
    path_in_repo="test_results.json",
    repo_id=REPO_ID,
    repo_type="model",
)
print("[Upload] test_results.json uploaded successfully.")

print("\n[Done] All updates pushed to Hugging Face successfully!")
