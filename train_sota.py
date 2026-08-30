import sys
import subprocess
import os
import zipfile

os.chdir("/content")
sys.path.insert(0, "/content")

# Extract and normalize bundle.zip (handling Windows backslashes on Linux)
if os.path.exists("/content/bundle.zip"):
    print("[Colab Setup] Extracting and normalizing /content/bundle.zip...")
    with zipfile.ZipFile("/content/bundle.zip", "r") as z:
        for file_info in z.infolist():
            normalized_path = os.path.join("/content", file_info.filename.replace("\\", "/"))
            if file_info.is_dir() or normalized_path.endswith("/"):
                os.makedirs(normalized_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(normalized_path), exist_ok=True)
                with z.open(file_info) as src, open(normalized_path, "wb") as dst:
                    dst.write(src.read())
    print("[Colab Setup] Extracted and normalized bundle.zip successfully!")

# Auto-install necessary dependencies on remote VM
print("[Colab Setup] Installing dependencies...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.36.0",
    "datasets>=2.14.0",
    "seqeval>=1.2.2",
    "evaluate>=0.4.0",
    "sentencepiece>=0.1.99",
    "accelerate>=0.25.0",
    "huggingface_hub>=0.20.0"
], check=True)

# Verify directories exist
assert os.path.exists("/content/data/label_map.json"), "data/label_map.json missing"
assert os.path.exists("/content/openthai_ner/__init__.py"), "openthai_ner package missing"

# Execute full SOTA training
print("[Colab Runner] Launching Ultimate Peak NER Training with CRF + Focal Loss + LLRD...")
cmd = [
    sys.executable, "train_ner.py",
    "--model_name", "Pavarissy/phayathaibert-thainer",
    "--data_dir", "data",
    "--output_dir", "models/openthai-ner-sota",
    "--epochs", "3",
    "--batch_size", "16",
    "--learning_rate", "3e-5",
    "--use_crf",
    "--loss_type", "focal",
    "--llrd",
    "--use_augmented"
]

print("Executing:", " ".join(cmd))
subprocess.run(cmd, check=True)
print("[Colab Runner] Ultimate SOTA Training Completed Successfully!")
