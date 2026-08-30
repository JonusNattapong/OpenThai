"""
Export OpenThai-NER to ONNX and perform Dynamic INT8 Quantization.
Accelerates inference by 3-5x and halves model size for CPU and Edge deployment.
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def export_onnx(model_path: str, output_dir: str = "models/onnx"):
    os.makedirs(output_dir, exist_ok=True)
    onnx_model_path = os.path.join(output_dir, "openthai_ner.onnx")
    quantized_model_path = os.path.join(output_dir, "openthai_ner_quantized.onnx")

    print(f"[Info] Loading model and tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    # Create dummy input for tracing
    dummy_text = "กระทรวงการคลัง ถนนพระราม 6"
    dummy_inputs = tokenizer(dummy_text, return_tensors="pt")

    print(f"[Info] Exporting PyTorch model to ONNX: {onnx_model_path}...")
    torch.onnx.export(
        model,
        (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
        onnx_model_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"[Success] ONNX model exported: {onnx_model_path}")

    # Quantization to INT8
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        print(f"[Info] Applying dynamic INT8 quantization -> {quantized_model_path}...")
        quantize_dynamic(
            model_input=onnx_model_path,
            model_output=quantized_model_path,
            weight_type=QuantType.QInt8,
        )
        print(f"[Success] Quantized model saved: {quantized_model_path}")
    except ImportError:
        print("[Warning] onnxruntime.quantization not available. Skipping INT8 quantization.")


def main():
    parser = argparse.ArgumentParser(description="Export OpenThai-NER to ONNX")
    parser.add_argument("--model", default="JonusNattapong/OpenThai-NER", help="Model name or local path")
    parser.add_argument("--output_dir", default="models/onnx", help="Output directory")
    args = parser.parse_args()
    export_onnx(args.model, args.output_dir)


if __name__ == "__main__":
    main()
