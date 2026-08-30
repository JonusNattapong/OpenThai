"""
Export Cross-Encoder Model to ONNX and perform Dynamic INT8 Quantization.
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    raise ImportError("Please install onnx and onnxruntime: pip install onnx onnxruntime")


def export_reranker_onnx(model_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    onnx_fp32_path = os.path.join(output_dir, "openthai_reranker_fp32.onnx")
    onnx_int8_path = os.path.join(output_dir, "openthai_reranker_quantized.onnx")

    print(f"[1/3] Loading PyTorch model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
    model.eval()

    print("[2/3] Exporting to ONNX (FP32)...")
    dummy_query = "อาการของโรคเบาหวาน"
    dummy_doc = "โรคเบาหวานเป็นภาวะที่มีน้ำตาลในเลือดสูง"
    dummy_inputs = tokenizer(
        dummy_query,
        dummy_doc,
        return_tensors="pt",
        padding="max_length",
        max_length=64,
    )

    torch.onnx.export(
        model,
        (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
        onnx_fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  Saved FP32 ONNX to {onnx_fp32_path}")

    print("[3/3] Performing Dynamic INT8 Quantization...")
    quantize_dynamic(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_path,
        weight_type=QuantType.QInt8,
    )
    print(f"  Saved INT8 Quantized ONNX to {onnx_int8_path}")

    fp32_size = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
    int8_size = os.path.getsize(onnx_int8_path) / (1024 * 1024)
    print(f"\n[Result] FP32 Size: {fp32_size:.1f} MB | INT8 Size: {int8_size:.1f} MB (Compression: {(1 - int8_size/fp32_size)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="airesearch/wangchanberta-base-att-spm-uncased")
    parser.add_argument("--output_dir", default="models/onnx")
    args = parser.parse_args()

    export_reranker_onnx(args.model, args.output_dir)
