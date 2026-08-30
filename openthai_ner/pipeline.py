"""
High-Level Production Inference Pipeline for OpenThai-NER.
Supports PyTorch (CPU/CUDA) and ONNX Runtime execution with subword span aggregation.
"""

from typing import Union, List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from .utils import reconstruct_spans, render_html_highlight


class OpenThaiNER:
    """
    High-level interface for Thai Named Entity Recognition.
    
    Example:
        >>> ner = OpenThaiNER()
        >>> results = ner.predict("นายสมชายทำงานที่กระทรวงการคลัง")
        >>> print(results)
    """

    def __init__(
        self,
        model_name_or_path: str = "JonusNattapong/OpenThai-NER",
        device: str = None,
        onnx_path: str = None,
    ):
        self.model_name = model_name_or_path
        self.onnx_path = onnx_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if onnx_path:
            import onnxruntime as ort
            self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            # Need config for id2label
            self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
            self.id2label = self.model.config.id2label
            self.device = "cpu"
            self.use_onnx = True
        else:
            self.use_onnx = False
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
            self.model.to(self.device)
            self.model.eval()
            self.id2label = self.model.config.id2label

    def predict(
        self,
        text: Union[str, List[str]],
        threshold: float = 0.0,
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        Extract named entities with character offsets and confidence scores.

        Args:
            text: Input text string or list of text strings.
            threshold: Minimum confidence score to include an entity.

        Returns:
            List of entity dictionaries, or list of lists for batch input.
        """
        is_single = isinstance(text, str)
        inputs = [text] if is_single else text

        all_results = []
        for item in inputs:
            spans = self._infer_single(item)
            if threshold > 0:
                spans = [s for s in spans if s["score"] >= threshold]
            all_results.append(spans)

        return all_results[0] if is_single else all_results

    def _infer_single(self, text: str) -> List[Dict[str, Any]]:
        """Run inference on a single Thai text string."""
        if not text or not text.strip():
            return []

        encodings = self.tokenizer(
            text,
            return_tensors="pt" if not self.use_onnx else "np",
            truncation=True,
            max_length=512,
            return_offsets_mapping=False,
        )

        tokens = self.tokenizer.convert_ids_to_tokens(
            encodings["input_ids"][0].tolist() if hasattr(encodings["input_ids"][0], "tolist") else encodings["input_ids"][0]
        )

        if self.use_onnx:
            ort_inputs = {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
            }
            ort_outputs = self.session.run(None, ort_inputs)
            logits = torch.tensor(ort_outputs[0])
        else:
            enc_cuda = {k: v.to(self.device) for k, v in encodings.items()}
            with torch.no_grad():
                outputs = self.model(**enc_cuda)
            logits = outputs.logits.cpu()

        probabilities = torch.softmax(logits, dim=-1)[0]
        pred_ids = torch.argmax(probabilities, dim=-1).tolist()
        pred_scores = [probabilities[i, pid].item() for i, pid in enumerate(pred_ids)]

        # Map prediction IDs to BIO tag strings
        predictions = [self.id2label.get(pid, self.id2label.get(str(pid), "O")) for pid in pred_ids]

        # Reconstruct into character spans
        return reconstruct_spans(text, tokens, predictions, pred_scores)

    def render_html(self, text: str) -> str:
        """Extract entities and return an interactive HTML view."""
        entities = self.predict(text)
        return render_html_highlight(text, entities)
