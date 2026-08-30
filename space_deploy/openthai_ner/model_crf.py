"""
Enhanced Transformer Model with CRF Layer and Focal Loss for Thai NER.
"""

from typing import Optional, List, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from .crf import CRF
from .losses import FocalLoss


class OpenThaiNERWithCRF(PreTrainedModel):
    """
    Token Classification Model with Transformer Backbone + Linear Classifier + CRF Layer.
    Guarantees syntactically valid BIO sequence predictions.
    """

    config_class = AutoConfig

    def __init__(
        self,
        config,
        use_crf: bool = True,
        loss_type: str = "crf",  # 'crf', 'focal', or 'ce'
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.use_crf = use_crf
        self.loss_type = loss_type

        # Backbone transformer (e.g. CamemBERT / RoBERTa)
        self.roberta = AutoModel.from_config(config)

        classifier_dropout = (
            config.classifier_dropout
            if getattr(config, "classifier_dropout", None) is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if self.use_crf:
            self.crf = CRF(num_tags=config.num_labels, batch_first=True)
            if hasattr(config, "id2label") and config.id2label:
                # Convert string keys to int if necessary
                id2label = {int(k): v for k, v in config.id2label.items()}
                self.crf.enforce_bio_constraints(id2label)

        if self.loss_type == "focal":
            self.loss_fct = FocalLoss(gamma=focal_gamma, alpha=class_weights, ignore_index=-100)
        elif self.loss_type == "ce":
            self.loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        else:
            self.loss_fct = None

        self.post_init()

    @classmethod
    def from_backbone(
        cls,
        model_name_or_path: str,
        num_labels: int,
        id2label: dict,
        label2id: dict,
        use_crf: bool = True,
        loss_type: str = "crf",
        class_weights: Optional[torch.Tensor] = None,
    ):
        """Instantiate model from pretrained transformer backbone weights."""
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        model = cls(config, use_crf=use_crf, loss_type=loss_type, class_weights=class_weights)
        pretrained_backbone = AutoModel.from_pretrained(model_name_or_path, config=config)
        model.roberta = pretrained_backbone
        return model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.use_crf and self.loss_type == "crf":
                # Valid mask is where labels != -100 and attention_mask == 1
                active_mask = (labels != -100) & (attention_mask == 1).bool()
                # Replace -100 with 0 for CRF tensor compatibility
                crf_labels = labels.clone()
                crf_labels[labels == -100] = 0
                loss = self.crf(logits, crf_labels, mask=active_mask.byte(), reduction="token_mean")
            elif self.loss_fct is not None:
                loss = self.loss_fct(logits, labels)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if return_dict else None,
            attentions=outputs.attentions if return_dict else None,
        )

    def decode(
        self,
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Decode emissions into predicted label sequence using CRF Viterbi algorithm."""
        if self.use_crf:
            if labels is not None:
                mask = (labels != -100) & (attention_mask == 1).bool()
            elif attention_mask is not None:
                mask = attention_mask.bool()
            else:
                mask = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)
            return self.crf.decode(logits, mask=mask.byte())
        else:
            return torch.argmax(logits, dim=-1).tolist()
