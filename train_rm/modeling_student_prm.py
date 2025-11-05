#!/usr/bin/env python3
"""Hugging Face compatible StudentPRM model.

Provides StudentPRMConfig + StudentPRM so that after training you can
push to the hub and later load with:

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

The 2-logit PRM head is included. The model pools the final hidden state
at the last occurrence of the configured pool token (e.g. </think>)."""
from typing import Optional
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput


def last_token_index(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """return position of last token in input_ids which is not padding"""
    seq_length = input_ids.size(1)
    lengths = attention_mask.sum(dim=1)  # shape (batch_size,)
    last_indices = lengths - 1  # shape (batch_size,)
    return last_indices


class StudentPRMConfig(PretrainedConfig):
    model_type = "student_prm"

    def __init__(
        self,
        base_model_name: str,
        pool_token: str,
        hidden_size: int,
        vocab_size: int,
        num_labels: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.pool_token = pool_token
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.auto_map = {"AutoModel": "modeling_student_prm.StudentPRM", "AutoConfig": "modeling_student_prm.StudentPRMConfig"}
        self.architectures = ["StudentPRM"]
        # Prevent default-diff logic from re-instantiating without required args
        self.has_no_defaults_at_init = True

    def _get_non_default_generation_parameters(self):
        # Classification model; no generation params to diff
        # Returning empty dict prevents transformers from calling self.__class__() with missing args
        return {}

    def to_diff_dict(self):  # override to bypass default instantiation logic
        return self.to_dict()


class StudentPRM(PreTrainedModel):
    config_class = StudentPRMConfig

    def __init__(self, config: StudentPRMConfig, base: Optional[PreTrainedModel] = None, tokenizer=None):
        super().__init__(config)
        if base is None:
            # Load base model; rely on remote code if needed
            base = AutoModel.from_pretrained(config.base_model_name, trust_remote_code=True)
        # Resize embeddings if vocab changed due to added special tokens
        try:
            current_vocab = base.get_input_embeddings().weight.shape[0]
            if current_vocab != config.vocab_size:
                base.resize_token_embeddings(config.vocab_size)
        except Exception:
            pass
        self.base = base
        self.head = nn.Linear(config.hidden_size, config.num_labels)
        self.tokenizer = tokenizer  # optional, only needed for pool id resolution when provided
        if tokenizer is not None and config.pool_token in tokenizer.get_vocab():
            self.pool_id = tokenizer.convert_tokens_to_ids(config.pool_token)
        else:
            # Will be resolved later if tokenizer added special token dynamically
            self.pool_id = None
        self.post_init()

    def _resolve_pool_id(self, input_ids: torch.Tensor):
        if self.pool_id is None and self.tokenizer is not None:
            self.pool_id = self.tokenizer.convert_tokens_to_ids(self.config.pool_token)
        if self.pool_id is None:
            raise ValueError("pool_id not set and tokenizer unavailable to resolve it.")
        return self.pool_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        pool_id = self._resolve_pool_id(input_ids)
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = out.hidden_states[-1]
        pos = last_token_index(input_ids, attention_mask)
        pooled = hidden[torch.arange(len(input_ids), device=input_ids.device), pos]
        # Match head weight dtype for safety (bfloat16 training etc.)
        if pooled.dtype != self.head.weight.dtype:
            pooled = pooled.to(self.head.weight.dtype)
        logits = self.head(pooled)
        return SequenceClassifierOutput(logits=logits)

    def save_pretrained(self, save_directory: str, *args, **kwargs):
        if not getattr(self.config, "auto_map", None):
            self.config.auto_map = {"AutoModel": "modeling_student_prm.StudentPRM", "AutoConfig": "modeling_student_prm.StudentPRMConfig"}
        if not getattr(self.config, "architectures", None):
            self.config.architectures = ["StudentPRM"]
        super().save_pretrained(save_directory, *args, **kwargs)
