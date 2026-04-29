"""ScoreLM model used by BSPO.

This is a lightweight adaptation of the paper's ScoreLM idea:
- keep the causal LM head to model behavior / next-token distribution
- add a scalar score head for reward modeling
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ScoreLMConfig:
    base_model_name: str
    score_head_dropout: float = 0.0


class ScoreLMModel(nn.Module):
    def __init__(self, config: ScoreLMConfig) -> None:
        super().__init__()
        self.scorelm_config = config
        self.backbone = AutoModelForCausalLM.from_pretrained(config.base_model_name)
        hidden_size = int(getattr(self.backbone.config, "hidden_size"))
        self.dropout = nn.Dropout(config.score_head_dropout)
        self.score_head = nn.Linear(hidden_size, 1)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def gradient_checkpointing_enable(self) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        if hasattr(self.backbone.config, "use_cache"):
            self.backbone.config.use_cache = False

    def forward_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_token_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del response_token_mask
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        last_index = attention_mask.long().sum(dim=-1).clamp(min=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_index]
        scores = self.score_head(self.dropout(pooled)).squeeze(-1)
        result = {
            "score": scores,
            "logits": outputs.logits,
            "hidden": hidden,
        }
        if labels is not None and outputs.loss is not None:
            result["lm_loss"] = outputs.loss
        return result

    def behavior_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        target_mask = response_token_mask[:, 1:].bool()
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        action_log_probs = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        full_log_probs = torch.zeros_like(input_ids, dtype=torch.float32)
        full_token_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        full_log_probs[:, 1:] = action_log_probs * target_mask.float()
        full_token_mask[:, 1:] = target_mask.float()
        return full_log_probs, full_token_mask

    def save_pretrained(self, output_dir: str | Path, tokenizer: Optional[AutoTokenizer] = None) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_path / "scorelm.pt")
        with (output_path / "scorelm_config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(self.scorelm_config), handle, indent=2, ensure_ascii=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_path)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: Optional[str | torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple["ScoreLMModel", AutoTokenizer]:
        model_path = Path(model_dir)
        with (model_path / "scorelm_config.json").open("r", encoding="utf-8") as handle:
            cfg = ScoreLMConfig(**json.load(handle))
        model = cls(cfg)
        state_dict = torch.load(model_path / "scorelm.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device)
        model.eval()
        return model, tokenizer
