"""Lightweight InfoRM model adapted from the paper's variational IB reward model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


@dataclass
class InfoRMConfig:
    base_model_name: str
    latent_dim: int = 128
    beta: float = 0.01
    dropout: float = 0.0
    pooling: str = "cls"


class RewardMLP(nn.Module):
    def __init__(self, latent_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = max(latent_dim, 64)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class InfoRMModel(nn.Module):
    """Information bottleneck reward model for preference learning.

    The model follows the paper's structure:
    - transformer backbone -> pooled hidden state
    - encoder head -> mean and variance of latent IB representation
    - decoder MLP -> scalar reward
    """

    def __init__(self, config: InfoRMConfig) -> None:
        super().__init__()
        self.inform_config = config
        backbone_config = AutoConfig.from_pretrained(config.base_model_name)
        self.backbone = AutoModel.from_pretrained(config.base_model_name, config=backbone_config)
        hidden_size = int(getattr(self.backbone.config, "hidden_size"))
        self.encode_head = nn.Linear(hidden_size, config.latent_dim * 2)
        self.decode_head = RewardMLP(config.latent_dim, dropout=config.dropout)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _pool_hidden(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.inform_config.pooling == "cls":
            return last_hidden_state[:, 0]
        if self.inform_config.pooling == "mean":
            weights = attention_mask.unsqueeze(-1).float()
            return (last_hidden_state * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        last_index = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        return last_hidden_state[torch.arange(last_hidden_state.size(0), device=last_hidden_state.device), last_index]

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        sample_latent: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids
        outputs = self.backbone(**model_kwargs)
        pooled = self._pool_hidden(outputs.last_hidden_state, attention_mask)
        stats = self.encode_head(pooled)
        mu, raw_scale = stats.chunk(2, dim=-1)
        std = F.softplus(raw_scale) + 1e-6
        if sample_latent:
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return z, mu, std

    def reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        sample_latent: bool = False,
    ) -> Dict[str, torch.Tensor]:
        z, mu, std = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            sample_latent=sample_latent,
        )
        rewards = self.decode_head(z)
        return {"reward": rewards, "latent": z, "mu": mu, "std": std}

    def _kl_to_standard_normal(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        var = std.pow(2)
        return 0.5 * torch.sum(mu.pow(2) + var - torch.log(var + 1e-8) - 1.0, dim=-1)

    def forward_pair(
        self,
        chosen: Dict[str, torch.Tensor],
        rejected: Dict[str, torch.Tensor],
        sample_latent: bool = True,
    ) -> Dict[str, torch.Tensor]:
        chosen_out = self.reward(sample_latent=sample_latent, **chosen)
        rejected_out = self.reward(sample_latent=sample_latent, **rejected)
        bt_loss = -F.logsigmoid(chosen_out["reward"] - rejected_out["reward"]).mean()
        kl_chosen = self._kl_to_standard_normal(chosen_out["mu"], chosen_out["std"])
        kl_rejected = self._kl_to_standard_normal(rejected_out["mu"], rejected_out["std"])
        kl_loss = 0.5 * (kl_chosen.mean() + kl_rejected.mean())
        total_loss = bt_loss + self.inform_config.beta * kl_loss
        return {
            "loss": total_loss,
            "bt_loss": bt_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "chosen_reward": chosen_out["reward"].detach(),
            "rejected_reward": rejected_out["reward"].detach(),
        }

    def score_batch(
        self,
        batch: Dict[str, torch.Tensor],
        sample_latent: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self.reward(sample_latent=sample_latent, **batch)

    def save_pretrained(self, output_dir: str | Path, tokenizer: Optional[AutoTokenizer] = None) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_path / "inform_rm.pt")
        with (output_path / "inform_config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(self.inform_config), handle, indent=2, ensure_ascii=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_path)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple["InfoRMModel", AutoTokenizer]:
        model_path = Path(model_dir)
        config_path = model_path / "inform_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing InfoRM config: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            config = InfoRMConfig(**json.load(handle))
        model = cls(config)
        state_dict = torch.load(model_path / "inform_rm.pt", map_location="cpu")
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
