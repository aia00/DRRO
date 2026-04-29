#!/usr/bin/env python3
"""Export InfoRM latent means for prompt/response pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .data import PromptResponseDataset, build_response_collate
    from .modeling_inform_rm import InfoRMModel
except ImportError:
    from data import PromptResponseDataset, build_response_collate
    from modeling_inform_rm import InfoRMModel



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract InfoRM IB latent vectors from a JSONL file.")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_npy", required=True)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--response_key", default="response")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = InfoRMModel.from_pretrained(args.model_dir, device=device)
    dataset = PromptResponseDataset(Path(args.input_jsonl), prompt_key=args.prompt_key, response_key=args.response_key)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=build_response_collate(tokenizer, args.max_length),
    )

    latents: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.score_batch(batch, sample_latent=False)
            latents.append(outputs["mu"].detach().cpu().numpy())

    array = np.concatenate(latents, axis=0)
    output_path = Path(args.output_npy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)
    meta = {
        "input_jsonl": args.input_jsonl,
        "output_npy": args.output_npy,
        "num_rows": int(array.shape[0]),
        "latent_dim": int(array.shape[1]),
        "prompt_key": args.prompt_key,
        "response_key": args.response_key,
    }
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=True)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
