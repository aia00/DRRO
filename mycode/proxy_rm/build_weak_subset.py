#!/usr/bin/env python3
"""Sample a smaller subset of preference pairs and inject label noise."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a weak/noisy subset from existing pairs.")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_size", type=int, default=1800)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--flip_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def flip_pairs(rows: List[Dict[str, object]], flip_ratio: float, rng: random.Random) -> int:
    n_flip = int(len(rows) * flip_ratio)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    flipped = 0
    for idx in indices[:n_flip]:
        row = rows[idx]
        row["chosen"], row["rejected"] = row["rejected"], row["chosen"]
        if "chosen_score" in row and "rejected_score" in row:
            row["chosen_score"], row["rejected_score"] = row["rejected_score"], row["chosen_score"]
        flipped += 1
    return flipped


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    train_rows = load_jsonl(Path(args.train_jsonl))
    val_rows = load_jsonl(Path(args.val_jsonl))

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    train_subset = train_rows[: args.train_size]
    val_subset = val_rows[: args.val_size]

    flipped = flip_pairs(train_subset, args.flip_ratio, rng)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(out_dir / "train.jsonl", train_subset)
    save_jsonl(out_dir / "val.jsonl", val_subset)

    meta = {
        "train_size": len(train_subset),
        "val_size": len(val_subset),
        "flip_ratio": args.flip_ratio,
        "flipped": flipped,
        "seed": args.seed,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {len(train_subset)} train / {len(val_subset)} val to {out_dir}, flipped={flipped}")


if __name__ == "__main__":
    main()
