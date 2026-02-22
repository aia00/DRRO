#!/usr/bin/env python3
"""Plot KL vs reward curves from training logs."""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot over-optimization curves.")
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="One or more log.csv files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="overopt_kl_curve.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--x_key",
        type=str,
        default="kl_seq",
        help="Column to use for x-axis (default: kl_seq; falls back to kl if missing).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Optional run labels for inputs (must match number of --inputs).",
    )
    parser.add_argument(
        "--group_by",
        type=str,
        default="run",
        choices=["run", "delta"],
        help="How to group points into curves. Default: run (one curve pair per input).",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=None,
        help="Optional max step to include (e.g., 300).",
    )
    return parser.parse_args()


def read_log(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def parse_delta_from_path(path: str) -> Optional[float]:
    base = os.path.basename(path)
    if "grpo" in path:
        return 0.0
    match = re.search(r"delta([0-9]+(?:\\.[0-9]+)?)", path)
    if match:
        return float(match.group(1))
    match = re.search(r"delta([0-9]+(?:\\.[0-9]+)?)", base)
    if match:
        return float(match.group(1))
    return None


def infer_run_label(path: str) -> str:
    parent = os.path.basename(os.path.dirname(path))
    if parent:
        return parent
    return os.path.splitext(os.path.basename(path))[0]


def collect_series(
    inputs: List[str],
    x_key: str,
    labels: Optional[List[str]],
    group_by: str,
    max_step: Optional[int],
) -> Dict[str, Dict[str, List[float]]]:
    data: Dict[str, Dict[str, List[float]]] = {}
    for idx, path in enumerate(inputs):
        run_label = labels[idx] if labels is not None else infer_run_label(path)
        file_delta = parse_delta_from_path(path)
        rows = read_log(path)
        for row in rows:
            step_raw = row.get("step")
            if max_step is not None and step_raw not in (None, ""):
                if int(float(step_raw)) > max_step:
                    continue
            if file_delta is not None:
                delta = file_delta
            else:
                delta_raw = row.get("delta", 0.0)
                if delta_raw in (None, ""):
                    delta_raw = 0.0
                delta = float(delta_raw)

            if group_by == "delta":
                key = f"delta={delta:g}"
            else:
                key = run_label

            if key not in data:
                data[key] = {"kl": [], "proxy": [], "gold": []}

            kl_raw = row.get(x_key)
            if kl_raw is None or kl_raw == "":
                kl_raw = row.get("kl_seq") or row.get("kl", 0.0)
            kl = float(kl_raw)
            proxy_raw = row.get("proxy_score_norm", row.get("proxy_score", 0.0))
            gold_raw = row.get("gold_score_norm", row.get("gold_score", 0.0))
            if proxy_raw in ("", None):
                proxy_raw = 0.0
            if gold_raw in ("", None):
                gold_raw = 0.0
            proxy = float(proxy_raw)
            gold = float(gold_raw)
            data[key]["kl"].append(kl)
            data[key]["proxy"].append(proxy)
            data[key]["gold"].append(gold)
    return data


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.inputs):
        raise ValueError("--labels count must match --inputs count.")

    series = collect_series(
        args.inputs,
        args.x_key,
        args.labels,
        args.group_by,
        args.max_step,
    )

    if not series:
        raise ValueError("No data found in inputs.")

    keys = list(series.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(keys), 1)))

    plt.figure(figsize=(7, 5))
    for color, key in zip(colors, keys):
        kl_vals = series[key]["kl"]
        proxy_vals = series[key]["proxy"]
        gold_vals = series[key]["gold"]
        order = np.argsort(kl_vals)
        kl_sorted = [kl_vals[i] for i in order]
        proxy_sorted = [proxy_vals[i] for i in order]
        gold_sorted = [gold_vals[i] for i in order]
        plt.plot(
            kl_sorted,
            gold_sorted,
            color=color,
            linestyle="-",
            label=f"{key} gold",
        )
        plt.plot(
            kl_sorted,
            proxy_sorted,
            color=color,
            linestyle="--",
            label=f"{key} proxy",
        )

    xlabel = "KL (policy || reference)"
    if args.x_key != "kl":
        xlabel = f"{args.x_key}"
    plt.xlabel(xlabel)
    plt.ylabel("Normalized reward")
    plt.title("RLHF over-optimization curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
