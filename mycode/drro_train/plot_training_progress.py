#!/usr/bin/env python3
"""Plot training-step progress curves from DRRO/PPO/ensemble logs."""

from __future__ import annotations

import argparse
import csv
import os
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reward or KL versus training step.")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more log.csv files.")
    parser.add_argument("--labels", nargs="+", required=True, help="Legend labels for --inputs.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    parser.add_argument(
        "--kind",
        choices=["gold", "proxy", "kl"],
        required=True,
        help="Series type to plot from each log.",
    )
    parser.add_argument(
        "--kl_key",
        default="kl_seq",
        help="KL column to use when --kind=kl (default: kl_seq).",
    )
    parser.add_argument("--max_step", type=int, default=None, help="Optional max training step to include.")
    parser.add_argument(
        "--no_center",
        action="store_true",
        help="Do not subtract the first logged value from each series.",
    )
    parser.add_argument(
        "--baseline_value",
        type=float,
        default=None,
        help="Optional shared baseline to subtract from every main curve.",
    )
    parser.add_argument(
        "--gold_scale",
        type=float,
        default=None,
        help="Optional divisor applied after baseline subtraction for gold scores.",
    )
    parser.add_argument("--colors", nargs="+", default=None, help="Optional colors for the main curves.")
    parser.add_argument(
        "--proxy_input",
        default=None,
        help="Optional extra log.csv for a dashed proxy reference line on gold plots.",
    )
    parser.add_argument("--proxy_label", default="Proxy", help="Legend label for --proxy_input.")
    parser.add_argument("--proxy_color", default="black", help="Color for the dashed proxy reference line.")
    parser.add_argument(
        "--proxy_baseline_value",
        type=float,
        default=None,
        help="Optional shared baseline to subtract from the proxy overlay line.",
    )
    parser.add_argument("--legend_loc", default="best", help="Legend location.")
    parser.add_argument("--figsize", nargs=2, type=float, default=[10.0, 7.0], help="Figure size.")
    return parser.parse_args()


def read_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def extract_series(rows: Sequence[dict], kind: str, max_step: int | None, kl_key: str) -> Tuple[np.ndarray, np.ndarray]:
    points: List[Tuple[int, float]] = []
    for row in rows:
        step_raw = row.get("step")
        if step_raw in (None, ""):
            continue
        step = int(float(step_raw))
        if max_step is not None and step > max_step:
            continue

        if kind == "gold":
            y_raw = row.get("gold_score")
        elif kind == "proxy":
            y_raw = row.get("proxy_score")
        else:
            y_raw = row.get(kl_key)
            if y_raw in (None, ""):
                y_raw = row.get("kl_seq") or row.get("kl")

        if y_raw in (None, ""):
            continue
        points.append((step, float(y_raw)))

    points.sort(key=lambda item: item[0])
    if not points:
        raise ValueError("No usable rows found in log.")
    steps = np.asarray([p[0] for p in points], dtype=float)
    values = np.asarray([p[1] for p in points], dtype=float)
    return steps, values


def transform_values(values: np.ndarray, center: bool, baseline_value: float | None) -> np.ndarray:
    out = values.copy()
    if baseline_value is not None:
        return out - baseline_value
    if center:
        return out - out[0]
    return out


def y_label(kind: str, centered: bool, baseline_value: float | None, gold_scale: float | None, kl_key: str) -> str:
    shifted = centered or baseline_value is not None
    if kind == "gold":
        if shifted and gold_scale is not None:
            return "Gold RM score"
        if shifted:
            return "Gold RM score"
        return "Gold RM score"
    if kind == "proxy":
        return "Proxy RM score"
    if shifted:
        return kl_key
    return kl_key


def plot_main_series(args: argparse.Namespace) -> None:
    if len(args.inputs) != len(args.labels):
        raise ValueError("--labels must match --inputs.")
    if args.colors is not None and len(args.colors) != len(args.inputs):
        raise ValueError("--colors must match --inputs when provided.")
    if args.kind != "gold" and args.proxy_input is not None:
        raise ValueError("--proxy_input is only supported for --kind=gold.")

    plt.figure(figsize=tuple(args.figsize))
    colors = args.colors or plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = [None] * len(args.inputs)

    center = not args.no_center

    for idx, (path, label) in enumerate(zip(args.inputs, args.labels)):
        rows = read_rows(path)
        steps, values = extract_series(rows, args.kind, args.max_step, args.kl_key)
        values = transform_values(values, center=center, baseline_value=args.baseline_value)
        if args.kind == "gold" and args.gold_scale is not None:
            values = values / args.gold_scale
        color = colors[idx] if idx < len(colors) else None
        plt.plot(steps, values, linewidth=2.5, color=color, label=label)

    if args.kind == "gold" and args.proxy_input is not None:
        rows = read_rows(args.proxy_input)
        steps, values = extract_series(rows, "proxy", args.max_step, args.kl_key)
        values = transform_values(values, center=center, baseline_value=args.proxy_baseline_value)
        plt.plot(
            steps,
            values,
            linewidth=2.8,
            linestyle="--",
            color=args.proxy_color,
            label=args.proxy_label,
        )

    plt.xlabel("Training step", fontsize=18)
    plt.ylabel(y_label(args.kind, center, args.baseline_value, args.gold_scale, args.kl_key), fontsize=18)
    if args.title:
        plt.title(args.title, fontsize=24, weight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc=args.legend_loc, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=180)
    plt.close()


if __name__ == "__main__":
    plot_main_series(parse_args())
