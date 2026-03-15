#!/usr/bin/env python3
"""Estimate proxy-point thresholds from eval logs using gold-peak rule."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate component thresholds at gold-score peak.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input CSV log files.")
    parser.add_argument("--output_theta_json", type=str, required=True)
    parser.add_argument("--output_diag_csv", type=str, default="")
    parser.add_argument("--x_key", type=str, default="kl_seq")
    parser.add_argument("--gold_key", type=str, default="gold_score")
    parser.add_argument(
        "--component_keys",
        type=str,
        default="",
        help="Comma-separated component columns (default: auto-detect component_*).",
    )
    parser.add_argument("--smooth_window", type=int, default=1, help="Moving-average window for gold score.")
    return parser.parse_args()


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    half = window // 2
    out: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        out.append(sum(values[start:end]) / max(end - start, 1))
    return out


def normalize_component_key(col: str) -> str:
    return col[len("component_") :] if col.startswith("component_") else col


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, float]] = []

    component_keys = [key.strip() for key in args.component_keys.split(",") if key.strip()]

    for input_path in args.inputs:
        with open(input_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                x_val = parse_float(raw.get(args.x_key))
                gold_val = parse_float(raw.get(args.gold_key))
                if x_val is None or gold_val is None:
                    continue

                if not component_keys:
                    component_keys = [key for key in raw.keys() if key.startswith("component_")]
                if not component_keys:
                    raise ValueError("No component keys found. Provide --component_keys or include component_* columns.")

                row: Dict[str, float] = {
                    args.x_key: x_val,
                    args.gold_key: gold_val,
                }
                valid = True
                for key in component_keys:
                    val = parse_float(raw.get(key))
                    if val is None:
                        valid = False
                        break
                    row[key] = val
                if valid:
                    rows.append(row)

    if not rows:
        raise ValueError("No valid rows found in input CSV files.")

    rows.sort(key=lambda item: item[args.x_key])
    gold_values = [row[args.gold_key] for row in rows]
    smoothed_gold = moving_average(gold_values, max(args.smooth_window, 1))

    peak_idx = max(range(len(rows)), key=lambda idx: smoothed_gold[idx])
    peak_row = rows[peak_idx]

    theta = {normalize_component_key(key): float(peak_row[key]) for key in component_keys}

    output_theta_path = Path(args.output_theta_json)
    output_theta_path.parent.mkdir(parents=True, exist_ok=True)
    output_theta_path.write_text(json.dumps(theta, indent=2, ensure_ascii=True), encoding="utf-8")

    output_diag = args.output_diag_csv
    if not output_diag:
        output_diag = str(output_theta_path.with_name(output_theta_path.stem + "_diagnostic.csv"))

    diag_path = Path(output_diag)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with diag_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [args.x_key, args.gold_key, "gold_smoothed", "selected"] + component_keys
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            out_row = {
                args.x_key: row[args.x_key],
                args.gold_key: row[args.gold_key],
                "gold_smoothed": smoothed_gold[idx],
                "selected": 1 if idx == peak_idx else 0,
            }
            for key in component_keys:
                out_row[key] = row[key]
            writer.writerow(out_row)

    print(f"[proxy-point] rows={len(rows)} peak_idx={peak_idx} peak_kl={peak_row[args.x_key]:.6f}")
    print(f"[proxy-point] wrote theta: {output_theta_path}")
    print(f"[proxy-point] wrote diagnostic: {diag_path}")


if __name__ == "__main__":
    main()
