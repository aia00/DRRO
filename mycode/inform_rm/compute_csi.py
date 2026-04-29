#!/usr/bin/env python3
"""Compute CSI between RLHF latents and SFT/reference latents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .csi import cluster_separation_index
except ImportError:
    from csi import cluster_separation_index



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Cluster Separation Index (CSI).")
    parser.add_argument("--red_points", required=True, help="RLHF/output latent vectors (.npy)")
    parser.add_argument("--blue_points", required=True, help="SFT/reference latent vectors (.npy)")
    parser.add_argument("--dbscan_eps", type=float, default=0.5)
    parser.add_argument("--dbscan_min_samples", type=int, default=5)
    parser.add_argument("--ignore_noise", action="store_true")
    parser.add_argument("--output_json", default="")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    red = np.load(args.red_points)
    blue = np.load(args.blue_points)
    csi_value, cluster_count = cluster_separation_index(
        red,
        blue,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        ignore_noise=args.ignore_noise,
    )
    metrics = {
        "csi": float(csi_value),
        "cluster_count": int(cluster_count),
        "red_points": int(red.shape[0]),
        "blue_points": int(blue.shape[0]),
        "dbscan_eps": args.dbscan_eps,
        "dbscan_min_samples": args.dbscan_min_samples,
        "ignore_noise": bool(args.ignore_noise),
    }
    print(json.dumps(metrics, indent=2))
    if args.output_json:
        with Path(args.output_json).open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
