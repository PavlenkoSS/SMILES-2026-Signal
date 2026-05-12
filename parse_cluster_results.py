"""Scan run directories for results.json (+ sibling config.json) and print a table.

Examples:
  python parse_cluster_results.py
  python parse_cluster_results.py --root runs --csv summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def _find_result_dirs(root: Path):
    """Yield directories that contain results.json (direct children of walk)."""
    root = root.resolve()
    if not root.is_dir():
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        if "results.json" in filenames:
            yield Path(dirpath)


def _flatten_config(cfg: dict) -> dict:
    keys = (
        "method",
        "mat",
        "ridge_lambda",
        "rank1_alpha",
        "k_left",
        "k_right",
        "enhanced_rank1_lambda",
        "alternating_iters",
        "checkpoint",
        "model",
    )
    return {k: cfg.get(k, "") for k in keys}


def _row(run_dir: Path, results: dict, cfg: dict | None) -> dict:
    cfg = cfg or {}
    meta = results.get("metadata") or {}
    method = meta.get("method", cfg.get("method", ""))
    base = results.get("baseline") or {}
    yours = results.get("yours") or {}
    b_ch = base.get("per_channel_db")
    y_ch = yours.get("per_channel_db")
    fc = _flatten_config(cfg)

    def ch_str(ch):
        if ch is None:
            return ""
        if isinstance(ch, (list, tuple)):
            return ";".join(f"{float(x):.6f}" for x in ch)
        return str(ch)

    return {
        "run_path": str(run_dir),
        "method": method,
        "baseline_avg_db": base.get("average_db", ""),
        "yours_avg_db": yours.get("average_db", ""),
        "per_channel_db": ch_str(y_ch),
        "baseline_per_channel_db": ch_str(b_ch),
        **{f"cfg_{k}": v for k, v in fc.items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=str,
        default="runs",
        help="Directory tree to walk for results.json (default: ./runs)",
    )
    p.add_argument(
        "--csv",
        type=str,
        default="",
        help="Write CSV to this path; default: print TSV to stdout",
    )
    args = p.parse_args()

    root = Path(args.root)
    rows = []
    for run_dir in sorted(_find_result_dirs(root), key=lambda p: str(p)):
        rp = run_dir / "results.json"
        try:
            results = json.loads(rp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cfg_path = run_dir / "config.json"
        cfg = None
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cfg = None
        rows.append(_row(run_dir, results, cfg))

    if not rows:
        print(f"No results.json found under {root}", file=sys.stderr)
        sys.exit(1)

    fieldnames = list(rows[0].keys())
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.csv}")
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
