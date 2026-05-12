"""Run one SIC / baseline experiment and write config + results under output_dir.

Examples:
  python run_single_config.py --method ridge_rank1 --ridge_lambda 1e-6 --rank1_alpha 0.5
  python run_single_config.py --method applicant --output_dir runs/smoke_applicant
  python run_single_config.py --method infer_neural --checkpoint best_model.pt --model cnngru
  python run_single_config.py --method train_neural --train_epochs 2 --checkpoint model.pt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _git_hash() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_int_tuple(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _parse_lags(k_left: int, k_right: int) -> list[int]:
    return list(range(k_left, k_right + 1))


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            if hasattr(s, "flush"):
                s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def _default_output_dir(method: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(c if c.isalnum() else "_" for c in method)[:40]
    return Path("runs") / f"{ts}_{slug}"


def _applicant_style_canceller(tx_n, rx, helpers, Fs, lags):
    from rank1_sic import grid_search_rank1_alpha
    from ridge_sic import grid_search_ridge_lambda

    rx_mid, _, _ = grid_search_ridge_lambda(
        tx_n,
        rx,
        helpers,
        lags,
        Fs,
        lambdas=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3),
        orders_mem=(1, 3, 5),
        orders_conj=(1, 3),
        use_conjugate=True,
        use_cross=False,
    )
    rx_hat, _, _ = grid_search_rank1_alpha(rx, rx_mid, helpers)
    return rx_hat


def _run_canceller(args, tx_n, rx, Fs, helpers, lags) -> tuple[np.ndarray, dict]:
    meta: dict = {"method": args.method}
    t0 = time.perf_counter()

    if args.method == "applicant":
        rx_hat = _applicant_style_canceller(tx_n, rx, helpers, Fs, lags)
        meta["note"] = "ridge lambda grid + rank1 alpha grid (matches applicant_solution your_canceller)"

    elif args.method == "ridge_rank1":
        from rank1_sic import subtract_rank1
        from ridge_sic import ridge_canceller

        rx_mid = ridge_canceller(
            tx_n,
            rx,
            helpers,
            lags,
            Fs,
            ridge_lambda=args.ridge_lambda,
            orders_mem=args.orders_mem,
            orders_conj=args.orders_conj,
            use_conjugate=args.use_conjugate,
            use_cross=args.use_cross,
        )
        rx_hat = subtract_rank1(rx_mid, helpers, alpha=args.rank1_alpha)
        meta["ridge_lambda"] = args.ridge_lambda
        meta["rank1_alpha"] = args.rank1_alpha

    elif args.method == "ridge_only":
        from ridge_sic import ridge_canceller

        rx_hat = ridge_canceller(
            tx_n,
            rx,
            helpers,
            lags,
            Fs,
            ridge_lambda=args.ridge_lambda,
            orders_mem=args.orders_mem,
            orders_conj=args.orders_conj,
            use_conjugate=args.use_conjugate,
            use_cross=args.use_cross,
        )
        meta["ridge_lambda"] = args.ridge_lambda

    elif args.method == "ridge_rank1_grid":
        from rank1_sic import grid_search_rank1_alpha
        from ridge_sic import ridge_canceller

        rx_mid = ridge_canceller(
            tx_n,
            rx,
            helpers,
            lags,
            Fs,
            ridge_lambda=args.ridge_lambda,
            orders_mem=args.orders_mem,
            orders_conj=args.orders_conj,
            use_conjugate=args.use_conjugate,
            use_cross=args.use_cross,
        )
        rx_hat, best_a, best_avg = grid_search_rank1_alpha(
            rx, rx_mid, helpers, alphas=args.rank1_alphas
        )
        meta["ridge_lambda"] = args.ridge_lambda
        meta["rank1_alpha_selected"] = best_a
        meta["rank1_grid_best_avg_db_internal"] = best_avg

    elif args.method == "enhanced_baseline":
        from enhanced_baseline import enhanced_canceller

        rx_hat = enhanced_canceller(
            tx_n, rx, helpers, lambda_rank1=args.enhanced_rank1_lambda
        )
        meta["enhanced_rank1_lambda"] = args.enhanced_rank1_lambda

    elif args.method == "alternating":
        from alternating_sic import alternating_canceller

        rx_hat = alternating_canceller(
            tx_n,
            rx,
            helpers,
            Fs,
            n_iter=args.alternating_iters,
            lags=lags,
            ridge_lambda=args.ridge_lambda,
            rank1_alphas=args.rank1_alphas,
            orders_mem=args.orders_mem,
            orders_conj=args.orders_conj,
            use_conjugate=args.use_conjugate,
            use_cross=args.use_cross,
        )
        meta["ridge_lambda"] = args.ridge_lambda
        meta["alternating_iters"] = args.alternating_iters
        meta["rank1_alphas"] = list(args.rank1_alphas)

    elif args.method == "blockwise":
        from blockwise_sic import blockwise_canceller

        rx_hat = blockwise_canceller(
            tx_n,
            rx,
            helpers,
            Fs,
            lags,
            ridge_lambda=args.ridge_lambda,
            orders_mem=args.orders_mem,
            orders_conj=args.orders_conj,
            use_conjugate=args.use_conjugate,
            use_cross=args.use_cross,
        )
        meta["ridge_lambda"] = args.ridge_lambda

    elif args.method == "infer_neural":
        from infer import load_and_infer
        from device_utils import pick_torch_device

        dev = pick_torch_device(args.device if args.device.strip() else None)
        rx_hat = load_and_infer(
            tx_n,
            rx,
            checkpoint=args.checkpoint,
            model_type=args.model,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            blend_baseline=args.blend_baseline,
            helpers=helpers,
            device=dev,
        )
        meta["checkpoint"] = args.checkpoint
        meta["model"] = args.model
        meta["device"] = str(dev)

    else:
        raise ValueError(f"unknown method {args.method}")

    meta["seconds_canceller"] = time.perf_counter() - t0
    return rx_hat, meta


def _run_train_subprocess(args, out_dir: Path) -> dict:
    repo = Path(__file__).resolve().parent
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = out_dir / ckpt.name
    cmd = [
        sys.executable,
        str(repo / "train.py"),
        "--model",
        args.model,
        "--epochs",
        str(args.train_epochs),
        "--batch_size",
        str(args.train_batch_size),
        "--lr",
        str(args.train_lr),
        "--window_size",
        str(args.train_window_size),
        "--stride",
        str(args.train_stride),
        "--checkpoint",
        str(ckpt),
        "--device",
        args.train_device,
        "--num_workers",
        str(args.train_num_workers),
    ]
    if args.train_enhanced:
        cmd.append("--enhanced")
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=str(repo))
    elapsed = time.perf_counter() - t0
    return {
        "train_command": cmd,
        "seconds_train": elapsed,
        "checkpoint": str(ckpt),
    }


def main():
    methods = [
        "applicant",
        "ridge_rank1",
        "ridge_rank1_grid",
        "ridge_only",
        "enhanced_baseline",
        "alternating",
        "blockwise",
        "train_neural",
        "infer_neural",
    ]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mat", type=str, default="challenge.mat")
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory for config.json, results.json, stdout.log. Default: runs/<UTC>_<method>",
    )
    p.add_argument("--method", type=str, choices=methods, required=True)
    p.add_argument("--k_left", type=int, default=-16)
    p.add_argument("--k_right", type=int, default=16)
    p.add_argument("--ridge_lambda", type=float, default=1e-6)
    p.add_argument("--orders_mem", type=str, default="1,3,5")
    p.add_argument("--orders_conj", type=str, default="1,3")
    p.add_argument("--use_conjugate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_cross", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--rank1_alpha", type=float, default=1.0)
    p.add_argument(
        "--rank1_alphas",
        type=str,
        default="",
        help="Comma-separated grid for alternating / ridge_rank1_grid; default 0:0.05:1.0 step 0.05",
    )
    p.add_argument("--enhanced_rank1_lambda", type=float, default=1.0)
    p.add_argument("--alternating_iters", type=int, default=3)
    p.add_argument("--checkpoint", type=str, default="best_model.pt")
    p.add_argument("--model", choices=["mlp", "cnngru", "transformer"], default="cnngru")
    p.add_argument("--chunk_size", type=int, default=8192)
    p.add_argument("--overlap", type=int, default=256)
    p.add_argument("--blend_baseline", type=float, default=0.0)
    p.add_argument("--device", type=str, default="", help="Torch device (infer_neural)")
    p.add_argument("--train_epochs", type=int, default=20)
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--train_lr", type=float, default=1e-3)
    p.add_argument("--train_window_size", type=int, default=512)
    p.add_argument("--train_stride", type=int, default=128)
    p.add_argument("--train_device", type=str, default="")
    p.add_argument("--train_num_workers", type=int, default=0)
    p.add_argument("--train_enhanced", action="store_true")
    args = p.parse_args()

    if args.rank1_alphas.strip():
        rank1_alphas = tuple(float(x) for x in args.rank1_alphas.split(",") if x.strip())
    else:
        rank1_alphas = tuple(round(a, 2) for a in np.arange(0.0, 1.01, 0.05))
    args.rank1_alphas = rank1_alphas

    args.orders_mem = _parse_int_tuple(args.orders_mem)
    args.orders_conj = _parse_int_tuple(args.orders_conj)

    if args.method != "train_neural":
        from data_utils import load_data
        from task_and_baseline import baseline

    out = Path(args.output_dir) if args.output_dir else _default_output_dir(args.method)
    out.mkdir(parents=True, exist_ok=True)

    log_path = out / "stdout.log"
    log_f = open(log_path, "w", encoding="utf-8")
    tee_out = _Tee(sys.stdout, log_f)
    tee_err = _Tee(sys.stderr, log_f)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = tee_out, tee_err
    try:
        cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}
        cfg["git_hash"] = _git_hash()
        cfg["cwd"] = os.getcwd()
        with open(out / "config.json", "w", encoding="utf-8") as f:
            json.dump(_json_sanitize(cfg), f, indent=2)

        results: dict = {"metadata": {"method": args.method}}

        if args.method == "train_neural":
            print("train_neural: skipping MAT load; train.py loads data.")
            train_meta = _run_train_subprocess(args, out)
            results["metadata"].update(train_meta)
            results["metadata"]["seconds_total"] = train_meta["seconds_train"]
            results["baseline"] = None
            results["yours"] = None
            print("train_neural finished; scoring skipped (no rx_hat in this mode).")
        else:
            print(f"Loading {args.mat} ...")
            tx_n, rx, Fs, N, helpers = load_data(args.mat)
            lags = _parse_lags(args.k_left, args.k_right)
            rx_hat, run_meta = _run_canceller(args, tx_n, rx, Fs, helpers, lags)
            results["metadata"].update(run_meta)

            print("\n=== Baseline ===")
            baseline_reds, baseline_avg = helpers["score"](
                rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
            )

            label = "yours" if args.method != "infer_neural" else "neural"
            print(f"=== {label} ===")
            yours_reds, yours_avg = helpers["score"](rx, rx_hat, label=label)

            results["baseline"] = {
                "per_channel_db": baseline_reds,
                "average_db": baseline_avg,
            }
            results["yours"] = {
                "per_channel_db": yours_reds,
                "average_db": yours_avg,
            }
            results["metadata"]["seconds_total"] = (
                results["metadata"].get("seconds_canceller", 0.0)
            )

        with open(out / "results.json", "w", encoding="utf-8") as f:
            json.dump(_json_sanitize(results), f, indent=2)
        print(f"\nWrote {out / 'config.json'} and {out / 'results.json'}")
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        log_f.close()


if __name__ == "__main__":
    main()
