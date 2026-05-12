"""Submit Slurm jobs directly: one `sbatch --wrap` per `run_single_config.py` run.

Requires `--conda_env` (no default). Does not embed secrets or API keys.

Examples:
  python run_cluster.py --conda_env myenv --dry_run
  python run_cluster.py --conda_env myenv -A myaccount
  python run_cluster.py --conda_env myenv -A myaccount --partition gpuA --gpus 1
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from string import Formatter


def create_argument_combinations():
    """Return modest default grid: ridge lambdas × rank1 alphas × two lag presets."""
    ridge_lambdas = (1e-6, 1e-5)
    rank1_alphas = (0.5, 1.0)
    lag_presets = ((-16, 16), (-12, 12))
    combos = []
    for k_left, k_right in lag_presets:
        for ridge_lambda in ridge_lambdas:
            for rank1_alpha in rank1_alphas:
                combos.append(
                    {
                        "method": "ridge_rank1",
                        "k_left": k_left,
                        "k_right": k_right,
                        "ridge_lambda": ridge_lambda,
                        "rank1_alpha": rank1_alpha,
                    }
                )
    return combos


def _format_output_dir(template: str, index: int, combo: dict) -> str:
    """Fill only field names present in the template (avoids str.format extra-key errors)."""
    names = {fn for _, fn, _, _ in Formatter().parse(template) if fn}
    fmt = {"i": index}
    for name in names:
        if name == "i":
            continue
        if name not in combo:
            raise ValueError(
                f"output_dir_template field {name!r} missing from combo keys {sorted(combo)}"
            )
        fmt[name] = combo[name]
    return template.format(**fmt)


def _argv_from_combo(combo: dict, mat: str, out_dir: str) -> list[str]:
    """Full argv including interpreter for the Slurm task line."""
    argv = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_single_config.py"),
        "--mat",
        mat,
        "--output_dir",
        out_dir,
    ]
    for key, val in combo.items():
        flag = f"--{key}"
        if isinstance(val, bool):
            argv.extend([flag, "true" if val else "false"])
        else:
            argv.extend([flag, str(val)])
    return argv


def _shell_quote_cmd(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def _wrap_bash_command(*, conda_env: str, run_argv: list[str]) -> str:
    """Single bash -lc argument for sbatch --wrap (conda + python)."""
    inner = " && ".join(
        [
            'eval "$(conda shell.bash hook)"',
            f"conda activate {shlex.quote(conda_env)}",
            _shell_quote_cmd(run_argv),
        ]
    )
    return f"bash -lc {shlex.quote(inner)}"


def _sbatch_argv(
    *,
    account: str,
    partition: str | None,
    time_limit: str,
    job_name: str,
    cpus: int,
    gpus: int,
    repo: Path,
    stdout_path: str,
    stderr_path: str,
    wrap: str,
) -> list[str]:
    argv = [
        "sbatch",
        "-A",
        account,
        "-J",
        job_name,
        "-t",
        time_limit,
        "-o",
        stdout_path,
        "-e",
        stderr_path,
        f"--cpus-per-task={cpus}",
        "--chdir",
        str(repo),
        "--export=ALL,PYTHONUNBUFFERED=1",
    ]
    if partition:
        argv.extend(["-p", partition])
    if gpus and gpus > 0:
        argv.append(f"--gres=gpu:{gpus}")
    argv.extend(["--wrap", wrap])
    return argv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conda_env", type=str, required=True, help="Conda environment name to activate")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--cpu_cores", type=int, default=4, help="SBATCH --cpus-per-task")
    p.add_argument("--gpus", type=int, default=0, help="If >0, adds --gres=gpu:N to sbatch")
    p.add_argument("-A", "--account", type=str, default="proj_XXXX", dest="account")
    p.add_argument("--partition", type=str, default="", help="Optional Slurm partition (-p)")
    p.add_argument("--time", type=str, default="0-1:00:00", dest="time_limit")
    p.add_argument("--job_name", type=str, default="sic_grid")
    p.add_argument(
        "--mat",
        type=str,
        default="challenge.mat",
        help="Passed to each run_single_config.py",
    )
    p.add_argument(
        "--repo",
        type=str,
        default="",
        help="Repository root (default: directory containing this script)",
    )
    p.add_argument(
        "--manifest_dir",
        type=str,
        default="",
        help="Optional: write manifest.json with sbatch argv and job ids here. Default: cluster_batches/<UTC>/",
    )
    p.add_argument(
        "--log_date",
        type=str,
        default="",
        help="Subfolder under sbatch_logs/ (default: today's UTC date YYYY-MM-DD)",
    )
    p.add_argument(
        "--output_dir_template",
        type=str,
        default="runs/cluster_{i}",
        help="Template for --output_dir; {i} is job index; other keys from combo allowed",
    )
    args = p.parse_args()

    repo = Path(args.repo) if args.repo else Path(__file__).resolve().parent
    log_date = args.log_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    combos = create_argument_combinations()
    manifest_dir = (
        Path(args.manifest_dir)
        if args.manifest_dir
        else repo / "cluster_batches" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    log_root = repo / "sbatch_logs" / log_date

    if args.dry_run:
        print(f"# {len(combos)} combinations; repo={repo}")
        print(f"# log_root={log_root}")

    partition = args.partition.strip() or None
    manifest: list[dict] = []

    for i, combo in enumerate(combos):
        out_dir = _format_output_dir(args.output_dir_template, i, combo)
        run_argv = _argv_from_combo(combo, args.mat, out_dir)
        job_slug = f"{args.job_name}_{i:04d}"
        log_job_dir = log_root / job_slug
        stdout_p = log_job_dir / "%j.out"
        stderr_p = log_job_dir / "%j.err"
        wrap = _wrap_bash_command(conda_env=args.conda_env, run_argv=run_argv)
        sbatch_argv = _sbatch_argv(
            account=args.account,
            partition=partition,
            time_limit=args.time_limit,
            job_name=job_slug,
            cpus=args.cpu_cores,
            gpus=args.gpus,
            repo=repo,
            stdout_path=str(stdout_p),
            stderr_path=str(stderr_p),
            wrap=wrap,
        )
        if args.dry_run:
            print("---")
            print(_shell_quote_cmd(sbatch_argv))
        else:
            manifest_dir.mkdir(parents=True, exist_ok=True)
            log_job_dir.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                sbatch_argv,
                cwd=str(repo),
                capture_output=True,
                text=True,
                check=False,
            )
            job_id = None
            if proc.stdout:
                # "Submitted batch job 12345"
                for line in proc.stdout.strip().splitlines():
                    if "Submitted batch job" in line:
                        job_id = line.strip().split()[-1]
            entry = {
                "job_slug": job_slug,
                "slurm_job_id": job_id,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "sbatch_argv": sbatch_argv,
                "combo": combo,
                "output_dir": out_dir,
            }
            manifest.append(entry)
            if proc.returncode != 0:
                print(f"ERROR sbatch {job_slug}: rc={proc.returncode}\n{proc.stderr}", file=sys.stderr)
            else:
                print(f"Submitted {job_slug} -> {job_id}")

    if not args.dry_run and manifest:
        mf = manifest_dir / "manifest.json"
        mf.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Manifest: {mf}")


if __name__ == "__main__":
    main()
