"""Generate Slurm batch scripts that each run one `run_single_config.py` job.

Requires `--conda_env` (no default). Does not embed secrets or API keys.

Examples:
  python run_cluster.py --conda_env myenv --dry_run
  python run_cluster.py --conda_env myenv --write_dir cluster_out/batch1
  python run_cluster.py --conda_env myenv -A myaccount --partition gpuA --gpus 1
"""
from __future__ import annotations

import argparse
import os
import shlex
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


def _render_sbatch(
    *,
    account: str,
    partition: str | None,
    time_limit: str,
    job_name: str,
    cpus: int,
    gpus: int,
    conda_env: str,
    repo: Path,
    stdout_path: str,
    stderr_path: str,
    run_argv: list[str],
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH -A {account}",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -t {time_limit}",
        f"#SBATCH -o {stdout_path}",
        f"#SBATCH -e {stderr_path}",
        f"#SBATCH --cpus-per-task={cpus}",
    ]
    if partition:
        lines.append(f"#SBATCH -p {partition}")
    if gpus and gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
    lines.append("#SBATCH --chdir=" + shlex.quote(str(repo)))
    lines.extend(
        [
            "set -euo pipefail",
            'eval "$(conda shell.bash hook)"',
            f"conda activate {shlex.quote(conda_env)}",
            _shell_quote_cmd(run_argv),
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conda_env", type=str, required=True, help="Conda environment name to activate")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--cpu_cores", type=int, default=4, help="SBATCH --cpus-per-task")
    p.add_argument("--gpus", type=int, default=0, help="If >0, adds #SBATCH --gres=gpu:N")
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
        "--write_dir",
        type=str,
        default="",
        help="Write per-job .slurm files here. Default: cluster_batches/<UTC>/",
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
    write_dir = Path(args.write_dir) if args.write_dir else repo / "cluster_batches" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_root = repo / "sbatch_logs" / log_date

    if args.dry_run:
        print(f"# {len(combos)} combinations; repo={repo}")
        print(f"# log_root={log_root}")

    partition = args.partition.strip() or None

    for i, combo in enumerate(combos):
        out_dir = _format_output_dir(args.output_dir_template, i, combo)
        run_argv = _argv_from_combo(combo, args.mat, out_dir)
        job_slug = f"{args.job_name}_{i:04d}"
        log_job_dir = log_root / job_slug
        stdout_p = log_job_dir / "%j.out"
        stderr_p = log_job_dir / "%j.err"
        body = _render_sbatch(
            account=args.account,
            partition=partition,
            time_limit=args.time_limit,
            job_name=job_slug,
            cpus=args.cpu_cores,
            gpus=args.gpus,
            conda_env=args.conda_env,
            repo=repo,
            stdout_path=str(stdout_p),
            stderr_path=str(stderr_p),
            run_argv=run_argv,
        )
        if args.dry_run:
            print("---")
            print(body)
            print(_shell_quote_cmd(["sbatch", f"{write_dir / (job_slug + '.slurm')}"]))
        else:
            write_dir.mkdir(parents=True, exist_ok=True)
            log_job_dir.mkdir(parents=True, exist_ok=True)
            slurm_path = write_dir / f"{job_slug}.slurm"
            slurm_path.write_text(body, encoding="utf-8")
            os.chmod(slurm_path, 0o755)

    if not args.dry_run:
        driver = write_dir / "submit_all.sh"
        lines = ["#!/bin/bash", "set -euo pipefail", f"cd {shlex.quote(str(repo))}"]
        for i in range(len(combos)):
            job_slug = f"{args.job_name}_{i:04d}"
            lines.append(f"sbatch {shlex.quote(str(write_dir / (job_slug + '.slurm')))}")
        driver.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.chmod(driver, 0o755)
        print(f"Wrote {len(combos)} Slurm scripts under {write_dir}")
        print(f"Submit with: bash {driver}")


if __name__ == "__main__":
    main()
