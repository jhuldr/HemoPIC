import argparse
import subprocess
import sys
from pathlib import Path

'''
Run lesion ROC/PR evaluation (HemoPIC vs ISLES silver-standard maps).

python Eval/lesion_roc/run_eval_lesion_roc.py \
  /path/to/ISLES2017_Training \
  /path/to/HemoPIC_outputs \
  /path/to/HemoPIC_eval/lesion_roc \
  --all
'''


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate OT lesion discrimination with ROC/PR curves (CBF, CBV, MTT)."
    )
    p.add_argument("dataset_root", type=Path)
    p.add_argument("fit_root", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--patients", type=int, nargs="*", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=("CBF", "CBV", "MTT"),
        default=None,
        help="Maps to evaluate (default: CBF CBV MTT).",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=("deconv", "hemopic"),
        default=None,
        help="Pipelines to compare (default: deconv and hemopic).",
    )
    p.add_argument(
        "--relative-mode",
        choices=("brain_median", "nonlesion_median"),
        default="brain_median",
    )
    args = p.parse_args()

    script = Path(__file__).resolve().parent / "lesion_eval.py"
    cmd = [
        sys.executable,
        str(script),
        "--dataset-root",
        str(args.dataset_root.expanduser().resolve()),
        "--fit-root",
        str(args.fit_root.expanduser().resolve()),
        "--out-dir",
        str(args.out_dir.expanduser().resolve()),
        "--relative-mode",
        args.relative_mode,
    ]

    if args.all:
        cmd.append("--all")
    elif args.patients:
        cmd.extend(["--patients", *[str(int(x)) for x in args.patients]])
    else:
        raise SystemExit("Use --all or --patients.")

    if args.metrics:
        cmd.extend(["--metrics", *args.metrics])

    if args.methods:
        cmd.extend(["--methods", *args.methods])

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
