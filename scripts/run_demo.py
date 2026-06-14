import argparse
import subprocess
import sys
from pathlib import Path

'''
Run the bundled patient 22 demo from the repository root:

python scripts/run_demo.py --eval
python scripts/run_demo.py --fit
python scripts/run_demo.py --fit --eval
'''


def main():
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    from HemoPIC.io.paths import require_fit_dir

    example_root = repo_root / "assets" / "example"
    dataset_root = example_root / "ISLES2017_Training"
    default_out_root = example_root / "HemoPIC_outputs"

    p = argparse.ArgumentParser(
        description="Run HemoPIC on bundled patient 22 demo data."
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=default_out_root,
        help="Output directory for HemoPIC fitting (default: assets/example/HemoPIC_outputs)",
    )
    p.add_argument(
        "--eval-dir",
        type=Path,
        default=example_root / "HemoPIC_eval",
        help="Output directory for evaluation figures (default: assets/example/HemoPIC_eval)",
    )
    p.add_argument(
        "--fit",
        action="store_true",
        help="Run Step 1 fitting (skipped by default; bundled outputs are included)",
    )
    p.add_argument(
        "--eval",
        action="store_true",
        help="Run slice-map evaluation for manuscript slices 13, 14, 15",
    )
    p.add_argument("--viz", action="store_true")
    args = p.parse_args()

    if not dataset_root.exists():
        raise SystemExit("Demo data not found: " + str(dataset_root))

    patient_num = 22
    k_gm = 8
    k_wm = 6
    seed = 0
    out_root = args.out_root.expanduser().resolve()

    if args.fit:
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "run_hemopic.py"),
            str(dataset_root),
            str(out_root),
            str(patient_num),
            str(k_gm),
            str(k_wm),
            str(seed),
        ]
        if args.viz:
            cmd.append("--viz")
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=repo_root)
    elif not args.eval:
        print("Bundled demo inputs and HemoPIC outputs are in assets/example/.")
        print("Use --eval to generate slice maps, or --fit to rerun Step 1.")

    if args.eval:
        require_fit_dir(out_root, patient_num)
        eval_dir = args.eval_dir.expanduser().resolve()
        slice_script = repo_root / "Eval" / "summary_maps" / "run_eval_slice_maps.py"
        for z in (13, 14, 15):
            eval_cmd = [
                sys.executable,
                str(slice_script),
                str(dataset_root),
                str(out_root),
                str(eval_dir),
                str(patient_num),
                str(z),
                "70",
                "7",
                "15",
            ]
            print("Running:", " ".join(eval_cmd))
            subprocess.run(eval_cmd, check=True, cwd=repo_root)
        print("EVAL_DIR", str(eval_dir))

    print("DATASET_ROOT", str(dataset_root))
    print("FIT_ROOT", str(out_root))
    print("Done.")


if __name__ == "__main__":
    main()
