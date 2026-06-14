import sys
import argparse
from pathlib import Path


### Entry point for tracer estimation evaluation
def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_root", type=Path)
    p.add_argument("fit_root", type=Path)
    p.add_argument("report_root", type=Path)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from Eval.utils.tracer_estimation.main import run_report

    run_report(
        dataset_root=args.dataset_root.expanduser().resolve(),
        fit_root=args.fit_root.expanduser().resolve(),
        report_root=args.report_root.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()