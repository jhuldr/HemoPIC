import sys
import argparse
from pathlib import Path

'''
Run:

DATASET_ROOT=/path/to/ISLES2017_Training
OUT_ROOT=/path/to/HemoPIC_outputs
PATIENT_NUM=<PATIENT_NUM>
K_GM=8
K_WM=6
SEED=0

python3 scripts/run_hemopic.py "$DATASET_ROOT" "$OUT_ROOT" "$PATIENT_NUM" "$K_GM" "$K_WM" "$SEED"
'''

### Entry point for fitting one patient
def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_root", type=Path)
    p.add_argument("out_root", type=Path)
    p.add_argument("patient_num", type=int)
    p.add_argument("k_gm", type=int)
    p.add_argument("k_wm", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("--viz", action="store_true")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    from HemoPIC.config import FitConfig
    from HemoPIC.core_pipeline import run_core

    cfg = FitConfig()
    cfg.enable_viz = bool(args.viz)

    core_out = run_core(
        dataset_root=args.dataset_root.expanduser().resolve(),
        out_root=args.out_root.expanduser().resolve(),
        patient_num=args.patient_num,
        k_gm=args.k_gm,
        k_wm=args.k_wm,
        seed=args.seed,
        cfg=cfg,
    )

    if cfg.enable_viz:
        run_viz(core_out, cfg)


if __name__ == "__main__":
    main()
