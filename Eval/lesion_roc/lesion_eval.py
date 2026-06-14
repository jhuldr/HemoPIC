#!/usr/bin/env python3
"""
Lesion discrimination evaluation for CBF, CBV, and MTT perfusion maps.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

EPS = 1.0e-12
_TRAPZ = getattr(np, "trapezoid", np.trapz)


@dataclass(frozen=True)
class MapSpec:
    name: str
    lesion_direction: str  # "low" or "high"


MAP_SPECS = (
    MapSpec("CBF", "low"),
    MapSpec("CBV", "low"),
    MapSpec("MTT", "high"),
)


def map_specs_for_evaluation(args) -> Tuple[MapSpec, ...]:
    """Maps to score (ROC/PR); default is all ``MAP_SPECS``."""
    if args.metrics is None:
        return MAP_SPECS
    allowed = {s.name for s in MAP_SPECS}
    for name in args.metrics:
        if name not in allowed:
            raise RuntimeError(f"Unknown metric {name!r}. Allowed: {sorted(allowed)}")
    chosen = set(args.metrics)
    return tuple(s for s in MAP_SPECS if s.name in chosen)


@dataclass(frozen=True)
class CurveResult:
    roc_fpr: np.ndarray
    roc_tpr: np.ndarray
    roc_thresholds: np.ndarray
    roc_auc: float
    pr_recall: np.ndarray
    pr_precision: np.ndarray
    pr_thresholds: np.ndarray
    pr_auc_trapezoid: float
    average_precision: float


def format_patient_pattern(pattern: str, patient_id: int) -> str:
    return pattern.format(patient=patient_id, id=patient_id, ID=patient_id)


def parse_last_integer(text: str) -> Optional[int]:
    matches = re.findall(r"\d+", text)
    if not matches:
        return None
    return int(matches[-1])


def resolve_nifti(base_dir: Path, name_or_stem: str) -> Optional[Path]:
    candidate = Path(name_or_stem)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    direct = base_dir / name_or_stem
    if direct.exists():
        return direct

    if name_or_stem.endswith(".nii") or name_or_stem.endswith(".nii.gz"):
        return None

    nii = base_dir / f"{name_or_stem}.nii"
    if nii.exists():
        return nii

    niigz = base_dir / f"{name_or_stem}.nii.gz"
    if niigz.exists():
        return niigz

    return None


def load_nifti(path: Path) -> Tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    data = np.asanyarray(img.get_fdata()).astype(np.float64)
    return img, data


def save_nifti_like(ref_img: nib.Nifti1Image, data: np.ndarray, out_path: Path, dtype=np.float32) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ref_img.header.copy()
    header.set_data_dtype(dtype)
    out = nib.Nifti1Image(data.astype(dtype), affine=ref_img.affine, header=header)
    nib.save(out, str(out_path))


def finite_positive_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return np.nan
    return float(np.median(values))


def get_brain_mask(patient_dir: Path, shape: Tuple[int, int, int], ref_data: np.ndarray, args) -> np.ndarray:
    if args.brain_mask_name:
        mask_path = resolve_nifti(patient_dir, args.brain_mask_name)
        if mask_path is not None:
            _, mask = load_nifti(mask_path)
            if tuple(mask.shape[:3]) == tuple(shape):
                return mask > 0

    seg_path = patient_dir / args.segmentation_relative_path
    if seg_path.exists():
        _, seg = load_nifti(seg_path)
        if tuple(seg.shape[:3]) == tuple(shape):
            return seg > 0

    return np.isfinite(ref_data) & (ref_data > 0.0)


def compute_relative_map(
    data: np.ndarray,
    brain_mask: np.ndarray,
    lesion_mask: np.ndarray,
    relative_mode: str,
) -> Tuple[np.ndarray, float, np.ndarray]:
    valid = brain_mask & np.isfinite(data) & (data > 0.0)

    if relative_mode == "brain_median":
        reference_mask = valid
    elif relative_mode == "nonlesion_median":
        reference_mask = valid & (~lesion_mask)
    else:
        raise ValueError(f"Unknown relative mode: {relative_mode}")

    denominator = finite_positive_median(data[reference_mask])
    if not np.isfinite(denominator) or denominator <= 0.0:
        denominator = finite_positive_median(data[valid])

    relative = np.full(data.shape, np.nan, dtype=np.float64)
    if np.isfinite(denominator) and denominator > 0.0:
        relative[valid] = data[valid] / denominator

    return relative, denominator, valid


def make_lesion_score(relative_map: np.ndarray, lesion_direction: str) -> np.ndarray:
    score = np.full(relative_map.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(relative_map) & (relative_map > 0.0)

    if lesion_direction == "low":
        score[valid] = 1.0 / np.maximum(relative_map[valid], EPS)
    elif lesion_direction == "high":
        score[valid] = relative_map[valid]
    else:
        raise ValueError(f"Unknown lesion direction: {lesion_direction}")

    return score


def compute_curves(y_true: np.ndarray, score: np.ndarray) -> CurveResult:
    y_true = np.asarray(y_true).astype(np.uint8)
    score = np.asarray(score).astype(np.float64)

    ok = np.isfinite(score)
    y_true = y_true[ok]
    score = score[ok]

    n = int(y_true.size)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n == 0 or n_pos == 0 or n_neg == 0:
        empty = np.array([], dtype=np.float64)
        return CurveResult(empty, empty, empty, np.nan, empty, empty, empty, np.nan, np.nan)

    order = np.argsort(-score, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = score[order]

    distinct_idx = np.r_[np.where(np.diff(score_sorted) != 0.0)[0], n - 1]
    thresholds = score_sorted[distinct_idx].astype(np.float64)

    true_positive = np.cumsum(y_sorted == 1)[distinct_idx].astype(np.float64)
    false_positive = np.cumsum(y_sorted == 0)[distinct_idx].astype(np.float64)

    tpr = true_positive / float(n_pos)
    fpr = false_positive / float(n_neg)

    roc_tpr = np.r_[0.0, tpr, 1.0]
    roc_fpr = np.r_[0.0, fpr, 1.0]
    roc_thresholds = np.r_[np.inf, thresholds, -np.inf]
    roc_auc = float(_TRAPZ(roc_tpr, roc_fpr))

    precision = true_positive / np.maximum(true_positive + false_positive, EPS)
    recall = true_positive / float(n_pos)

    pr_recall = np.r_[0.0, recall]
    pr_precision = np.r_[1.0, precision]
    pr_auc_trapezoid = float(_TRAPZ(pr_precision, pr_recall))

    recall_step = np.r_[0.0, recall]
    precision_step = np.r_[precision[0], precision]
    average_precision = float(np.sum((recall_step[1:] - recall_step[:-1]) * precision_step[1:]))

    return CurveResult(
        roc_fpr=roc_fpr,
        roc_tpr=roc_tpr,
        roc_thresholds=roc_thresholds,
        roc_auc=roc_auc,
        pr_recall=pr_recall,
        pr_precision=pr_precision,
        pr_thresholds=thresholds,
        pr_auc_trapezoid=pr_auc_trapezoid,
        average_precision=average_precision,
    )


def write_curve_csv(path: Path, columns: Tuple[str, str], x: np.ndarray, y: np.ndarray, thresholds: Optional[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        if thresholds is None:
            writer.writerow([columns[0], columns[1]])
            for xi, yi in zip(x, y):
                writer.writerow([float(xi), float(yi)])
        else:
            writer.writerow([columns[0], columns[1], "threshold"])
            for idx, (xi, yi) in enumerate(zip(x, y)):
                thr = thresholds[idx] if idx < len(thresholds) else np.nan
                if np.isfinite(thr):
                    writer.writerow([float(xi), float(yi), float(thr)])
                else:
                    writer.writerow([float(xi), float(yi), str(thr)])


def plot_curves(curves: Dict[str, CurveResult], out_dir: Path, title_suffix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(1, 1, 1)
    for label, curve in curves.items():
        if curve.roc_fpr.size > 0:
            ax.plot(curve.roc_fpr, curve.roc_tpr, label=f"{label} AUC={curve.roc_auc:.3f}")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle=":")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves {title_suffix}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(1, 1, 1)
    for label, curve in curves.items():
        if curve.pr_recall.size > 0:
            ax.plot(curve.pr_recall, curve.pr_precision, label=f"{label} AP={curve.average_precision:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall curves {title_suffix}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=220)
    plt.close(fig)


def get_deconv_names(args) -> Dict[str, str]:
    return {
        "CBF": args.deconv_cbf_name,
        "CBV": args.deconv_cbv_name,
        "MTT": args.deconv_mtt_name,
    }


def get_hemopic_names(args) -> Dict[str, str]:
    return {
        "CBF": args.hemopic_cbf_name,
        "CBV": args.hemopic_cbv_name,
        "MTT": args.hemopic_mtt_name,
    }


def evaluate_patient(patient_id: int, args) -> Tuple[List[dict], List[Tuple[str, np.ndarray, np.ndarray]]]:
    patient_dir = args.dataset_root / format_patient_pattern(args.dataset_patient_pattern, patient_id)
    out_dir = args.out_dir / f"patient_{patient_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_hemopic = args.methods is None or "hemopic" in args.methods
    if run_hemopic:
        if args.fit_root is None:
            raise RuntimeError("fit_root is required for hemopic evaluation.")
        fit_dir = args.fit_root / format_patient_pattern(args.fit_patient_pattern, patient_id)
        if not fit_dir.exists():
            raise FileNotFoundError(f"Missing HemoPIC output directory for ID {patient_id}")
    else:
        fit_dir = None

    if not patient_dir.exists():
        raise FileNotFoundError(f"Missing patient directory for ID {patient_id}")

    lesion_path = resolve_nifti(patient_dir, args.lesion_mask_name)
    if lesion_path is None:
        raise FileNotFoundError(f"Missing lesion mask for ID {patient_id}")

    ref_path = resolve_nifti(patient_dir, args.deconv_cbf_name)
    if ref_path is None:
        raise FileNotFoundError(f"Missing reference CBF map for ID {patient_id}")

    ref_img, ref_data = load_nifti(ref_path)
    shape = tuple(ref_data.shape[:3])

    _, lesion_data = load_nifti(lesion_path)
    if tuple(lesion_data.shape[:3]) != shape:
        raise RuntimeError(f"Lesion mask shape mismatch for ID {patient_id}")
    lesion_mask = lesion_data > 0

    brain_mask = get_brain_mask(patient_dir, shape, ref_data, args)
    save_nifti_like(ref_img, brain_mask.astype(np.uint8), out_dir / "brain_mask.nii.gz", dtype=np.uint8)
    save_nifti_like(ref_img, lesion_mask.astype(np.uint8), out_dir / "lesion_mask.nii.gz", dtype=np.uint8)

    full_method_config = {"deconv": (patient_dir, get_deconv_names(args))}
    if fit_dir is not None:
        full_method_config["hemopic"] = (fit_dir, get_hemopic_names(args))
    if args.methods is None:
        method_config = full_method_config
    else:
        unknown_m = set(args.methods) - {"deconv", "hemopic"}
        if unknown_m:
            raise RuntimeError(f"Unknown method(s): {sorted(unknown_m)}. Use deconv and/or hemopic.")
        method_config = {k: full_method_config[k] for k in args.methods}

    map_specs = map_specs_for_evaluation(args)
    if not map_specs:
        raise RuntimeError("No metrics selected (--metrics produced an empty list).")

    rows: List[dict] = []
    pooled_entries: List[Tuple[str, np.ndarray, np.ndarray]] = []
    patient_curves: Dict[str, CurveResult] = {}

    for method_name, (map_dir, map_names) in method_config.items():
        for spec in map_specs:
            map_path = resolve_nifti(map_dir, map_names[spec.name])
            if map_path is None:
                print(f"[WARN] Missing {method_name} {spec.name} for patient {patient_id}")
                continue

            _, data = load_nifti(map_path)
            if tuple(data.shape[:3]) != shape:
                raise RuntimeError(f"Shape mismatch for {method_name} {spec.name}, patient {patient_id}")

            relative_map, denominator, valid_mask = compute_relative_map(
                data=data,
                brain_mask=brain_mask,
                lesion_mask=lesion_mask,
                relative_mode=args.relative_mode,
            )
            score_map = make_lesion_score(relative_map, spec.lesion_direction)

            eval_mask = valid_mask & np.isfinite(relative_map) & np.isfinite(score_map)
            y_true = lesion_mask[eval_mask].astype(np.uint8)
            y_score = score_map[eval_mask].astype(np.float64)

            curve = compute_curves(y_true, y_score)
            label = f"{method_name}_{spec.name}"
            patient_curves[label] = curve
            pooled_entries.append((label, y_true, y_score))

            save_nifti_like(
                ref_img,
                relative_map,
                out_dir / "relative_maps" / f"{label}_relative.nii.gz",
                dtype=np.float32,
            )
            save_nifti_like(
                ref_img,
                score_map,
                out_dir / "score_maps" / f"{label}_score.nii.gz",
                dtype=np.float32,
            )

            curve_dir = out_dir / "curves_csv" / label
            write_curve_csv(
                curve_dir / "roc_curve.csv",
                ("false_positive_rate", "true_positive_rate"),
                curve.roc_fpr,
                curve.roc_tpr,
                curve.roc_thresholds,
            )
            write_curve_csv(
                curve_dir / "pr_curve.csv",
                ("recall", "precision"),
                curve.pr_recall,
                curve.pr_precision,
                None,
            )

            n_eval = int(y_true.size)
            n_lesion = int(np.sum(y_true == 1))
            n_nonlesion = int(np.sum(y_true == 0))

            rows.append(
                {
                    "patient_id": int(patient_id),
                    "method": method_name,
                    "metric": spec.name,
                    "score_rule": "inverse_relative" if spec.lesion_direction == "low" else "relative",
                    "relative_mode": args.relative_mode,
                    "normalization_denominator": float(denominator) if np.isfinite(denominator) else np.nan,
                    "n_eval_voxels": n_eval,
                    "n_lesion_voxels": n_lesion,
                    "n_nonlesion_voxels": n_nonlesion,
                    "lesion_fraction": float(n_lesion / n_eval) if n_eval > 0 else np.nan,
                    "roc_auc": curve.roc_auc,
                    "pr_auc_trapezoid": curve.pr_auc_trapezoid,
                    "average_precision": curve.average_precision,
                    "source_file": map_path.name,
                }
            )

    plot_curves(patient_curves, out_dir, f"patient {patient_id}")
    write_dict_rows(rows, out_dir / "patient_metrics.csv")

    with open(out_dir / "run_info.json", "w", encoding="utf8") as f:
        json.dump(
            {
                "patient_id": int(patient_id),
                "relative_mode": args.relative_mode,
                "n_brain_voxels": int(np.sum(brain_mask)),
                "n_lesion_voxels_in_brain": int(np.sum(brain_mask & lesion_mask)),
            },
            f,
            indent=2,
        )

    return rows, pooled_entries


def write_dict_rows(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_patient_metrics(rows: List[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["metric"]), []).append(row)

    summary = []
    for (method, metric), group in sorted(grouped.items()):
        roc = np.array([float(x["roc_auc"]) for x in group], dtype=np.float64)
        pr_auc = np.array([float(x["pr_auc_trapezoid"]) for x in group], dtype=np.float64)
        ap = np.array([float(x["average_precision"]) for x in group], dtype=np.float64)

        summary.append(
            {
                "method": method,
                "metric": metric,
                "n_patients": len(group),
                "mean_roc_auc": float(np.nanmean(roc)),
                "std_roc_auc": float(np.nanstd(roc)),
                "mean_pr_auc_trapezoid": float(np.nanmean(pr_auc)),
                "std_pr_auc_trapezoid": float(np.nanstd(pr_auc)),
                "mean_average_precision": float(np.nanmean(ap)),
                "std_average_precision": float(np.nanstd(ap)),
            }
        )
    return summary


def save_pooled_results(entries: List[Tuple[str, np.ndarray, np.ndarray]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
    for label, y_true, score in entries:
        if label not in grouped:
            grouped[label] = ([], [])
        grouped[label][0].append(y_true)
        grouped[label][1].append(score)

    rows = []
    curves = {}
    for label, (ys, scores) in sorted(grouped.items()):
        y = np.concatenate(ys) if ys else np.array([], dtype=np.uint8)
        s = np.concatenate(scores) if scores else np.array([], dtype=np.float64)
        curve = compute_curves(y, s)
        curves[label] = curve

        method, metric = label.split("_", 1)
        rows.append(
            {
                "method": method,
                "metric": metric,
                "n_voxels": int(y.size),
                "n_lesion_voxels": int(np.sum(y == 1)),
                "n_nonlesion_voxels": int(np.sum(y == 0)),
                "roc_auc": curve.roc_auc,
                "pr_auc_trapezoid": curve.pr_auc_trapezoid,
                "average_precision": curve.average_precision,
            }
        )

        curve_dir = out_dir / "curves_csv" / label
        write_curve_csv(
            curve_dir / "roc_curve.csv",
            ("false_positive_rate", "true_positive_rate"),
            curve.roc_fpr,
            curve.roc_tpr,
            curve.roc_thresholds,
        )
        write_curve_csv(
            curve_dir / "pr_curve.csv",
            ("recall", "precision"),
            curve.pr_recall,
            curve.pr_precision,
            None,
        )

    write_dict_rows(rows, out_dir / "pooled_voxel_metrics.csv")
    plot_curves(curves, out_dir, "pooled voxels")


def discover_patients(dataset_root: Path) -> List[int]:
    patient_ids = []
    for path in sorted(dataset_root.iterdir()):
        if not path.is_dir():
            continue
        patient_id = parse_last_integer(path.name)
        if patient_id is not None:
            patient_ids.append(patient_id)
    return sorted(set(patient_ids))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate relative perfusion maps for OT lesion discrimination.")

    parser.add_argument("--dataset-root", type=Path, required=True, help="Root directory containing patient perfusion data.")
    parser.add_argument(
        "--fit-root",
        type=Path,
        default=None,
        help="Root directory containing HemoPIC outputs (training_<ID>/). Required unless --methods deconv only.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for evaluation outputs.")

    parser.add_argument("--patients", type=int, nargs="*", default=None, help="Patient IDs to evaluate.")
    parser.add_argument("--all", action="store_true", help="Evaluate all patient directories found under dataset-root.")

    parser.add_argument(
        "--metrics",
        nargs="+",
        metavar="NAME",
        default=None,
        help="Maps to evaluate (CBF CBV MTT). Default: all maps.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        metavar="M",
        default=None,
        choices=("deconv", "hemopic"),
        help="Which pipelines to run (deconv hemopic). Default: both.",
    )

    parser.add_argument("--dataset-patient-pattern", default="training_{patient}", help="Patient subdirectory pattern under dataset-root.")
    parser.add_argument("--fit-patient-pattern", default="training_{patient}", help="Patient subdirectory pattern under fit-root.")

    parser.add_argument("--deconv-cbf-name", default="MR_rCBF", help="Filename or stem for ISLES silver-standard CBF map.")
    parser.add_argument("--deconv-cbv-name", default="MR_rCBV", help="Filename or stem for ISLES silver-standard CBV map.")
    parser.add_argument("--deconv-mtt-name", default="MR_MTT", help="Filename or stem for ISLES silver-standard MTT map.")

    parser.add_argument("--hemopic-cbf-name", default="HemoPIC_CBF", help="Filename or stem for HemoPIC CBF map.")
    parser.add_argument("--hemopic-cbv-name", default="HemoPIC_CBV", help="Filename or stem for HemoPIC CBV map.")
    parser.add_argument("--hemopic-mtt-name", default="HemoPIC_MTT", help="Filename or stem for HemoPIC MTT map.")

    parser.add_argument("--lesion-mask-name", default="OT", help="Filename or stem for the lesion mask.")
    parser.add_argument("--brain-mask-name", default=None, help="Optional filename or stem for an explicit brain mask.")
    parser.add_argument(
        "--segmentation-relative-path",
        default="hemopic_seg4/seg4_label_ref24.nii.gz",
        help="Segmentation path under each patient directory; used as brain mask when available.",
    )
    parser.add_argument(
        "--relative-mode",
        choices=("brain_median", "nonlesion_median"),
        default="brain_median",
        help="brain_median avoids using lesion labels for normalization; nonlesion_median is retrospective.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_hemopic = args.methods is None or "hemopic" in args.methods
    if run_hemopic and args.fit_root is None:
        raise RuntimeError("Provide --fit-root when evaluating hemopic, or when running both methods (default).")

    if args.all:
        patient_ids = discover_patients(args.dataset_root)
    elif args.patients:
        patient_ids = [int(x) for x in args.patients]
    else:
        raise RuntimeError("Use --all or --patients.")

    if not patient_ids:
        raise RuntimeError("No patients found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    pooled_entries: List[Tuple[str, np.ndarray, np.ndarray]] = []
    failures: List[str] = []

    for patient_id in patient_ids:
        try:
            print(f"Evaluating patient {patient_id}")
            rows, entries = evaluate_patient(patient_id, args)
            all_rows.extend(rows)
            pooled_entries.extend(entries)
        except Exception as exc:  # keep batch evaluation running
            message = f"patient {patient_id}: {type(exc).__name__}: {exc}"
            print(f"[FAIL] {message}")
            failures.append(message)

    write_dict_rows(all_rows, args.out_dir / "all_patient_metrics.csv")
    write_dict_rows(summarize_patient_metrics(all_rows), args.out_dir / "summary_patient_mean_metrics.csv")
    save_pooled_results(pooled_entries, args.out_dir / "pooled_voxel_results")

    with open(args.out_dir / "failures.txt", "w", encoding="utf8") as f:
        for message in failures:
            f.write(message + "\n")

    print("Done")
    print(f"Output: {args.out_dir}")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
