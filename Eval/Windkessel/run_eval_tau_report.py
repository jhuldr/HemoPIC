from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nibabel as nib

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from HemoPIC.io.paths import list_fit_patient_ids, resolve_fit_dir, training_subject_id

'''
DATASET_ROOT=/path/to/ISLES2017_Training
FIT_ROOT=/path/to/HemoPIC_outputs
OUT_DIR=/path/to/HemoPIC_eval

python Eval/Windkessel/run_eval_tau_report.py "$DATASET_ROOT" "$FIT_ROOT" "$OUT_DIR"
'''
# ============== Global constants: time grid ============== #
T_MAX = 40.0
N_T = 401
T_GRID = np.linspace(0.0, T_MAX, N_T)

# ============== Global constants: quantiles and calibration target ============== #
Q_LO = 0.40
Q_MED = 0.50
Q_HI = 0.60
TARGET_END = 0.30

# ============== Global constants: plot colors ============== #
COLOR_GM = "tab:red"
COLOR_WM = "tab:blue"
COLOR_LESION = "tab:purple"
COLOR_C_LESION = "tab:green"

# ============== Data container: one ROI record ============== #
@dataclass
class Row:
    '''
    one ROI level record used for cohort aggregation
    '''
    patient: int
    roi_id: int
    tissue: str
    tau: float
    w: float
    region: str = ""


# ============== logging helper ==============
def info(msg: str) -> None:
    print("[INFO] " + str(msg))


# ============== Path helper: find nii or nii gz ============== #
def find_nii(base: Path, stem: str) -> Optional[Path]:
    p1 = base / (stem + ".nii.gz")
    if p1.exists():
        return p1
    p2 = base / (stem + ".nii")
    if p2.exists():
        return p2
    return None


def patient_dir(dataset_root: Path, pid: int) -> Path:
    return dataset_root / training_subject_id(int(pid))


### Resolve fit global dir and support new and old layout
def resolve_fit_global_dir(fit_root: Path, pid: int) -> Optional[Path]:
    return resolve_fit_dir(fit_root, pid)


def list_patient_ids(fit_root: Path) -> List[int]:
    return list_fit_patient_ids(fit_root)


### Weighted quantile with linear interpolation
def weighted_quantile(values: np.ndarray, weights: np.ndarray, qs: Sequence[float]) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    m = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    v = v[m]
    w = w[m]

    if int(v.size) < 1:
        return np.full(len(list(qs)), np.nan, dtype=float)

    idx = np.argsort(v)
    v = v[idx]
    w = w[idx]

    cw = np.cumsum(w)
    cw = cw / float(cw[-1])
    q = np.asarray(list(qs), dtype=float)
    return np.interp(q, cw, v)


### Exponential curve exp( - gamma t / tau )
def exp_curve(t: np.ndarray, tau: float, gamma: float) -> np.ndarray:
    if (not np.isfinite(tau)) or tau <= 0.0:
        return np.full_like(t, np.nan, dtype=float)
    if (not np.isfinite(gamma)) or gamma <= 0.0:
        return np.full_like(t, np.nan, dtype=float)
    return np.exp(-(gamma * t) / tau)


### Find a reasonable weight column name
def choose_weight_key(fieldnames: Sequence[str]) -> Optional[str]:
    for c in ["n_vox_fit", "n_voxels_fit", "n_vox", "n_voxels_all", "n_vox_all"]:
        if c in fieldnames:
            return c
    return None


### Parse bool from csv cell
def parse_bool(x) -> bool:
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ["1", "true", "t", "yes", "y"]


### Load roi fit summary from csv
def load_roi_fit_summary(csv_path: Path) -> Tuple[List[Dict[str, object]], Optional[str], Optional[str], Optional[str]]:
    with open(csv_path, "r", encoding="utf8") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames is None:
            raise RuntimeError("empty header in " + str(csv_path))

        fields = [str(x) for x in rd.fieldnames]
        wkey = choose_weight_key(fields)

        tau_key = None
        if "tau_fit" in fields:
            tau_key = "tau_fit"
        elif "tau" in fields:
            tau_key = "tau"
        else:
            raise RuntimeError("missing tau column in " + str(csv_path))

        rows: List[Dict[str, object]] = []
        for r in rd:
            rows.append(r)

    return rows, tau_key, wkey, ("success" if "success" in (fields or []) else None)


### Load nifti as bool mask
def load_nifti_bool(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata()
    return (data > 0.5).astype(np.bool_)


### Load nifti as int label map
def load_nifti_int(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata()
    return np.rint(data).astype(np.int32)


### Find label map in a fit global dir
def find_labelmap(global_dir: Path) -> Optional[Path]:
    cand = [
        global_dir / "roi_labels_seg4_within_kmeans.nii.gz",
        global_dir / "roi_labels_seg4_within_kmeans.nii",
        global_dir / "roi_labels.nii.gz",
        global_dir / "roi_labels.nii",
        global_dir / "roi_id_map.nii.gz",
        global_dir / "roi_id_map.nii",
        global_dir / "brain_roi_labels.nii.gz",
        global_dir / "brain_roi_labels.nii",
    ]
    for p in cand:
        if p.exists():
            return p

    parent = global_dir.parent
    cand2 = [
        parent / "roi_labels_seg4_within_kmeans.nii.gz",
        parent / "roi_labels_seg4_within_kmeans.nii",
        parent / "roi_labels.nii.gz",
        parent / "roi_labels.nii",
        parent / "roi_id_map.nii.gz",
        parent / "roi_id_map.nii",
    ]
    for p in cand2:
        if p.exists():
            return p

    return None


### Collect gm wm rows and lesion overlap rows
def collect_rows(dataset_root: Path, fit_root: Path) -> Tuple[List[Row], List[Row]]:
    gmwm_rows: List[Row] = []
    les_rows: List[Row] = []

    patient_ids = list_patient_ids(fit_root)

    for pid in patient_ids:
        gdir = resolve_fit_global_dir(fit_root, pid)
        if gdir is None:
            continue

        csv_path = gdir / "roi_fit_summary.csv"
        if not csv_path.exists():
            continue

        rows, tau_key, wkey, success_key = load_roi_fit_summary(csv_path)

        tau_by_roi: Dict[int, float] = {}
        tissue_by_roi: Dict[int, str] = {}
        w_by_roi: Dict[int, float] = {}

        for r in rows:
            if success_key is not None:
                if not parse_bool(r.get(success_key, "")):
                    continue

            rid_raw = r.get("roi_id", None)
            tis = str(r.get("tissue", "")).strip()
            if rid_raw is None:
                continue

            try:
                rid = int(float(rid_raw))
            except Exception:
                continue

            tau_raw = r.get(tau_key, None)
            try:
                tau = float(tau_raw)
            except Exception:
                continue

            if (not np.isfinite(tau)) or tau <= 0.0:
                continue

            if wkey is None:
                w = 1.0
            else:
                try:
                    w = float(r.get(wkey, 0.0))
                except Exception:
                    w = 0.0

            if (not np.isfinite(w)) or w <= 0.0:
                w = 0.0

            if rid not in tau_by_roi:
                tau_by_roi[rid] = float(tau)
                tissue_by_roi[rid] = str(tis)
                w_by_roi[rid] = float(w)

            if tis in ["GM", "WM"] and w > 0.0:
                gmwm_rows.append(
                    Row(
                        patient=int(pid),
                        roi_id=int(rid),
                        tissue=str(tis),
                        tau=float(tau),
                        w=float(w),
                        region="",
                    )
                )

        ot_path = find_nii(patient_dir(dataset_root, pid), "OT")
        if ot_path is None:
            continue

        label_path = find_labelmap(gdir)
        if label_path is None:
            continue

        ot = load_nifti_bool(ot_path)
        labels = load_nifti_int(label_path)

        if ot.shape != labels.shape:
            continue

        brain = labels > 0
        lesion = ot & brain
        c_lesion = np.flip(lesion, axis=0) & brain

        lab_les = labels[lesion]
        lab_c = labels[c_lesion]

        max_lab = int(max(int(labels.max()), 1))
        counts_les = np.bincount(lab_les[lab_les > 0].astype(np.int64), minlength=int(max_lab + 1)) if lab_les.size > 0 else np.zeros(int(max_lab + 1), dtype=np.int64)
        counts_c = np.bincount(lab_c[lab_c > 0].astype(np.int64), minlength=int(max_lab + 1)) if lab_c.size > 0 else np.zeros(int(max_lab + 1), dtype=np.int64)

        def count_for(counts: np.ndarray, rid: int) -> int:
            if rid < 0:
                return 0
            if rid >= int(counts.size):
                return 0
            return int(counts[int(rid)])

        for rid, tau in tau_by_roi.items():
            w_les = count_for(counts_les, int(rid))
            w_c = count_for(counts_c, int(rid))
            tis = tissue_by_roi.get(int(rid), "")
            if int(w_les) > 0:
                les_rows.append(
                    Row(
                        patient=int(pid),
                        roi_id=int(rid),
                        tissue=str(tis),
                        tau=float(tau),
                        w=float(w_les),
                        region="lesion",
                    )
                )
            if int(w_c) > 0:
                les_rows.append(
                    Row(
                        patient=int(pid),
                        roi_id=int(rid),
                        tissue=str(tis),
                        tau=float(tau),
                        w=float(w_c),
                        region="c-lesion",
                    )
                )

    return gmwm_rows, les_rows


### Compute shared gamma so that exp( -gamma T_MAX / tau_med ) = TARGET_END
def compute_gamma_from_gmwm(gmwm_rows: List[Row]) -> float:
    tau = np.asarray([r.tau for r in gmwm_rows], dtype=float)
    w = np.asarray([r.w for r in gmwm_rows], dtype=float)
    tau_med = float(weighted_quantile(tau, w, [Q_MED])[0])

    if (not np.isfinite(tau_med)) or tau_med <= 0.0:
        return 1.0

    gamma = (-math.log(float(TARGET_END))) * float(tau_med) / float(T_MAX)
    if (not np.isfinite(gamma)) or gamma <= 0.0:
        return 1.0
    return float(gamma)


### Plot one group as band q40 q60 and thick q50
def plot_group(ax, tau: np.ndarray, w: np.ndarray, label: str, color: str, gamma: float) -> Dict[str, float]:
    q = weighted_quantile(tau, w, [Q_LO, Q_MED, Q_HI])
    qlo, qmed, qhi = float(q[0]), float(q[1]), float(q[2])

    y_lo = exp_curve(T_GRID, qlo, gamma)
    y_med = exp_curve(T_GRID, qmed, gamma)
    y_hi = exp_curve(T_GRID, qhi, gamma)

    ax.fill_between(T_GRID, y_lo, y_hi, color=color, alpha=0.20, linewidth=0.0)
    ax.plot(T_GRID, y_med, color=color, linewidth=4.0, label=label)

    ax.set_xlim(0.0, float(T_MAX))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("t seconds")
    ax.set_ylabel("normalized deviation")
    ax.legend(loc="upper right", frameon=True)

    return {"q40": qlo, "q50": qmed, "q60": qhi}


### Write csv outputs
def write_csv_outputs(out_dir: Path, gmwm_rows: List[Row], les_rows: List[Row], gamma: float) -> None:
    out1 = out_dir / "cohort_roi_tau_values.csv"
    with open(out1, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["patient", "roi_id", "tissue", "tau", "w"])
        for r in gmwm_rows:
            w.writerow([int(r.patient), int(r.roi_id), str(r.tissue), float(r.tau), float(r.w)])

    out2 = out_dir / "cohort_lesion_overlap_roi_tau_values.csv"
    with open(out2, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["patient", "region", "roi_id", "tissue", "tau", "w"])
        for r in les_rows:
            w.writerow([int(r.patient), str(r.region), int(r.roi_id), str(r.tissue), float(r.tau), float(r.w)])

    def summarize_group(name: str, tau: np.ndarray, w: np.ndarray) -> Dict[str, object]:
        q = weighted_quantile(tau, w, [0.25, Q_LO, Q_MED, Q_HI, 0.75])
        w_sum = float(np.sum(w)) if int(w.size) > 0 else 0.0
        wmean = float(np.average(tau, weights=w)) if (int(tau.size) > 0 and w_sum > 0.0) else float("nan")
        return dict(
            group=str(name),
            n=int(tau.size),
            w_sum=float(w_sum),
            q25=float(q[0]),
            q40=float(q[1]),
            q50=float(q[2]),
            q60=float(q[3]),
            q75=float(q[4]),
            wmean=float(wmean),
        )

    rows: List[Dict[str, object]] = []

    gm_tau = np.asarray([r.tau for r in gmwm_rows if str(r.tissue) == "GM"], dtype=float)
    gm_w = np.asarray([r.w for r in gmwm_rows if str(r.tissue) == "GM"], dtype=float)
    wm_tau = np.asarray([r.tau for r in gmwm_rows if str(r.tissue) == "WM"], dtype=float)
    wm_w = np.asarray([r.w for r in gmwm_rows if str(r.tissue) == "WM"], dtype=float)

    if int(gm_tau.size) > 0:
        rows.append(summarize_group("GM", gm_tau, gm_w))
    if int(wm_tau.size) > 0:
        rows.append(summarize_group("WM", wm_tau, wm_w))

    les_tau = np.asarray([r.tau for r in les_rows if str(r.region) == "lesion"], dtype=float)
    les_w = np.asarray([r.w for r in les_rows if str(r.region) == "lesion"], dtype=float)
    cl_tau = np.asarray([r.tau for r in les_rows if str(r.region) == "c-lesion"], dtype=float)
    cl_w = np.asarray([r.w for r in les_rows if str(r.region) == "c-lesion"], dtype=float)

    if int(les_tau.size) > 0:
        rows.append(summarize_group("lesion", les_tau, les_w))
    if int(cl_tau.size) > 0:
        rows.append(summarize_group("c-lesion", cl_tau, cl_w))

    out3 = out_dir / "tau_quantiles_summary.csv"
    with open(out3, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["group", "n", "w_sum", "q25", "q40", "q50", "q60", "q75", "wmean", "gamma", "t_max", "target_end"])
        for r in rows:
            w.writerow([
                str(r["group"]),
                int(r["n"]),
                float(r["w_sum"]),
                float(r["q25"]),
                float(r["q40"]),
                float(r["q50"]),
                float(r["q60"]),
                float(r["q75"]),
                float(r["wmean"]),
                float(gamma),
                float(T_MAX),
                float(TARGET_END),
            ])


### Write figures
def write_figures(out_dir: Path, gmwm_rows: List[Row], les_rows: List[Row], gamma: float) -> None:
    gm_tau = np.asarray([r.tau for r in gmwm_rows if str(r.tissue) == "GM"], dtype=float)
    gm_w = np.asarray([r.w for r in gmwm_rows if str(r.tissue) == "GM"], dtype=float)
    wm_tau = np.asarray([r.tau for r in gmwm_rows if str(r.tissue) == "WM"], dtype=float)
    wm_w = np.asarray([r.w for r in gmwm_rows if str(r.tissue) == "WM"], dtype=float)

    les_tau = np.asarray([r.tau for r in les_rows if str(r.region) == "lesion"], dtype=float)
    les_w = np.asarray([r.w for r in les_rows if str(r.region) == "lesion"], dtype=float)
    cl_tau = np.asarray([r.tau for r in les_rows if str(r.region) == "c-lesion"], dtype=float)
    cl_w = np.asarray([r.w for r in les_rows if str(r.region) == "c-lesion"], dtype=float)

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })

    fig1, ax1 = plt.subplots(figsize=(8.0, 5.0), dpi=200)
    if int(gm_tau.size) > 0:
        plot_group(ax1, gm_tau, gm_w, "GM", COLOR_GM, gamma)
    if int(wm_tau.size) > 0:
        plot_group(ax1, wm_tau, wm_w, "WM", COLOR_WM, gamma)
    fig1.tight_layout(pad=0.4)
    fig1.savefig(out_dir / "cohort_gm_wm_exp_40s.png")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8.0, 5.0), dpi=200)
    if int(les_tau.size) > 0:
        plot_group(ax2, les_tau, les_w, "lesion", COLOR_LESION, gamma)
    if int(cl_tau.size) > 0:
        plot_group(ax2, cl_tau, cl_w, "c-lesion", COLOR_C_LESION, gamma)
    fig2.tight_layout(pad=0.4)
    fig2.savefig(out_dir / "cohort_lesion_c_lesion_exp_40s.png")
    plt.close(fig2)

    fig3, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=200)
    fig3.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.98, wspace=0.15)

    if int(gm_tau.size) > 0:
        plot_group(axL, gm_tau, gm_w, "GM", COLOR_GM, gamma)
    if int(wm_tau.size) > 0:
        plot_group(axL, wm_tau, wm_w, "WM", COLOR_WM, gamma)

    if int(les_tau.size) > 0:
        plot_group(axR, les_tau, les_w, "lesion", COLOR_LESION, gamma)
    if int(cl_tau.size) > 0:
        plot_group(axR, cl_tau, cl_w, "c-lesion", COLOR_C_LESION, gamma)

    fig3.savefig(out_dir / "cohort_exp_40s_side_by_side.png")
    plt.close(fig3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("fit_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    fit_root = args.fit_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gmwm_rows, les_rows = collect_rows(dataset_root, fit_root)
    info("gmwm_rows " + str(len(gmwm_rows)))
    info("lesion_rows " + str(len(les_rows)))

    gamma = compute_gamma_from_gmwm(gmwm_rows)
    info("gamma " + str(gamma))

    write_csv_outputs(out_dir, gmwm_rows, les_rows, gamma)
    write_figures(out_dir, gmwm_rows, les_rows, gamma)

    info("done out_dir " + str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())