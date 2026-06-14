import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from HemoPIC.io.paths import list_fit_patient_ids, resolve_fit_dir as locate_fit_dir, hemopic_seg4_label_path, pick_hemopic_map, training_subject_id

'''
DATASET_ROOT=/path/to/ISLES2017_Training
FIT_ROOT=/path/to/HemoPIC_outputs
OUT_DIR=/path/to/HemoPIC_eval

python Eval/cvt_check/run_eval_cvt.py "$DATASET_ROOT" "$FIT_ROOT" "$OUT_DIR"
'''


# ======================== Imports and constants ======================== #
CVT_SCALE = 60.0

COLOR_BRAIN = "#9C755F"
COLOR_GM = "#59A14F"
COLOR_WM = "#4E79A7"
COLOR_LESION = "#E15759"

X_TICKS = np.asarray([0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0], dtype=np.float64)
Y_TICKS = np.asarray([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], dtype=np.float64)

X_LIM = (0.0, 15.0)
Y_LIM = (0.0, 16.0)

X_LABEL = "Deconv MTT"
Y_LABEL = r"HemoPIC $60 \cdot \mathrm{CBV}/\mathrm{CBF}$"


# ======================== File system and NIfTI IO helpers ======================== #
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_nifti(path):
    img = nib.load(str(path))
    data = np.asanyarray(img.get_fdata())
    return data

def find_nii(base, stem):
    ''' Find nii gz or nii '''
    base = Path(base)

    p1 = base / (str(stem) + ".nii.gz")
    if p1.exists():
        return p1

    p2 = base / (str(stem) + ".nii")
    if p2.exists():
        return p2

    return None

def pick_first(paths):
    ''' Pick first existing path '''
    for p in list(paths):
        if p is None:
            continue
        pp = Path(p)
        if pp.exists():
            return pp
    return None


# ======================== Fit output discovery helpers ======================== #
def pick_model_map(fit_dir, tag):
    return pick_hemopic_map(fit_dir, tag)


# ======================== Segmentation and mask utilities ======================== #
def finite_pos_mask(a):
    x = a.astype(np.float64)
    return np.isfinite(x) & (x > 0.0)

def seg4_masks(seg4):
    s = seg4.astype(np.uint8)
    inside = s > 0
    gm = s == 1
    wm = s == 2
    csf = s == 3
    return inside, gm, wm, csf

# ======================== Contra lesion construction utilities ======================== #
def mirror_mask(mask3, axis):
    return np.flip(mask3, axis=int(axis))

def choose_mirror_axis(lesion3, brain3):
    best_axis = 0
    best_in = None
    best_overlap = None

    for ax in [0, 1]:
        m = mirror_mask(lesion3, ax)
        in_brain = int(np.sum(m & brain3))
        overlap = int(np.sum(m & lesion3))

        if best_in is None:
            best_axis = int(ax)
            best_in = int(in_brain)
            best_overlap = int(overlap)
            continue

        if int(in_brain) > int(best_in):
            best_axis = int(ax)
            best_in = int(in_brain)
            best_overlap = int(overlap)
            continue

        if int(in_brain) == int(best_in) and int(overlap) < int(best_overlap):
            best_axis = int(ax)
            best_in = int(in_brain)
            best_overlap = int(overlap)

    return int(best_axis)

def build_c_lesion(lesion3, brain3, mirror_axis):
    m = mirror_mask(lesion3, mirror_axis)
    return m & brain3 & (np.logical_not(lesion3))


# ======================== Statistics core ======================== #
def vec_clean(v):
    a = np.asarray(v, dtype=np.float64)
    a = a[np.isfinite(a)]
    return a

def linear_fit_params(x, y):
    x = vec_clean(x)
    y = vec_clean(y)
    n = int(x.size)
    if n < 3 or int(y.size) < 3:
        return None

    xm = float(np.mean(x))
    ym = float(np.mean(y))
    xc = np.subtract(x, xm)
    yc = np.subtract(y, ym)

    sxx = float(np.sum(np.multiply(xc, xc)))
    if sxx == 0.0:
        return None

    sxy = float(np.sum(np.multiply(xc, yc)))
    slope = float(np.divide(sxy, sxx))
    intercept = float(np.subtract(ym, np.multiply(slope, xm)))

    yhat = np.add(np.multiply(slope, x), intercept)
    resid = np.subtract(y, yhat)

    ss_res = float(np.sum(np.multiply(resid, resid)))
    dof = int(np.subtract(n, 2))
    if dof <= 0:
        return None

    sigma2 = float(np.divide(ss_res, float(dof)))
    sigma = float(np.sqrt(sigma2))

    return dict(slope=slope, intercept=intercept, n=n, xm=xm, sxx=sxx, sigma=sigma)

def regression_band(fit, xs, z_value):
    n = int(fit["n"])
    xm = float(fit["xm"])
    sxx = float(fit["sxx"])
    sigma = float(fit["sigma"])

    dx = np.subtract(xs, xm)
    term1 = np.divide(1.0, float(n))
    term2 = np.divide(np.multiply(dx, dx), float(sxx))
    term = np.add(term1, term2)

    se = np.multiply(sigma, np.sqrt(term))

    ys = np.add(np.multiply(float(fit["slope"]), xs), float(fit["intercept"]))
    half = np.multiply(float(z_value), se)

    lo = np.subtract(ys, half)
    hi = np.add(ys, half)
    return ys, lo, hi

# ======================== Plot helpers ======================== #
def apply_fixed_axes(ax):
    ax.set_xlim(float(X_LIM[0]), float(X_LIM[1]))
    ax.set_ylim(float(Y_LIM[0]), float(Y_LIM[1]))
    ax.set_xticks([float(v) for v in X_TICKS.tolist()])
    ax.set_yticks([float(v) for v in Y_TICKS.tolist()])

def plot_scatter(ax, x, y, color, title):
    x = vec_clean(x)
    y = vec_clean(y)

    ax.scatter(x, y, s=18, alpha=0.80, c=color, marker="o", edgecolors="none")
    ax.set_title(str(title), fontsize=15)
    ax.set_xlabel(X_LABEL, fontsize=15)
    ax.set_ylabel(Y_LABEL, fontsize=15)
    ax.tick_params(axis="both", labelsize=15)

    apply_fixed_axes(ax)

    fit = linear_fit_params(x, y)
    if fit is None:
        return

    xs = np.linspace(float(X_LIM[0]), float(X_LIM[1]), 220, dtype=np.float64)
    ys, lo, hi = regression_band(fit, xs, z_value=1.0)

    ax.fill_between(xs, lo, hi, alpha=0.18, color=color, linewidth=0.0)
    ax.plot(xs, ys, linewidth=2.2, color=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("fit_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    fit_root = args.fit_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    base = out_dir / "cvt"
    fig_dir = base / "figures"
    tab_dir = base / "tables"
    ensure_dir(fig_dir)
    ensure_dir(tab_dir)

    patient_ids = list_fit_patient_ids(fit_root)

    rows = []

    for pid in patient_ids:
        patient_folder = dataset_root / training_subject_id(int(pid))
        fit_dir = locate_fit_dir(fit_root, int(pid))

        if (not patient_folder.exists()) or fit_dir is None:
            continue

        seg_path = hemopic_seg4_label_path(patient_folder)
        ot_path = find_nii(patient_folder, "OT")

        if (not seg_path.exists()) or ot_path is None:
            continue

        seg4 = load_nifti(seg_path)
        ot3 = load_nifti(Path(ot_path))

        if int(seg4.ndim) != 3 or int(ot3.ndim) != 3:
            continue
        if tuple(seg4.shape[:3]) != tuple(ot3.shape[:3]):
            continue

        inside, gm3, wm3, csf3 = seg4_masks(seg4)
        lesion3 = (ot3 > 0).astype(bool)
        brain_base = inside & (np.logical_not(csf3))

        mirror_axis = choose_mirror_axis(lesion3, brain_base)
        c_lesion3 = build_c_lesion(lesion3, brain_base, mirror_axis)

        model_cbf_p = pick_model_map(fit_dir, "CBF")
        model_cbv_p = pick_model_map(fit_dir, "CBV")
        data_mtt_p = find_nii(patient_folder, "MR_MTT")

        data_cbf_p = find_nii(patient_folder, "MR_rCBF")
        data_cbv_p = find_nii(patient_folder, "MR_rCBV")

        if model_cbf_p is None or model_cbv_p is None:
            continue
        if data_mtt_p is None or data_cbf_p is None or data_cbv_p is None:
            continue

        model_cbf = load_nifti(Path(model_cbf_p)).astype(np.float64)
        model_cbv = load_nifti(Path(model_cbv_p)).astype(np.float64)

        data_mtt = load_nifti(Path(data_mtt_p)).astype(np.float64)
        data_cbf = load_nifti(Path(data_cbf_p)).astype(np.float64)
        data_cbv = load_nifti(Path(data_cbv_p)).astype(np.float64)

        if tuple(model_cbf.shape[:3]) != tuple(seg4.shape[:3]):
            continue
        if tuple(model_cbv.shape[:3]) != tuple(seg4.shape[:3]):
            continue
        if tuple(data_mtt.shape[:3]) != tuple(seg4.shape[:3]):
            continue
        if tuple(data_cbf.shape[:3]) != tuple(seg4.shape[:3]):
            continue
        if tuple(data_cbv.shape[:3]) != tuple(seg4.shape[:3]):
            continue

        regions = [
            ("GM", gm3, 50),
            ("WM", wm3, 50),
            ("Lesion", lesion3, 20),
            ("Brain", brain_base, 50),
        ]

        for region_name, region_mask, min_n in regions:
            valid_data = brain_base & finite_pos_mask(data_mtt) & finite_pos_mask(data_cbf) & finite_pos_mask(data_cbv)
            valid_model = brain_base & finite_pos_mask(model_cbf) & finite_pos_mask(model_cbv)

            m_data = valid_data & region_mask
            m_model = valid_model & region_mask

            n_data = int(np.sum(m_data))
            n_model = int(np.sum(m_model))

            if n_data < int(min_n) or n_model < int(min_n):
                continue

            mean_mtt = float(np.mean(data_mtt[m_data]))
            mean_cbf = float(np.mean(model_cbf[m_model]))
            mean_cbv = float(np.mean(model_cbv[m_model]))

            if (not np.isfinite(mean_mtt)) or (not np.isfinite(mean_cbf)) or (not np.isfinite(mean_cbv)):
                continue
            if mean_cbf <= 0.0:
                continue

            y = float(np.multiply(CVT_SCALE, float(np.divide(mean_cbv, mean_cbf))))

            rows.append(dict(
                patient=int(pid),
                region=str(region_name),
                x_data_mtt=mean_mtt,
                y_model_ratio=y,
                n_vox=int(min(n_data, n_model)),
            ))

    out_csv = tab_dir / "points.csv"
    with open(out_csv, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["patient", "region", "x_data_mtt", "y_model_ratio", "n_vox"])
        for r in rows:
            w.writerow([int(r["patient"]), str(r["region"]), r["x_data_mtt"], r["y_model_ratio"], int(r["n_vox"])])

    x_gm = [float(r["x_data_mtt"]) for r in rows if str(r["region"]) == "GM"]
    y_gm = [float(r["y_model_ratio"]) for r in rows if str(r["region"]) == "GM"]

    x_wm = [float(r["x_data_mtt"]) for r in rows if str(r["region"]) == "WM"]
    y_wm = [float(r["y_model_ratio"]) for r in rows if str(r["region"]) == "WM"]

    x_ls = [float(r["x_data_mtt"]) for r in rows if str(r["region"]) == "Lesion"]
    y_ls = [float(r["y_model_ratio"]) for r in rows if str(r["region"]) == "Lesion"]

    x_br = [float(r["x_data_mtt"]) for r in rows if str(r["region"]) == "Brain"]
    y_br = [float(r["y_model_ratio"]) for r in rows if str(r["region"]) == "Brain"]

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.1))

    plot_scatter(axes[0], np.asarray(x_gm), np.asarray(y_gm), COLOR_GM, "Gray Matter")
    plot_scatter(axes[1], np.asarray(x_wm), np.asarray(y_wm), COLOR_WM, "White Matter")
    plot_scatter(axes[2], np.asarray(x_ls), np.asarray(y_ls), COLOR_LESION, "Lesion")
    plot_scatter(axes[3], np.asarray(x_br), np.asarray(y_br), COLOR_BRAIN, "Full Brain")

    fig.tight_layout(pad=0.18, w_pad=0.22)
    fig.savefig(str(fig_dir / "row.png"), dpi=240, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print("out_dir", str(base))


if __name__ == "__main__":
    main()