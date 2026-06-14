import argparse
import csv
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

'''
Run:
DATASET_ROOT=/path/to/ISLES2017_Training
FIT_ROOT=/path/to/HemoPIC_outputs
OUT_DIR=/path/to/HemoPIC_eval

python3 "/path/to/run_eval_summary_stats.py" "$DATASET_ROOT" "$FIT_ROOT" "$REPORT_ROOT"
'''

# ============== plot colors, constants ============== #
COLOR_MU = "#F28E2B"
COLOR_SIGMA = "#4E79A7"
COLOR_T = "#E15759"


# ============== IO helper: create directory and nifti  ============== #
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_nifti(path: Path):
    img = nib.load(str(path))
    data = np.asanyarray(img.get_fdata())
    return data

def find_nii(base: Path, stem: str):
    p1 = base / (str(stem) + ".nii.gz")
    if p1.exists():
        return p1
    p2 = base / (str(stem) + ".nii")
    if p2.exists():
        return p2
    return None

def pick_first(paths):
    for p in list(paths):
        if p is None:
            continue
        pp = Path(p)
        if pp.exists():
            return pp
    return None


# ============== Cohort helper: list patient ids from fit_root ============== #
def list_patient_ids(fit_root: Path):
    out = []
    for d in sorted(Path(fit_root).glob("p*")):
        if not d.is_dir():
            continue
        name = str(d.name)
        if not name.startswith("p"):
            continue
        digits = []
        for ch in name[1:]:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if int(len(digits)) < 1:
            continue
        out.append(int("".join(digits)))
    return sorted(list(set(out)))

def resolve_fit_global_dir(fit_root: Path, pid: int):
    '''
    Resolve the per patient fit directory.
    '''
    pid_str = "p" + str(int(pid))
    cand = [
        Path(fit_root) / pid_str / "global" / "core",
        Path(fit_root) / pid_str / "global",
        Path(fit_root) / (pid_str + "_kmeans") / "global" / "core",
        Path(fit_root) / (pid_str + "_kmeans") / "global",
    ]
    for d in cand:
        if d.exists():
            return d
    return None

def pick_model_map(fit_dir: Path, tag: str):
    '''
    Pick a fitted model map for a tag such as CBF or CBV ith common stems
    '''
    tag_u = str(tag).upper()
    cand = [
        fit_dir / ("Model_" + tag_u + "_raw_clamped.nii.gz"),
        fit_dir / ("Model_" + tag_u + "_raw.nii.gz"),
        fit_dir / ("Model_" + tag_u + "_scaled.nii.gz"),
        fit_dir / ("Model_" + tag_u + "_raw_clamped.nii"),
        fit_dir / ("Model_" + tag_u + "_raw.nii"),
        fit_dir / ("Model_" + tag_u + "_scaled.nii"),
    ]
    p = pick_first(cand)
    if p is not None:
        return p

    found = sorted(list(fit_dir.glob("*" + tag_u + "*.nii*")))
    found = [q for q in found if q.is_file()]
    if int(len(found)) < 1:
        return None
    return found[0]

# ============== Segmentation and lesion construction ============== #
def seg4_masks(seg4):
    s = seg4.astype(np.uint8)
    inside = s > 0
    gm = s == 1
    wm = s == 2
    csf = s == 3
    return inside, gm, wm, csf

def finite_pos_mask(a):
    x = a.astype(np.float64)
    return np.isfinite(x) & (x > 0.0)

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
    return m & brain3 & (~lesion3)

def safe_ratio(a, b):
    fa = float(a)
    fb = float(b)
    if (not np.isfinite(fa)) or (not np.isfinite(fb)):
        return None
    if fb == 0.0:
        return None
    return float(np.divide(fa, fb))



# ============== Extract ROI samples and compute metrics ============== #
def compute_pair_metrics(map3, mask_a3, mask_b3, valid3, min_voxels):
    a = map3.astype(np.float64)

    ma = mask_a3 & valid3 & np.isfinite(a)
    mb = mask_b3 & valid3 & np.isfinite(a)

    x = a[ma]
    y = a[mb]

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    out = dict()
    out["n_a"] = int(x.size)
    out["n_b"] = int(y.size)

    if int(x.size) < int(min_voxels) or int(y.size) < int(min_voxels):
        out["ok"] = False
        return out

    mx = float(np.mean(x))
    my = float(np.mean(y))
    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))

    out["mean_a"] = mx
    out["mean_b"] = my
    out["std_a"] = sx
    out["std_b"] = sy

    out["mu_r"] = safe_ratio(mx, my)
    out["sigma_r"] = safe_ratio(sx, sy)

    t_res = ttest_ind(x, y, equal_var=False, nan_policy="omit")
    stat = float(t_res.statistic) if np.isfinite(t_res.statistic) else None
    pval = float(t_res.pvalue) if np.isfinite(t_res.pvalue) else None

    out["t_value"] = stat
    out["p_value"] = pval
    out["ok"] = True
    return out

def push_if_finite(dst, v):
    if v is None:
        return
    fv = float(v)
    if np.isfinite(fv):
        dst.append(fv)

def set_box_facecolor(bp, facecolor, alpha):
    for box in bp["boxes"]:
        box.set_facecolor(facecolor)
        box.set_alpha(float(alpha))
    for med in bp["medians"]:
        med.set_linewidth(1.6)
        
def plot_relative(ax, labels, mu_list, sigma_list, title):
    n = int(len(labels))
    centers = np.arange(1, int(np.add(n, 1)), dtype=np.float64)
    off = 0.16

    pos_mu = [float(np.subtract(c, off)) for c in centers.tolist()]
    pos_sg = [float(np.add(c, off)) for c in centers.tolist()]

    bp_mu = ax.boxplot(mu_list, positions=pos_mu, widths=0.26, patch_artist=True, showfliers=False, manage_ticks=False)
    bp_sg = ax.boxplot(sigma_list, positions=pos_sg, widths=0.26, patch_artist=True, showfliers=False, manage_ticks=False)

    set_box_facecolor(bp_mu, COLOR_MU, 0.86)
    set_box_facecolor(bp_sg, COLOR_SIGMA, 0.86)

    ax.set_title(str(title), pad=6)
    ax.set_xticks(centers.tolist())
    ax.set_xticklabels([str(x) for x in labels], rotation=0)
    ax.axhline(1.0, linewidth=1.0, color="C0")
    ax.legend([bp_mu["boxes"][0], bp_sg["boxes"][0]], ["mu ratio", "sigma ratio"], loc="upper left")


def plot_t(ax, labels, t_list, title):
    n = int(len(labels))
    centers = np.arange(1, int(np.add(n, 1)), dtype=np.float64)

    bp_t = ax.boxplot(t_list, positions=centers.tolist(), widths=0.30, patch_artist=True, showfliers=False)
    set_box_facecolor(bp_t, COLOR_T, 0.86)

    ax.set_title(str(title), pad=6)
    ax.set_xticks(centers.tolist())
    ax.set_xticklabels([str(x) for x in labels], rotation=0)
    ax.axhline(0.0, linewidth=1.0, color="C0")

def write_rows_csv(out_csv: Path, rows):
    ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["patient", "metric", "mu_r", "sigma_r", "t_value", "p_value", "n_a", "n_b"])
        for r in rows:
            w.writerow([
                int(r["patient"]),
                str(r["metric"]),
                r.get("mu_r", None),
                r.get("sigma_r", None),
                r.get("t_value", None),
                r.get("p_value", None),
                int(r.get("n_a", 0)),
                int(r.get("n_b", 0)),
            ])

# ============== Main: parse arguments ============== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("fit_root", type=Path)
    ap.add_argument("report_root", type=Path)
    ap.add_argument("min_voxels_lesion", nargs="?", type=int, default=20)
    ap.add_argument("min_voxels_gmwm", nargs="?", type=int, default=2000)
    args = ap.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    fit_root = args.fit_root.expanduser().resolve()
    report_root = args.report_root.expanduser().resolve()

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = report_root / "summary_stats" / ("run_" + str(tag))
    fig_dir = run_dir / "figures"
    tab_dir = run_dir / "tables"
    ensure_dir(fig_dir)
    ensure_dir(tab_dir)
    
     # ======= Main: define metrics and tick labels ======= #
    metrics = ["HemoPIC CBF", "HemoPIC CBV", "Deconv CBF", "Deconv CBV"]
    tick_labels = ["HemoPIC\nCBF", "HemoPIC\nCBV", "Deconv\nCBF", "Deconv\nCBV"]

    mu_l = {k: [] for k in metrics}
    sg_l = {k: [] for k in metrics}
    tv_l = {k: [] for k in metrics}
    rows_l = []

    mu_g = {k: [] for k in metrics}
    sg_g = {k: [] for k in metrics}
    tv_g = {k: [] for k in metrics}
    rows_g = []

    skipped = []
    ok_patients = 0

    patient_ids = list_patient_ids(fit_root)
    
     # ======= Main: iterate patients and collect metrics ======= #
    for pid in patient_ids:
        patient_folder = dataset_root / ("training_" + str(int(pid)))
        fit_dir = resolve_fit_global_dir(fit_root, int(pid))

        if not patient_folder.exists():
            skipped.append(dict(patient=int(pid), reason="missing_patient_folder"))
            continue
        if fit_dir is None or (not fit_dir.exists()):
            skipped.append(dict(patient=int(pid), reason="missing_fit_dir"))
            continue
        
         # ======= locate segmentation and lesion label ======= #
        seg_path = patient_folder / "synthseg" / "seg_export_fixed" / "seg4_label_ref24.nii.gz"
        ot_path = find_nii(patient_folder, "OT")

        if not seg_path.exists():
            skipped.append(dict(patient=int(pid), reason="missing_seg4"))
            continue
        if ot_path is None:
            skipped.append(dict(patient=int(pid), reason="missing_ot"))
            continue
        
         # ======= load seg4 and lesion volume, and shape validation ======= #
        seg4 = load_nifti(seg_path)
        ot3 = load_nifti(Path(ot_path))

        if int(seg4.ndim) != 3 or int(ot3.ndim) != 3:
            skipped.append(dict(patient=int(pid), reason="ndim_not_3"))
            continue
        if tuple(seg4.shape[:3]) != tuple(ot3.shape[:3]):
            skipped.append(dict(patient=int(pid), reason="shape_mismatch_seg_ot"))
            continue
        
         # ======= build tissue and brain masks ======= #
        inside, gm3, wm3, csf3 = seg4_masks(seg4)
        lesion3 = (ot3 > 0).astype(bool)
        brain_base = inside & (~csf3)
        
         # ======= locate model and data maps ======= #
        model_cbf = pick_model_map(fit_dir, "CBF")
        model_cbv = pick_model_map(fit_dir, "CBV")

        data_cbf = find_nii(patient_folder, "MR_rCBF")
        data_cbv = find_nii(patient_folder, "MR_rCBV")

        if model_cbf is None or model_cbv is None or data_cbf is None or data_cbv is None:
            skipped.append(dict(patient=int(pid), reason="missing_maps"))
            continue

        m_model_cbf = load_nifti(Path(model_cbf))
        m_model_cbv = load_nifti(Path(model_cbv))
        m_data_cbf = load_nifti(Path(data_cbf))
        m_data_cbv = load_nifti(Path(data_cbv))

        if tuple(m_model_cbf.shape[:3]) != tuple(seg4.shape[:3]):
            skipped.append(dict(patient=int(pid), reason="shape_mismatch_model_cbf"))
            continue
        if tuple(m_model_cbv.shape[:3]) != tuple(seg4.shape[:3]):
            skipped.append(dict(patient=int(pid), reason="shape_mismatch_model_cbv"))
            continue
        if tuple(m_data_cbf.shape[:3]) != tuple(seg4.shape[:3]):
            skipped.append(dict(patient=int(pid), reason="shape_mismatch_data_cbf"))
            continue
        if tuple(m_data_cbv.shape[:3]) != tuple(seg4.shape[:3]):
            skipped.append(dict(patient=int(pid), reason="shape_mismatch_data_cbv"))
            continue

        # ======= Per patient: bundle all map variants ======= #
        maps = {
            "HemoPIC CBF": m_model_cbf,
            "HemoPIC CBV": m_model_cbv,
            "Deconv CBF": m_data_cbf,
            "Deconv CBV": m_data_cbv,
        }

        mirror_axis = choose_mirror_axis(lesion3, brain_base)
        c_lesion3 = build_c_lesion(lesion3, brain_base, mirror_axis)

        ok_patients = int(np.add(ok_patients, 1))

        for k in metrics:
            # ======= define valid voxels for this map (per metric)======= #
            m3 = maps[k]
            valid3 = brain_base & finite_pos_mask(m3)

            met_l = compute_pair_metrics(
                map3=m3,
                mask_a3=lesion3,
                mask_b3=c_lesion3,
                valid3=valid3,
                min_voxels=int(args.min_voxels_lesion),
            )

            if bool(met_l.get("ok", False)):
                push_if_finite(mu_l[k], met_l.get("mu_r", None))
                push_if_finite(sg_l[k], met_l.get("sigma_r", None))
                push_if_finite(tv_l[k], met_l.get("t_value", None))

                rows_l.append(dict(
                    patient=int(pid),
                    metric=str(k),
                    mu_r=met_l.get("mu_r", None),
                    sigma_r=met_l.get("sigma_r", None),
                    t_value=met_l.get("t_value", None),
                    p_value=met_l.get("p_value", None),
                    n_a=int(met_l.get("n_a", 0)),
                    n_b=int(met_l.get("n_b", 0)),
                ))

            gm_ok = gm3 & finite_pos_mask(m3)
            wm_ok = wm3 & finite_pos_mask(m3)

            met_g = compute_pair_metrics(
                map3=m3,
                mask_a3=wm_ok,
                mask_b3=gm_ok,
                valid3=valid3,
                min_voxels=int(args.min_voxels_gmwm),
            )

            if bool(met_g.get("ok", False)):
                push_if_finite(mu_g[k], met_g.get("mu_r", None))
                push_if_finite(sg_g[k], met_g.get("sigma_r", None))
                push_if_finite(tv_g[k], met_g.get("t_value", None))

                rows_g.append(dict(
                    patient=int(pid),
                    metric=str(k),
                    mu_r=met_g.get("mu_r", None),
                    sigma_r=met_g.get("sigma_r", None),
                    t_value=met_g.get("t_value", None),
                    p_value=met_g.get("p_value", None),
                    n_a=int(met_g.get("n_a", 0)),
                    n_b=int(met_g.get("n_b", 0)),
                ))

    write_rows_csv(tab_dir / "lesion.csv", rows_l)
    write_rows_csv(tab_dir / "gmwm.csv", rows_g)

    mu_vals_l = [np.asarray(mu_l[k], dtype=np.float64) for k in metrics]
    sg_vals_l = [np.asarray(sg_l[k], dtype=np.float64) for k in metrics]
    tv_vals_l = [np.asarray(tv_l[k], dtype=np.float64) for k in metrics]

    mu_vals_g = [np.asarray(mu_g[k], dtype=np.float64) for k in metrics]
    sg_vals_g = [np.asarray(sg_g[k], dtype=np.float64) for k in metrics]
    tv_vals_g = [np.asarray(tv_g[k], dtype=np.float64) for k in metrics]

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.unicode_minus": False,
    })

    fig = plt.figure(figsize=(13.4, 5.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.85, 1.0, 0.85], wspace=0.18)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])

    plot_relative(ax0, tick_labels, mu_vals_l, sg_vals_l, "(a) Relative values lesion")
    plot_t(ax1, tick_labels, tv_vals_l, "(b) T value lesion vs c lesion")
    plot_relative(ax2, tick_labels, mu_vals_g, sg_vals_g, "(c) Relative values WM over GM")
    plot_t(ax3, tick_labels, tv_vals_g, "(d) T value WM vs GM")

    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.24, top=0.90)
    fig.savefig(str(fig_dir / "panel.png"), dpi=280, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    manifest = dict(
        run_dir=str(run_dir),
        patients_found=int(len(patient_ids)),
        patients_ok=int(ok_patients),
        patients_skipped=skipped,
        outputs=dict(
            figure="figures/panel.png",
            lesion_csv="tables/lesion.csv",
            gmwm_csv="tables/gmwm.csv",
        ),
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")

    print("run_dir", str(run_dir))
    print("patients_ok", int(ok_patients))


if __name__ == "__main__":
    main()
