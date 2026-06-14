import csv
import json
from pathlib import Path
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .manifest import write_manifest
from .io_utils import (
    ensure_dir,
    ensure_clean_dir,
    read_roi_csv,
    interp_to_grid,
    nan_mean,
    nan_q,
    fmt_num,
)
from .plots import plot_combo

from HemoPIC.io.io_nifti import find_nii, load_nifti, squeeze_ctc, safe_dt_from_header
from HemoPIC.core.preprocessing import smooth_curve_gaussian, compute_peak_map, weight_from_aif
from HemoPIC.models.models_windkessel import windkessel_outflow_from_r_tau
from HemoPIC.models.models_tracer import tracer_update


### Write cohort summary table
def write_latex_table_summary(out_path: Path, rows, caption, label):
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{" + caption + "}")
    lines.append("\\label{" + label + "}")
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\hline")
    lines.append("Region & N & mean & std & median & q25 & q75 \\\\")
    lines.append("\\hline")
    for r in rows:
        lines.append(
            f"{r['region']} & {int(r['N'])} & {r['mean']} & {r['std']} & {r['median']} & {r['q25']} & {r['q75']} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    out_path.write_text("\n".join(lines), encoding="utf8")


### Compute observed region mean curve
def region_obs_curve(ctc, mask3, baseline_frames, smooth_sigma_frames, clip_nonneg):
    n = int(np.sum(mask3))
    if n < 1:
        return None, 0

    c_raw = ctc[mask3].mean(axis=0).astype(np.float64)
    c_obs = smooth_curve_gaussian(
        c_raw,
        sigma_frames=smooth_sigma_frames,
        baseline_frames=baseline_frames,
        clip_nonneg=clip_nonneg,
    )
    return c_obs, n


### Compute fitted region mean curve by ROI curve mixture
def region_fit_curve(roi_fit_curve, sub3, mask3, t_len):
    n = int(np.sum(mask3))
    if n < 1:
        return None, 0

    num = np.zeros(t_len, dtype=np.float64)
    den = 0.0

    for rid, cfit in roi_fit_curve.items():
        a = int(np.sum(mask3 & (sub3 == int(rid))))
        if a < 1:
            continue
        num = np.add(num, np.multiply(float(a), cfit))
        den = float(np.add(den, float(a)))

    if den <= 0.0:
        return None, n

    return np.divide(num, den).astype(np.float64), n


### Compute weighted errors on a selected time window
def weighted_errors(c_obs, c_fit, w_t, w_mask):
    diff = np.subtract(c_fit, c_obs)
    diff2 = np.multiply(diff, diff)
    obs2 = np.multiply(c_obs, c_obs)

    ww = np.multiply(w_mask.astype(np.float64), w_t.astype(np.float64))
    wsum = float(np.sum(ww))
    if wsum <= 0.0:
        return float("nan"), float("nan"), float("nan")

    rmse = float(np.sqrt(np.divide(float(np.sum(np.multiply(ww, diff2))), wsum)))

    if int(np.sum(w_mask)) > 0:
        peak = float(np.max(c_obs[w_mask]))
    else:
        peak = float(np.max(c_obs))

    nrmse = float("nan")
    if np.isfinite(peak) and peak > 0.0:
        nrmse = float(np.divide(rmse, peak))

    eps = 1.0 / float(10.0 ** 12)
    denom = float(np.sum(np.multiply(ww, obs2)))
    nmse = float(np.divide(float(np.sum(np.multiply(ww, diff2))), float(np.add(denom, eps))))

    return rmse, nrmse, nmse


### Main entry used by scripts run_eval_tracer_estimation
def run_report(dataset_root: Path, fit_root: Path, report_root: Path):
    dataset_root = Path(dataset_root)
    fit_root = Path(fit_root)
    report_root = Path(report_root)

    ensure_dir(report_root)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = report_root / ("run_" + str(tag))
    ensure_clean_dir(run_dir)

    fig_dir = run_dir / "figures"
    tab_dir = run_dir / "tables"
    curve_dir = run_dir / "curves"
    ensure_dir(fig_dir)
    ensure_dir(tab_dir)
    ensure_dir(curve_dir)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )

    ### Shared settings
    smooth_sigma_frames = 1.8
    clip_nonneg = True
    q_floor_ratio = 0.02
    p0 = 1.0
    w_thr = 0.30

    region_names = ["GM", "WM", "lesion", "c_lesion", "full_brain"]

    ### Discover fitted patients
    patient_dirs = sorted([p for p in fit_root.glob("p*") if p.is_dir()])
    patient_nums = []
    for pdir in patient_dirs:
        name = pdir.name
        if not name.startswith("p"):
            continue
        num_str = name.split("_")[0].replace("p", "")
        if len(num_str) < 1:
            continue
        patient_nums.append(int(num_str))
    patient_nums = sorted(list(set(patient_nums)))

    per_patient_metrics = []
    per_patient_curves = {k: [] for k in region_names}
    dt_list = []
    tend_list = []

    for patient_num in patient_nums:
        pdir = fit_root / ("p" + str(int(patient_num)))
        if not pdir.exists():
            pdir = fit_root / ("p" + str(int(patient_num)) + "_kmeans")
        core_dir = pdir / "global" / "core"
        if not core_dir.exists():
            core_dir = pdir / "global"

        csv_path = core_dir / "roi_fit_summary.csv"
        info_path = core_dir / "run_info_core.json"
        lab_path = core_dir / "roi_labels_seg4_within_kmeans.nii.gz"

        if (not csv_path.exists()) or (not info_path.exists()) or (not lab_path.exists()):
            continue

        base = dataset_root / ("training_" + str(int(patient_num)))
        ctc_path = find_nii(base, "CTC_from_MR_4DPWI")
        if ctc_path is None:
            continue

        ctc_img, ctc_raw = load_nifti(ctc_path)
        ctc = squeeze_ctc(ctc_raw)

        _, sub3 = load_nifti(lab_path)
        sub3 = sub3.astype(np.int32)

        dt = safe_dt_from_header(ctc_img)
        t_len = int(ctc.shape[3])
        t = np.arange(t_len, dtype=np.float64) * float(dt)

        run_info = json.loads(info_path.read_text(encoding="utf8"))
        baseline_frames = int(run_info["baseline_frames"])
        thr = float(run_info["aif_peak_threshold"])

        inside = sub3 > 0
        ctc_finite3 = np.all(np.isfinite(ctc), axis=3)
        show_mask3 = inside & ctc_finite3

        peak3 = compute_peak_map(ctc, baseline_frames=baseline_frames)
        fit_mask3 = show_mask3 & (peak3 > float(thr))
        if int(np.sum(fit_mask3)) < 1:
            fit_mask3 = show_mask3

        cart_raw = ctc[fit_mask3].mean(axis=0).astype(np.float64)
        cart_smooth = smooth_curve_gaussian(
            cart_raw,
            sigma_frames=smooth_sigma_frames,
            baseline_frames=baseline_frames,
            clip_nonneg=clip_nonneg,
        )
        cart_peak = float(np.max(cart_smooth))
        cart_den = float(max(cart_peak, 1.0 / float(10.0 ** 12)))
        cart_norm = np.divide(cart_smooth, cart_den).astype(np.float64)

        w_t = weight_from_aif(cart_norm)
        w_mask = w_t >= float(w_thr)
        if int(np.sum(w_mask)) < 3:
            w_mask = w_t >= 0.10

        roi_rows = read_roi_csv(csv_path)

        roi_tissue = {}
        for r in roi_rows:
            roi_tissue[int(r["roi_id"])] = str(r["tissue"]).strip()

        gm_roi_ids = [rid for rid, tis in roi_tissue.items() if tis == "GM"]
        wm_roi_ids = [rid for rid, tis in roi_tissue.items() if tis == "WM"]

        gm_mask = np.zeros(sub3.shape, dtype=bool)
        wm_mask = np.zeros(sub3.shape, dtype=bool)
        for rid in gm_roi_ids:
            gm_mask = gm_mask | (sub3 == int(rid))
        for rid in wm_roi_ids:
            wm_mask = wm_mask | (sub3 == int(rid))

        gm_mask = gm_mask & fit_mask3
        wm_mask = wm_mask & fit_mask3
        full_mask = fit_mask3

        ot = None
        ot_path = find_nii(base, "OT")
        if ot_path is not None:
            _, ot_raw = load_nifti(ot_path)
            if int(ot_raw.ndim) == 3:
                ot = (ot_raw > 0).astype(np.uint8)

        if ot is None:
            les_mask = np.zeros(sub3.shape, dtype=bool)
            cles_mask = np.zeros(sub3.shape, dtype=bool)
        else:
            les_mask = (ot > 0) & fit_mask3
            cles_mask = (np.flip(ot, axis=0) > 0) & fit_mask3

        ### Reconstruct fitted curve per ROI
        roi_fit_curve = {}
        for r in roi_rows:
            ok = str(r.get("success", "")).strip() == "True"
            if not ok:
                continue

            rid = int(r["roi_id"])
            R = float(r["R_fit"])
            tau = float(r["tau_fit"])
            F = float(r["F_fit"])
            V = float(r["V_fit"])
            Cin_bar = float(r["Cin_bar_fit"])

            q_out = windkessel_outflow_from_r_tau(t, float(F), float(R), float(tau), float(p0))
            c_in_t = np.multiply(float(Cin_bar), cart_norm)
            c_fit_r = tracer_update(q_out, c_in_t, float(V), float(F), float(dt), q_floor_ratio=float(q_floor_ratio))
            roi_fit_curve[rid] = c_fit_r

        region_masks = {
            "GM": gm_mask,
            "WM": wm_mask,
            "lesion": les_mask,
            "c_lesion": cles_mask,
            "full_brain": full_mask,
        }

        for reg in region_names:
            m3 = region_masks[reg]
            c_obs, n_obs = region_obs_curve(ctc, m3, baseline_frames, smooth_sigma_frames, clip_nonneg)

            c_fit = None
            if int(len(roi_fit_curve)) > 0:
                c_fit, _ = region_fit_curve(roi_fit_curve, sub3, m3, t_len)

            rmse = float("nan")
            nrmse = float("nan")
            nmse = float("nan")

            if (c_obs is not None) and (c_fit is not None):
                rmse, nrmse, nmse = weighted_errors(c_obs, c_fit, w_t, w_mask)
                per_patient_curves[reg].append(
                    {
                        "patient": int(patient_num),
                        "t": t.copy(),
                        "obs": c_obs.copy(),
                        "fit": c_fit.copy(),
                    }
                )

            per_patient_metrics.append(
                {
                    "patient": int(patient_num),
                    "region": reg,
                    "n_vox": int(n_obs),
                    "dt_seconds": float(dt),
                    "t_end_seconds": float(t[int(np.subtract(t.size, 1))]),
                    "rmse": rmse,
                    "nrmse": nrmse,
                    "nmse": nmse,
                }
            )

        dt_list.append(float(dt))
        tend_list.append(float(t[int(np.subtract(t.size, 1))]))

    ### Save per patient metrics
    out_csv = tab_dir / "metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf8") as f:
        cols = ["patient", "region", "n_vox", "dt_seconds", "t_end_seconds", "rmse", "nrmse", "nmse"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(per_patient_metrics, key=lambda x: (int(x["patient"]), str(x["region"]))):
            w.writerow(r)

    ### Cohort summary tables
    def cohort_summary(metric_key):
        rows = []
        for reg in region_names:
            vals = []
            for r in per_patient_metrics:
                if str(r["region"]) != reg:
                    continue
                x = float(r[metric_key])
                if not np.isfinite(x):
                    continue
                vals.append(x)

            a = np.asarray(vals, dtype=np.float64)
            if int(a.size) < 1:
                rows.append({"region": reg, "N": 0, "mean": "", "std": "", "median": "", "q25": "", "q75": ""})
                continue

            std = float(np.std(a, ddof=1)) if int(a.size) > 1 else float("nan")

            rows.append(
                {
                    "region": reg,
                    "N": int(a.size),
                    "mean": fmt_num(float(np.mean(a)), 4),
                    "std": fmt_num(std, 4),
                    "median": fmt_num(float(np.median(a)), 4),
                    "q25": fmt_num(float(np.percentile(a, 25.0)), 4),
                    "q75": fmt_num(float(np.percentile(a, 75.0)), 4),
                }
            )
        return rows

    write_latex_table_summary(tab_dir / "cohort_nrmse.tex", cohort_summary("nrmse"), "Cohort summary of regional nRMSE", "tab_cohort_nrmse")
    write_latex_table_summary(tab_dir / "cohort_nmse.tex", cohort_summary("nmse"), "Cohort summary of regional NMSE", "tab_cohort_nmse")
    write_latex_table_summary(tab_dir / "cohort_rmse.tex", cohort_summary("rmse"), "Cohort summary of regional RMSE", "tab_cohort_rmse")

    ### Cohort curves and key figure
    if int(len(dt_list)) > 0 and int(len(tend_list)) > 0:
        dt_common = float(np.median(np.asarray(dt_list, dtype=np.float64)))
        tend_common = float(np.min(np.asarray(tend_list, dtype=np.float64)))

        if (not np.isfinite(dt_common)) or dt_common <= 0.0:
            dt_common = 1.0
        if (not np.isfinite(tend_common)) or tend_common <= 0.0:
            tend_common = 1.0

        t_grid = np.arange(0.0, tend_common + 0.5 * dt_common, dt_common).astype(np.float64)

        region_curve_data = {}
        for reg in region_names:
            items = per_patient_curves[reg]
            if int(len(items)) < 1:
                continue

            obs_mat = []
            fit_mat = []
            for it in items:
                obs_mat.append(interp_to_grid(it["t"], it["obs"], t_grid))
                fit_mat.append(interp_to_grid(it["t"], it["fit"], t_grid))

            obs_mat = np.stack(obs_mat, axis=0)
            fit_mat = np.stack(fit_mat, axis=0)

            region_curve_data[reg] = {
                "t": t_grid,
                "obs_mu": nan_mean(obs_mat),
                "fit_mu": nan_mean(fit_mat),
            }

        ### Save mean curves
        name_map = {
            "GM": "gm_mean.csv",
            "WM": "wm_mean.csv",
            "lesion": "lesion_mean.csv",
            "c_lesion": "c_lesion_mean.csv",
            "full_brain": "full_mean.csv",
        }

        for reg in region_names:
            d = region_curve_data.get(reg, None)
            if d is None:
                continue

            out_curve_csv = curve_dir / name_map[reg]
            with open(out_curve_csv, "w", newline="", encoding="utf8") as f:
                w = csv.writer(f)
                w.writerow(["time_seconds", "obs_mean", "fit_mean"])
                for i in range(int(t_grid.size)):
                    w.writerow([float(t_grid[i]), float(d["obs_mu"][i]), float(d["fit_mu"][i])])

        plot_tmax = float(min(40.0, float(t_grid[int(np.subtract(t_grid.size, 1))])))

        fig, ax = plt.subplots(1, 2, figsize=(13.6, 4.8))
        plot_combo(ax[0], region_curve_data, ["GM", "WM", "full_brain"], "Tracer estimation curves GM WM full", plot_tmax, 12)
        plot_combo(ax[1], region_curve_data, ["lesion", "c_lesion"], "Tracer estimation curves lesion c lesion", plot_tmax, 12)
        fig.tight_layout(pad=0.4, w_pad=1.0)
        fig.savefig(fig_dir / "combo_two.png", dpi=220)
        plt.close(fig)

    payload = {
        "run_dir": str(run_dir),
        "n_patients_found": int(len(patient_nums)),
        "tables": [
            "tables/metrics.csv",
            "tables/cohort_nrmse.tex",
            "tables/cohort_nmse.tex",
            "tables/cohort_rmse.tex",
        ],
        "figures": [
            "figures/combo_two.png",
        ],
        "curves": [
            "curves/gm_mean.csv",
            "curves/wm_mean.csv",
            "curves/lesion_mean.csv",
            "curves/c_lesion_mean.csv",
            "curves/full_mean.csv",
        ],
    }
    write_manifest(run_dir, payload)

    print("run_dir", str(run_dir))
    print("n_patients_found", int(len(patient_nums)))
