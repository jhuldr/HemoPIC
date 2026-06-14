import csv
import json
from pathlib import Path

import numpy as np

from HemoPIC.io.io_nifti import (
    find_nii,
    load_nifti,
    squeeze_ctc,
    safe_dt_from_header,
    ensure_clean_dir,
    save_nifti_like,
)
from HemoPIC.core.preprocessing import (
    smooth_curve_gaussian,
    robust_median_positive,
    compute_peak_map,
    choose_peak_threshold,
)
from HemoPIC.core.features import (
    compute_features_for_kmeans,
    voxel_proxy_from_ctc,
)
from HemoPIC.core.clustering import (
    kmeans_numpy,
    merge_small_clusters,
    fill_zero_labels_by_nearest,
)
from HemoPIC.core.filling import fill_map_nearest_3d
from HemoPIC.core.roi_fit import fit_roi_curve
from HemoPIC.models.models_tracer import tracer_update
from HemoPIC.config import FitConfig


### Run core fitting and save core outputs only
def run_core(dataset_root: Path, out_root: Path, patient_num, k_gm, k_wm, seed, cfg: FitConfig):
    patient_num = int(patient_num)
    k_gm = int(k_gm)
    k_wm = int(k_wm)
    seed = int(seed)

    dataset_root = Path(dataset_root)
    out_root = Path(out_root)

    patient_id = "training_" + str(int(patient_num))
    base = dataset_root / patient_id
    if not base.exists():
        raise FileNotFoundError("Missing patient folder " + str(base))

    cbf_path = find_nii(base, "MR_rCBF")
    cbv_path = find_nii(base, "MR_rCBV")
    mtt_path = find_nii(base, "MR_MTT")
    ctc_path = find_nii(base, "CTC_from_MR_4DPWI")
    if cbf_path is None or cbv_path is None or mtt_path is None or ctc_path is None:
        raise FileNotFoundError("Missing one of MR_rCBF MR_rCBV MR_MTT CTC_from_MR_4DPWI")

    seg_path = base / "synthseg" / "seg_export_fixed" / "seg4_label_ref24.nii.gz"
    if not seg_path.exists():
        raise FileNotFoundError("Missing seg4 file " + str(seg_path))

    cbf_img, cbf3 = load_nifti(cbf_path)
    _, cbv3 = load_nifti(cbv_path)
    _, mtt3 = load_nifti(mtt_path)

    ctc_img, ctc_raw = load_nifti(ctc_path)
    ctc = squeeze_ctc(ctc_raw)

    _, seg4 = load_nifti(seg_path)
    seg4 = seg4.astype(np.int32)

    if tuple(seg4.shape[:3]) != tuple(cbf3.shape[:3]):
        raise RuntimeError("Seg shape mismatch perfusion shape")

    x_dim = int(cbf3.shape[0])
    y_dim = int(cbf3.shape[1])
    z_dim = int(cbf3.shape[2])

    t_len = int(ctc.shape[3])
    dt = safe_dt_from_header(ctc_img)
    t = np.arange(t_len, dtype=np.float64) * float(dt)

    patient_out = out_root / ("p" + str(int(patient_num)))
    ensure_clean_dir(patient_out)

    global_dir = patient_out / "global"
    global_dir.mkdir(parents=True, exist_ok=True)

    core_dir = global_dir / "core"
    core_dir.mkdir(parents=True, exist_ok=True)

    inside = seg4 > 0
    ctc_finite3 = np.all(np.isfinite(ctc), axis=3)
    show_mask3 = inside & ctc_finite3

    peak3 = compute_peak_map(ctc, baseline_frames=cfg.baseline_frames)
    thr = choose_peak_threshold(peak3, show_mask3)
    fit_mask3 = show_mask3 & (peak3 > float(thr))
    if int(np.sum(fit_mask3)) < 5000:
        fit_mask3 = show_mask3

    cart_raw = ctc[fit_mask3].mean(axis=0).astype(np.float64)
    cart_smooth = smooth_curve_gaussian(
        cart_raw,
        sigma_frames=cfg.smooth_sigma_frames,
        baseline_frames=cfg.baseline_frames,
        clip_nonneg=cfg.clip_nonneg,
    )
    cart_peak = float(np.max(cart_smooth))
    cart_den = float(max(cart_peak, 1.0 / 1e12))
    cart_norm = (cart_smooth / cart_den).astype(np.float64)

    ### K means helper
    def build_subclusters_for_tissue(tissue_id, k, offset):
        sel_all3 = (seg4 == int(tissue_id)) & show_mask3
        coords = np.argwhere(sel_all3)
        n = int(coords.shape[0])

        if n < 800:
            out = np.zeros(seg4.shape, dtype=np.int32)
            ids = []
            if n > 0:
                out[coords[:, 0], coords[:, 1], coords[:, 2]] = int(offset) + 1
                ids.append(int(offset) + 1)
            return out, ids

        ctc_mat = ctc[sel_all3].astype(np.float32)
        feats = compute_features_for_kmeans(ctc_mat, coords, dt=dt, baseline_frames=cfg.baseline_frames)

        feats[:, 3] = feats[:, 3] * float(cfg.coord_weight)
        feats[:, 4] = feats[:, 4] * float(cfg.coord_weight)
        feats[:, 5] = feats[:, 5] * float(cfg.coord_weight)

        rs = np.random.RandomState(int(seed) + int(tissue_id) + 7)
        if n > int(cfg.sample_max):
            pick = rs.choice(n, size=int(cfg.sample_max), replace=False).astype(np.int64)
            feats_fit = feats[pick]
        else:
            feats_fit = feats

        lab_fit, cent = kmeans_numpy(
            feats_fit,
            k=int(k),
            seed=int(seed) + int(tissue_id) + 11,
            max_iter=int(cfg.kmeans_max_iter),
        )

        best_dist = np.full(n, np.inf, dtype=np.float32)
        best_lab = np.zeros(n, dtype=np.int32)
        for j in range(int(cent.shape[0])):
            diff = np.subtract(feats, cent[j])
            dist = np.sum(diff * diff, axis=1).astype(np.float32)
            take = dist < best_dist
            best_dist[take] = dist[take]
            best_lab[take] = int(j)

        best_lab2, _ = merge_small_clusters(best_lab, cent, min_vox=int(cfg.min_cluster_vox))

        out = np.zeros(seg4.shape, dtype=np.int32)
        ids = []
        used = np.unique(best_lab2).astype(np.int32)
        for j in used.tolist():
            roi_id = int(offset) + int(j) + 1
            selj = best_lab2 == int(j)
            pts = coords[selj]
            out[pts[:, 0], pts[:, 1], pts[:, 2]] = int(roi_id)
            ids.append(int(roi_id))
        return out, ids

    gm_sub3, gm_ids = build_subclusters_for_tissue(1, k=int(k_gm), offset=0)
    off2 = int(max(gm_ids)) if len(gm_ids) > 0 else 0
    wm_sub3, wm_ids = build_subclusters_for_tissue(2, k=int(k_wm), offset=off2)

    off3 = int(max(wm_ids)) if len(wm_ids) > 0 else int(off2)
    csf_id = int(off3) + 1
    bs_id = int(off3) + 2

    sub3 = np.zeros(seg4.shape, dtype=np.int32)
    sub3[gm_sub3 > 0] = gm_sub3[gm_sub3 > 0]
    sub3[wm_sub3 > 0] = wm_sub3[wm_sub3 > 0]
    sub3[(seg4 == 3) & show_mask3] = int(csf_id)
    sub3[(seg4 == 4) & show_mask3] = int(bs_id)

    all_ids = []
    all_ids.extend(gm_ids)
    all_ids.extend(wm_ids)
    all_ids.append(int(csf_id))
    all_ids.append(int(bs_id))

    sub3 = fill_zero_labels_by_nearest(sub3, fill_mask3=show_mask3)
    unlabeled = int(np.sum(show_mask3 & (sub3 == 0)))

    roi_labels_path = core_dir / "roi_labels_seg4_within_kmeans.nii.gz"
    save_nifti_like(cbf_img, sub3.astype(np.int32), roi_labels_path, dtype=np.int32)

    priors = {}
    fits = {}
    roi_curve_items = []

    fit_csv = core_dir / "roi_fit_summary.csv"
    with open(fit_csv, "w", newline="", encoding="utf8") as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow(
            [
                "roi_id",
                "tissue",
                "n_vox_all",
                "n_vox_fit",
                "F_prior",
                "V_prior",
                "Cin_prior",
                "R_fit",
                "C_wk_fit",
                "tau_fit",
                "F_fit",
                "V_fit",
                "Cin_bar_fit",
                "loss",
                "success",
                "message",
            ]
        )

        for roi_id in all_ids:
            sel_all = sub3 == int(roi_id)
            n_all = int(np.sum(sel_all))
            if n_all < 200:
                continue

            if int(roi_id) in gm_ids:
                tissue = "GM"
            elif int(roi_id) in wm_ids:
                tissue = "WM"
            elif int(roi_id) == int(csf_id):
                tissue = "CSF"
            else:
                tissue = "BrainStem"

            sel_fit = sel_all & fit_mask3
            n_fit = int(np.sum(sel_fit))
            if n_fit < 80:
                sel_fit = sel_all
                n_fit = int(np.sum(sel_fit))

            c_raw = ctc[sel_fit].mean(axis=0).astype(np.float64)
            c_smooth = smooth_curve_gaussian(
                c_raw,
                sigma_frames=cfg.smooth_sigma_frames,
                baseline_frames=cfg.baseline_frames,
                clip_nonneg=cfg.clip_nonneg,
            )

            f_prior = robust_median_positive(cbf3[sel_all].astype(np.float64), eps=cfg.eps_small)
            v_prior = robust_median_positive(cbv3[sel_all].astype(np.float64), eps=cfg.eps_small)
            cin_prior = float(max(float(np.max(c_smooth)), cfg.eps_small))

            priors[int(roi_id)] = {
                "F_prior": float(f_prior),
                "V_prior": float(v_prior),
                "Cin_prior": float(cin_prior),
            }

            fit = fit_roi_curve(
                c_obs=c_smooth,
                cart_norm=cart_norm,
                t=t,
                dt=dt,
                f_prior=f_prior,
                v_prior=v_prior,
                cin_prior=cin_prior,
                lambda_f=cfg.lambda_f,
                lambda_v=cfg.lambda_v,
                lambda_in=cfg.lambda_in,
                tau_cap_seconds=cfg.tau_cap_seconds,
                r_max=cfg.r_max,
                q_floor_ratio=cfg.q_floor_ratio,
                reg_lambda_logr=cfg.reg_lambda_logr,
                p0=cfg.p0,
            )

            if not bool(fit["success"]):
                q_out = np.full(t.shape[0], float(f_prior), dtype=np.float64)
                c_in_t = float(cin_prior) * cart_norm
                c_fit = tracer_update(q_out, c_in_t, float(v_prior), float(f_prior), float(dt), q_floor_ratio=cfg.q_floor_ratio)
                fit = {
                    "R": 1.0,
                    "tau": 1.0,
                    "C_wk": 1.0,
                    "V": float(v_prior),
                    "F": float(f_prior),
                    "Cin_bar": float(cin_prior),
                    "loss": float(1e9),
                    "success": False,
                    "message": "fallback_to_priors",
                    "Q_out": q_out,
                    "C_fit": c_fit,
                }

            fits[int(roi_id)] = fit
            roi_curve_items.append(
                {
                    "roi_id": int(roi_id),
                    "tissue": tissue,
                    "c_obs": c_smooth.astype(np.float64),
                    "c_fit": fit["C_fit"].astype(np.float64),
                }
            )

            wcsv.writerow(
                [
                    int(roi_id),
                    tissue,
                    int(n_all),
                    int(n_fit),
                    float(f_prior),
                    float(v_prior),
                    float(cin_prior),
                    float(fit["R"]),
                    float(fit["C_wk"]),
                    float(fit["tau"]),
                    float(fit["F"]),
                    float(fit["V"]),
                    float(fit["Cin_bar"]),
                    float(fit["loss"]),
                    bool(fit["success"]),
                    str(fit["message"]),
                ]
            )

    ### Build model maps
    md_cbf = np.full((x_dim, y_dim, z_dim), np.nan, dtype=np.float64)
    md_cbv = np.full((x_dim, y_dim, z_dim), np.nan, dtype=np.float64)

    for roi_id, fit in fits.items():
        sel_all = sub3 == int(roi_id)
        n_all = int(np.sum(sel_all))
        if n_all < 50:
            continue

        q_out = fit["Q_out"].astype(np.float64)
        tail_len = int(max(3, int(np.floor(float(q_out.size) * float(cfg.tail_frac)))))
        start = int(max(0, int(np.subtract(q_out.size, tail_len))))
        q_tail = q_out[start:]
        q_tail = q_tail[np.isfinite(q_tail)]
        if int(q_tail.size) < 1:
            q_mean = float(max(float(fit["F"]), cfg.eps_small))
        else:
            q_mean = float(max(float(np.mean(q_tail)), cfg.eps_small))

        v_roi = float(max(float(fit["V"]), cfg.eps_small))

        if int(roi_id) == int(csf_id) or int(roi_id) == int(bs_id):
            md_cbf[sel_all] = float(q_mean)
            md_cbv[sel_all] = float(v_roi)
            continue

        ctc_roi = ctc[sel_all].astype(np.float64)
        proxy, _ = voxel_proxy_from_ctc(ctc_roi, dt=dt, baseline_frames=cfg.baseline_frames, eps=cfg.eps_small)

        proxy_ratio = np.ones(int(n_all), dtype=np.float64)
        proxy_ok = proxy[np.isfinite(proxy) & (proxy > 0.0)]
        if int(proxy_ok.size) >= 20:
            med = float(np.median(proxy_ok))
            med = float(max(med, cfg.eps_small))
            proxy_ratio = np.divide(proxy, med)
            proxy_ratio = np.where(np.isfinite(proxy_ratio), proxy_ratio, 1.0)
            proxy_ratio = np.clip(proxy_ratio, cfg.proxy_clip_lo, cfg.proxy_clip_hi)

        mean_ratio = float(np.mean(proxy_ratio[np.isfinite(proxy_ratio)])) if int(np.sum(np.isfinite(proxy_ratio))) > 0 else 1.0
        mean_ratio = float(max(mean_ratio, cfg.eps_small))
        cbf_vox = np.divide(q_mean * proxy_ratio, mean_ratio).astype(np.float64)

        cbv_vox_raw = cbv3[sel_all].astype(np.float64)
        cbv_pos = cbv_vox_raw[np.isfinite(cbv_vox_raw) & (cbv_vox_raw > 0.0)]
        denom = float(max(float(np.median(cbv_pos)) if int(cbv_pos.size) >= 20 else float(priors[int(roi_id)]["V_prior"]), cfg.eps_small))

        cbv_ratio = np.divide(np.where(np.isfinite(cbv_vox_raw) & (cbv_vox_raw > 0.0), cbv_vox_raw, denom), denom)
        cbv_ratio = np.where(np.isfinite(cbv_ratio), cbv_ratio, 1.0)
        cbv_ratio = np.clip(cbv_ratio, cfg.cbv_ratio_clip_lo, cfg.cbv_ratio_clip_hi)

        mean_ratio_v = float(np.mean(cbv_ratio[np.isfinite(cbv_ratio)])) if int(np.sum(np.isfinite(cbv_ratio))) > 0 else 1.0
        mean_ratio_v = float(max(mean_ratio_v, cfg.eps_small))
        cbv_vox = np.divide(v_roi * cbv_ratio, mean_ratio_v).astype(np.float64)

        md_cbf[sel_all] = cbf_vox
        md_cbv[sel_all] = cbv_vox

    valid_cbf = show_mask3 & np.isfinite(md_cbf) & (md_cbf > cfg.eps_small)
    valid_cbv = show_mask3 & np.isfinite(md_cbv) & (md_cbv > cfg.eps_small)

    md_cbf = fill_map_nearest_3d(md_cbf, inside_mask3=show_mask3, valid_mask3=valid_cbf)
    md_cbv = fill_map_nearest_3d(md_cbv, inside_mask3=show_mask3, valid_mask3=valid_cbv)

    md_cbf = np.where(show_mask3, md_cbf, np.nan)
    md_cbv = np.where(show_mask3, md_cbv, np.nan)

    gt_cbf = cbf3.astype(np.float64)
    gt_mtt = mtt3.astype(np.float64)

    gt_cbf_mask = show_mask3 & np.isfinite(gt_cbf) & (gt_cbf > 0.0)
    gt_mtt_mask = show_mask3 & np.isfinite(gt_mtt) & (gt_mtt > 0.0)

    if int(np.sum(gt_cbf_mask)) > 200:
        cbf_floor = float(np.percentile(gt_cbf[gt_cbf_mask], 1.0))
        cbf_cap = float(np.percentile(gt_cbf[gt_cbf_mask], 99.5))
    else:
        cbf_floor = 1.0
        cbf_cap = 200.0

    if int(np.sum(gt_mtt_mask)) > 200:
        mtt_floor = float(np.percentile(gt_mtt[gt_mtt_mask], 1.0))
        mtt_cap = float(np.percentile(gt_mtt[gt_mtt_mask], 99.5))
    else:
        mtt_floor = 1.0
        mtt_cap = 30.0

    cbf_floor = float(max(cbf_floor, cfg.eps_small))
    cbf_cap = float(max(cbf_cap, cbf_floor))
    mtt_floor = float(max(mtt_floor, cfg.eps_small))
    mtt_cap = float(max(mtt_cap, mtt_floor))

    den = np.maximum(md_cbf, cbf_floor)
    den = np.minimum(den, cbf_cap)

    md_mtt = np.full_like(md_cbf, np.nan, dtype=np.float64)
    ok3 = show_mask3 & np.isfinite(md_cbv) & np.isfinite(den) & (md_cbv > 0.0) & (den > 0.0)
    md_mtt[ok3] = 60.0 * np.divide(md_cbv[ok3], den[ok3])

    md_mtt = np.where(show_mask3, np.maximum(md_mtt, mtt_floor), np.nan)
    md_mtt = np.where(show_mask3, np.minimum(md_mtt, mtt_cap), np.nan)

    model_cbf_path = core_dir / "Model_CBF_raw.nii.gz"
    model_cbv_path = core_dir / "Model_CBV_raw.nii.gz"
    model_mtt_path = core_dir / "Model_MTT_raw_clamped.nii.gz"

    save_nifti_like(cbf_img, md_cbf, model_cbf_path, dtype=np.float32)
    save_nifti_like(cbf_img, md_cbv, model_cbv_path, dtype=np.float32)
    save_nifti_like(cbf_img, md_mtt, model_mtt_path, dtype=np.float32)

    core_info = {
        "patient": int(patient_num),
        "dt_seconds": float(dt),
        "k_gm": int(k_gm),
        "k_wm": int(k_wm),
        "seed": int(seed),
        "baseline_frames": int(cfg.baseline_frames),
        "aif_peak_threshold": float(thr),
        "unlabeled_after_fill": int(unlabeled),
        "cbf_floor_from_gt_p1": float(cbf_floor),
        "cbf_cap_from_gt_p99_5": float(cbf_cap),
        "mtt_floor_from_gt_p1": float(mtt_floor),
        "mtt_cap_from_gt_p99_5": float(mtt_cap),
    }
    with open(core_dir / "run_info_core.json", "w", encoding="utf8") as f:
        json.dump(core_info, f, indent=2)

    return {
        "patient_out": patient_out,
        "global_dir": global_dir,
        "core_dir": core_dir,
        "t": t,
        "dt": float(dt),
        "cart_norm": cart_norm,
        "show_mask3": show_mask3,
        "fit_mask3": fit_mask3,
        "sub3": sub3,
        "gm_ids": gm_ids,
        "wm_ids": wm_ids,
        "csf_id": int(csf_id),
        "bs_id": int(bs_id),
        "gt_cbf": gt_cbf,
        "gt_mtt": gt_mtt,
        "md_cbf": md_cbf,
        "md_cbv": md_cbv,
        "md_mtt": md_mtt,
        "roi_curve_items": roi_curve_items,
        "paths": {
            "roi_labels": roi_labels_path,
            "fit_csv": fit_csv,
            "model_cbf": model_cbf_path,
            "model_cbv": model_cbv_path,
            "model_mtt": model_mtt_path,
        },
    }