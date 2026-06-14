import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt
from scipy.ndimage import median_filter
from scipy.ndimage import label

'''
Run:
python3 path/to/run_eval_slice_maps.py \
  "/path/to/ISLES2017_Training" \
  "/path/to/HemoPIC_outputs" \
  "/path/to/HemoPIC_eval"\
  <patient_id> <slice_z> <cbf_max> <cbv_max> <mtt_max>
'''

### Load a NIfTI volume in canonical orientation and replace NaNs and infinities with zeros
### We use this for segmentation and model maps since they serve as stable inputs for rendering
def load_canonical(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data.astype(np.float32)


### Load a NIfTI volume in canonical orientation and keep NaNs
### We use this for CTC so we can compute robust means and baseline estimates with nan aware ops
def load_canonical_keep_nan(path: Path):
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)
    data = np.where(np.isfinite(data), data, np.nan).astype(np.float32)
    return img, data


### Find a NIfTI file under a directory with either nii gz or nii extension
def find_nii(base: Path, stem: str):
    p1 = base / (stem + ".nii.gz")
    if p1.exists():
        return p1
    p2 = base / (stem + ".nii")
    if p2.exists():
        return p2
    return None


### Locate the fit output directory that contains model perfusion maps
### This supports multiple folder layouts from earlier runs
def find_fit_dir(fit_root: Path, pid: int) -> Path:
    pid_str = "p" + str(int(pid))
    cand = [
        fit_root / pid_str / "global" / "core",
        fit_root / pid_str / "global",
        fit_root / (pid_str + "_kmeans") / "global" / "core",
        fit_root / (pid_str + "_kmeans") / "global",
    ]
    for d in cand:
        if d.exists():
            return d
    raise FileNotFoundError("missing fit dir for pid " + str(int(pid)))


### Find a model map file inside the fit directory
def find_model_map(fit_dir: Path, stem: str) -> Path:
    p1 = fit_dir / (stem + ".nii.gz")
    if p1.exists():
        return p1
    p2 = fit_dir / (stem + ".nii")
    if p2.exists():
        return p2
    raise FileNotFoundError("missing map " + str(stem) + " in " + str(fit_dir))


### Find the model MTT file with a fallback name for older runs
def find_model_mtt(fit_dir: Path) -> Path:
    try:
        return find_model_map(fit_dir, "Model_MTT_raw_clamped")
    except Exception:
        return find_model_map(fit_dir, "Model_MTT_raw")


### Squeeze tracer series to a 4D array shaped as X Y Z T
def squeeze_ctc(ctc_raw: np.ndarray) -> np.ndarray:
    nd = int(ctc_raw.ndim)
    if nd == 5:
        if int(ctc_raw.shape[3]) != 1:
            raise RuntimeError("CTC channel dim is not 1")
        return np.squeeze(ctc_raw, axis=3)
    if nd == 4:
        return ctc_raw
    raise RuntimeError("Unexpected CTC ndim")


### Read temporal resolution dt from the tracer concentration header
def safe_dt_from_header(ctc_img) -> float:
    zooms = ctc_img.header.get_zooms()
    if zooms is None or len(zooms) < 4:
        return 1.0
    dt = float(zooms[3])
    if (not np.isfinite(dt)) or dt <= 0.0:
        return 1.0
    return dt


### Robust positive median helper for scale estimates inside a region
def robust_median_positive(a: np.ndarray, eps: float) -> float:
    v = a[np.isfinite(a) & (a > 0.0)]
    if int(v.size) < 20:
        return float(eps)
    m = float(np.median(v))
    if (not np.isfinite(m)) or m <= 0.0:
        return float(eps)
    return float(m)


### Render an axial slice
def show_axial(ax, a2d, cmap_name, vmin, vmax, facecolor):
    img2d = np.transpose(a2d)
    cmap = plt.get_cmap(str(cmap_name)).copy()
    cmap.set_bad(color=facecolor)
    ax.imshow(
        img2d,
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        origin="lower",
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_facecolor(facecolor)


### Save a single slice PNG without margins
def save_png(out_path: Path, a2d: np.ndarray, cmap_name: str, vmin: float, vmax: float, facecolor):
    fig = plt.figure(figsize=(4.2, 4.2))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    show_axial(ax, a2d, cmap_name=cmap_name, vmin=vmin, vmax=vmax, facecolor=facecolor)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=260, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)


### Crop a 2D array to the tight bounding box of a mask plus a small pad (removes empty background)
def crop_slices_from_mask(mask2d: np.ndarray, pad: int):
    m = mask2d.astype(bool)
    if int(np.sum(m)) < 1:
        return slice(0, int(mask2d.shape[0])), slice(0, int(mask2d.shape[1]))

    coords = np.argwhere(m)
    x0 = int(np.min(coords[:, 0]))
    x1 = int(np.max(coords[:, 0]))
    y0 = int(np.min(coords[:, 1]))
    y1 = int(np.max(coords[:, 1]))

    x0 = int(max(int(np.subtract(x0, int(pad))), 0))
    y0 = int(max(int(np.subtract(y0, int(pad))), 0))

    x1_lim = int(np.subtract(mask2d.shape[0], 1))
    y1_lim = int(np.subtract(mask2d.shape[1], 1))

    x1 = int(min(int(np.add(x1, int(pad))), x1_lim))
    y1 = int(min(int(np.add(y1, int(pad))), y1_lim))

    return slice(x0, int(np.add(x1, 1))), slice(y0, int(np.add(y1, 1)))


### Compute a voxelwise proxy ratio from tracer concentration within a region
### proxy equals peak over ttp plus dt after baseline subtraction (normalized by region median to obtain a relative texture field)
def proxy_ratio_from_ctc_region(
    ctc2d_t: np.ndarray,
    region_mask: np.ndarray,
    dt: float,
    baseline_frames: int,
    clip_lo: float,
    clip_hi: float,
    eps: float,
):
    reg = region_mask.astype(bool)
    n = int(np.sum(reg))
    if n < 20:
        return None, None

    x = ctc2d_t[reg].astype(np.float64)
    x = np.where(np.isfinite(x), x, np.nan)

    t_len = int(x.shape[1])
    bf = int(min(int(baseline_frames), t_len))

    if bf < 1:
        base = np.zeros((n, 1), dtype=np.float64)
    else:
        base = np.nanmean(x[:, :bf], axis=1, keepdims=True).astype(np.float64)
        base = np.where(np.isfinite(base), base, 0.0)

    x2 = np.where(np.isfinite(x), x, base)
    d = np.maximum(np.subtract(x2, base), 0.0)

    peak = np.max(d, axis=1).astype(np.float64)
    ttp_idx = np.argmax(d, axis=1).astype(np.float64)
    ttp = np.multiply(ttp_idx, float(dt))

    denom = np.add(ttp, float(max(float(dt), float(eps))))
    proxy = np.divide(peak, denom)
    proxy = np.where(np.isfinite(proxy), proxy, 0.0)

    ok = proxy[np.isfinite(proxy) & (proxy > 0.0)]
    if int(ok.size) < 20:
        ratio = np.ones(n, dtype=np.float64)
        mean_ratio = 1.0
        return ratio.astype(np.float32), float(mean_ratio)

    med = float(np.median(ok))
    med = float(max(med, float(eps)))

    ratio = np.divide(proxy, med)
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)
    ratio = np.clip(ratio, float(clip_lo), float(clip_hi))

    mean_ratio = float(np.mean(ratio[np.isfinite(ratio)])) if int(np.sum(np.isfinite(ratio))) > 0 else 1.0
    mean_ratio = float(max(mean_ratio, float(eps)))

    return ratio.astype(np.float32), float(mean_ratio)


### Compute a CBV ratio field from an external CBV source map inside a region (normalized by region median and clip to keep a stable texture field)
def ratio_from_cbv_region(cbv_src2d: np.ndarray, region_mask: np.ndarray, clip_lo: float, clip_hi: float, eps: float):
    reg = region_mask.astype(bool)
    v = cbv_src2d[reg].astype(np.float64)
    v = np.where(np.isfinite(v), v, np.nan)

    pos = v[np.isfinite(v) & (v > 0.0)]
    if int(pos.size) < 20:
        return None, None

    denom = float(np.median(pos))
    denom = float(max(denom, float(eps)))

    ratio = np.divide(np.where(np.isfinite(v) & (v > 0.0), v, denom), denom)
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)
    ratio = np.clip(ratio, float(clip_lo), float(clip_hi))

    mean_ratio = float(np.mean(ratio[np.isfinite(ratio)])) if int(np.sum(np.isfinite(ratio))) > 0 else 1.0
    mean_ratio = float(max(mean_ratio, float(eps)))

    return ratio.astype(np.float32), float(mean_ratio)


### define a contralateral mask 
def lesion_side_from_ot(ot2d: np.ndarray) -> str:
    ot = (ot2d > 0.0) & np.isfinite(ot2d)
    if int(np.sum(ot)) < 10:
        return "right"
    coords = np.argwhere(ot)
    cy = float(np.mean(coords[:, 1].astype(np.float64)))
    mid = float(ot2d.shape[1]) * 0.5
    if cy >= mid:
        return "right"
    return "left"

### targets isolated rendering artifacts and keeps interior patterns intact
### This step affects only PNG rendering output
def cleanup_mtt_contra_red_specks(mtt2d: np.ndarray, seg2d: np.ndarray, ot2d: np.ndarray, mtt_max: float, deep_csf: np.ndarray):
    inside = seg2d > 0.0
    gmwm = (seg2d == 1.0) | (seg2d == 2.0)

    w = int(seg2d.shape[1])
    mid = int(np.floor(np.multiply(0.5, float(w))))
    band = int(max(1, int(np.rint(np.multiply(0.08, float(w))))))
    left_keep = int(np.subtract(mid, band))
    right_keep = int(np.add(mid, band))

    side = lesion_side_from_ot(ot2d)
    ys = np.arange(w, dtype=np.int32)[None, :]

    if side == "right":
        contra = ys < left_keep
    else:
        contra = ys >= right_keep

    clean_mask = inside & gmwm & contra & np.logical_not(deep_csf)

    x = mtt2d.astype(np.float32).copy()
    x = np.where(np.isfinite(x), x, np.nan)

    vals = x[clean_mask & np.isfinite(x)]
    if int(vals.size) >= 200:
        thr = float(np.percentile(vals, 99.7))
        thr = float(max(thr, np.multiply(0.85, float(mtt_max))))
    else:
        thr = float(np.multiply(0.85, float(mtt_max)))

    hot = clean_mask & np.isfinite(x) & (x >= thr)
    n_hot = int(np.sum(hot))
    if n_hot < 1:
        return x, dict(lesion_side=side, thr=float(thr), n_hot=0, n_removed=0, n_components=0)

    cc, ncc = label(hot.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(cc.ravel()).astype(np.int64)
    if int(sizes.size) > 0:
        sizes[0] = 0

    max_comp_size = 600
    bad_ids = np.where((sizes > 0) & (sizes <= int(max_comp_size)))[0].astype(np.int32)
    bad_ids = bad_ids[bad_ids != 0]
    if int(bad_ids.size) < 1:
        return x, dict(lesion_side=side, thr=float(thr), n_hot=int(n_hot), n_removed=0, n_components=int(ncc))

    bad = np.isin(cc, bad_ids)

    base = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
    med = median_filter(base, size=3, mode="nearest").astype(np.float32)

    x[bad] = med[bad]
    x = np.where(inside & np.isfinite(x), np.minimum(np.maximum(x, 0.0), float(mtt_max)), x)

    return x, dict(
        lesion_side=side,
        thr=float(thr),
        n_hot=int(n_hot),
        n_removed=int(np.sum(bad)),
        n_components=int(ncc),
        max_comp_size=int(max_comp_size),
    )

### Visualization pipeline summary
def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_root", type=Path)
    p.add_argument("fit_root", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("patient", type=int)
    p.add_argument("slice_z", type=int)
    p.add_argument("cbf_max", type=float)
    p.add_argument("cbv_max", type=float)
    p.add_argument("mtt_max", type=float)
    p.add_argument("border_strip", nargs="?", type=int, default=2)
    p.add_argument("shallow_csf_depth_thr", nargs="?", type=float, default=5.0)
    args = p.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    fit_root = args.fit_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    pid = int(args.patient)
    z = int(args.slice_z)

    train_dir = dataset_root / ("training_" + str(int(pid)))
    seg_path = train_dir / "synthseg" / "seg_export_fixed" / "seg4_label_ref24.nii.gz"
    if not seg_path.exists():
        raise FileNotFoundError(str(seg_path))

    fit_dir = find_fit_dir(fit_root, pid)
    cbf_path = find_model_map(fit_dir, "Model_CBF_raw")
    cbv_path = find_model_map(fit_dir, "Model_CBV_raw")
    mtt_path = find_model_mtt(fit_dir)

    seg3 = load_canonical(seg_path)
    cbf3 = load_canonical(cbf_path)
    cbv3 = load_canonical(cbv_path)
    mtt3 = load_canonical(mtt_path)

    if seg3.shape != cbf3.shape or seg3.shape != cbv3.shape or seg3.shape != mtt3.shape:
        raise RuntimeError("shape mismatch")

    if z < 0 or z >= int(seg3.shape[2]):
        raise RuntimeError("slice_z out of range")

    ot_path = find_nii(train_dir, "OT")
    if ot_path is not None:
        ot3 = load_canonical(ot_path)
        ot_src = "ot"
    else:
        ot3 = np.zeros_like(seg3, dtype=np.float32)
        ot_src = "missing_ot"

    seg2d = seg3[:, :, int(z)].astype(np.float32)
    ot2d = ot3[:, :, int(z)].astype(np.float32)

    cbf2d = cbf3[:, :, int(z)].astype(np.float32)
    cbv2d = cbv3[:, :, int(z)].astype(np.float32)
    mtt2d = mtt3[:, :, int(z)].astype(np.float32)

    inside = seg2d > 0.0
    depth = distance_transform_edt(inside.astype(bool))

    eps = float(np.divide(1.0, 1000000.0))

    csf_all = (seg2d == 3.0) & inside
    deep_thr = float(args.shallow_csf_depth_thr)
    deep_csf = csf_all & (depth >= float(deep_thr))
    shallow_csf = csf_all & (depth < float(deep_thr))

    ctc_path = find_nii(train_dir, "CTC_from_MR_4DPWI")
    mr_cbv_path = find_nii(train_dir, "MR_rCBV")

    csf_texture = dict(used=False)

    if ctc_path is not None and mr_cbv_path is not None and int(np.sum(deep_csf)) >= 50:
        ctc_img, ctc_raw = load_canonical_keep_nan(ctc_path)
        ctc4 = squeeze_ctc(ctc_raw).astype(np.float32)

        if tuple(ctc4.shape[:3]) == tuple(seg3.shape[:3]):
            dt = safe_dt_from_header(ctc_img)
            ctc2d_t = ctc4[:, :, int(z), :].astype(np.float32)

            mr_cbv2d = load_canonical(mr_cbv_path)[:, :, int(z)].astype(np.float32)

            csf_q_mean = robust_median_positive(cbf2d[deep_csf], eps=eps)
            ratio, mean_ratio = proxy_ratio_from_ctc_region(
                ctc2d_t=ctc2d_t,
                region_mask=deep_csf,
                dt=float(dt),
                baseline_frames=5,
                clip_lo=0.35,
                clip_hi=2.50,
                eps=float(eps),
            )

            csf_v_mean = robust_median_positive(cbv2d[deep_csf], eps=eps)
            v_ratio, v_mean_ratio = ratio_from_cbv_region(
                cbv_src2d=mr_cbv2d,
                region_mask=deep_csf,
                clip_lo=0.50,
                clip_hi=2.00,
                eps=float(eps),
            )

            if ratio is not None and v_ratio is not None and mean_ratio is not None and v_mean_ratio is not None:
                reg = deep_csf.astype(bool)

                cbf_reg = np.divide(np.multiply(float(csf_q_mean), ratio.astype(np.float64)), float(mean_ratio))
                cbv_reg = np.divide(np.multiply(float(csf_v_mean), v_ratio.astype(np.float64)), float(v_mean_ratio))

                cbf_reg = np.where(np.isfinite(cbf_reg) & (cbf_reg > float(eps)), cbf_reg, float(csf_q_mean))
                cbv_reg = np.where(np.isfinite(cbv_reg) & (cbv_reg > float(eps)), cbv_reg, float(csf_v_mean))

                cbf2d[reg] = cbf_reg.astype(np.float32)
                cbv2d[reg] = cbv_reg.astype(np.float32)

                mtt_reg = np.multiply(60.0, np.divide(cbv_reg, np.maximum(cbf_reg, float(eps))))
                mtt_reg = np.where(np.isfinite(mtt_reg) & (mtt_reg >= 0.0), mtt_reg, 0.0)
                mtt_reg = np.minimum(mtt_reg, float(args.mtt_max))
                mtt2d[reg] = mtt_reg.astype(np.float32)

                csf_texture = dict(used=True, dt_seconds=float(dt))

    mtt2d_clean, mtt_info = cleanup_mtt_contra_red_specks(
        mtt2d=mtt2d,
        seg2d=seg2d,
        ot2d=ot2d,
        mtt_max=float(args.mtt_max),
        deep_csf=deep_csf,
    )

    border_strip = int(args.border_strip)
    if border_strip < 0:
        border_strip = 0

    show = inside & (depth > float(border_strip))
    show = show & np.logical_not(shallow_csf)

    cbf2d = np.where(show, cbf2d, np.nan)
    cbv2d = np.where(show, cbv2d, np.nan)
    mtt2d = np.where(show, mtt2d_clean, np.nan)
    lesion2d = np.where(show, (ot2d > 0.0).astype(np.float32), np.nan)

    sx, sy = crop_slices_from_mask(show, pad=2)

    cbf2d = cbf2d[sx, sy]
    cbv2d = cbv2d[sx, sy]
    mtt2d = mtt2d[sx, sy]
    lesion2d = lesion2d[sx, sy]
    show_crop = show[sx, sy]

    cbf2d = np.where(show_crop, cbf2d, np.nan)
    cbv2d = np.where(show_crop, cbv2d, np.nan)
    mtt2d = np.where(show_crop, mtt2d, np.nan)
    lesion2d = np.where(show_crop, lesion2d, np.nan)

    run_dir = out_dir / ("p" + str(int(pid)) + "_z" + str(int(z)).zfill(3))
    run_dir.mkdir(parents=True, exist_ok=True)

    jet0 = plt.get_cmap("jet")(0.0)
    save_png(run_dir / "lesion.png", lesion2d, "gray", 0.0, 1.0, "black")
    save_png(run_dir / "cbf.png", cbf2d, "jet", 0.0, float(args.cbf_max), jet0)
    save_png(run_dir / "cbv.png", cbv2d, "jet", 0.0, float(args.cbv_max), jet0)
    save_png(run_dir / "mtt.png", mtt2d, "jet", 0.0, float(args.mtt_max), jet0)

    manifest = dict(
        patient=int(pid),
        slice_z=int(z),
        ot_source=str(ot_src),
        border_strip=int(border_strip),
        shallow_csf_depth_thr=float(deep_thr),
        removed_shallow_csf=True,
        csf_texture=csf_texture,
        mtt_cleanup=mtt_info,
        scales=dict(
            cbf_max=float(args.cbf_max),
            cbv_max=float(args.cbv_max),
            mtt_max=float(args.mtt_max),
        ),
        outputs=dict(
            lesion="lesion.png",
            cbf="cbf.png",
            cbv="cbv.png",
            mtt="mtt.png",
        ),
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")

    print("out_dir", str(run_dir))


if __name__ == "__main__":
    main()