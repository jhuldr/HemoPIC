from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import nibabel as nib

try:
    from nibabel.processing import resample_from_to
except Exception as e:
    raise RuntimeError("Need nibabel.processing.resample_from_to") from e

'''
Run:
python /path/to/hemopic_seg4.py \
  /path/to/ISLES2017_Training \
  /path/to/output_synthseg \
  /path/to/output_hemopic_seg4
'''

# ==================================================================== #

# Build seg4 from SynthSeg label map, Output labels:
# 0 background
# 1 gray matter
# 2 white matter
# 3 csf and ventricles
# 4 brain stem

def seg30_to_seg4(seg30: np.ndarray) -> np.ndarray:
    seg30 = seg30.astype(np.int32)

    wm_labels = {2, 41}
    csf_labels = {4, 5, 14, 24, 43, 44}
    brain_stem_labels = {16}

    seg4 = np.zeros(seg30.shape, dtype=np.uint8)

    wm_mask = np.isin(seg30, list(wm_labels))
    csf_mask = np.isin(seg30, list(csf_labels))
    bs_mask = np.isin(seg30, list(brain_stem_labels))

    seg4[wm_mask] = np.uint8(2)
    seg4[csf_mask] = np.uint8(3)
    seg4[bs_mask] = np.uint8(4)

    nonzero = seg30 > 0
    already = wm_mask | csf_mask | bs_mask
    gm_mask = nonzero & (~already)
    seg4[gm_mask] = np.uint8(1)

    return seg4


### Simple gray image for quick overlay view
def seg4_to_gray(seg4: np.ndarray) -> np.ndarray:
    gray = np.zeros(seg4.shape, dtype=np.uint8)
    gray[seg4 == 1] = np.uint8(150)
    gray[seg4 == 2] = np.uint8(240)
    return gray


def file_ok(p: Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > 0


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def first_existing(paths) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def as_int(x) -> int:
    if np.isscalar(x):
        return int(x)
    return int(np.asarray(x).reshape(-1)[0])


### Save data using ref affine and qform sform codes
def save_like(ref_img: nib.Nifti1Image, data: np.ndarray, out_path: Path, dtype) -> None:
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(dtype)

    out = nib.Nifti1Image(data.astype(dtype), affine=ref_img.affine, header=hdr)

    qaff = ref_img.get_qform()
    saff = ref_img.get_sform()

    qcode = as_int(ref_img.header.get("qform_code", 0))
    scode = as_int(ref_img.header.get("sform_code", 0))

    if qaff is not None and np.all(np.isfinite(qaff)):
        out.set_qform(qaff, code=qcode)
    if saff is not None and np.all(np.isfinite(saff)):
        out.set_sform(saff, code=scode)

    nib.save(out, str(out_path))


def iter_subjects(root: Path):
    for d in sorted(root.glob("training_*")):
        if d.is_dir():
            yield d


def choose_seg30(seg_root_subj: Path, prefer_sym: bool) -> Path | None:
    sdir = seg_root_subj / "synthseg"

    cand_sym = [
        sdir / "seg_synthseg_sym.nii.gz",
        sdir / "seg_synthseg_sym.nii",
        sdir / "MR_ADC_resample_synthseg_sym.nii.gz",
        sdir / "MR_ADC_resample_synthseg_sym.nii",
    ]
    cand_plain = [
        sdir / "seg_synthseg.nii.gz",
        sdir / "seg_synthseg.nii",
        sdir / "MR_ADC_resample_synthseg.nii.gz",
        sdir / "MR_ADC_resample_synthseg.nii",
    ]

    cands = cand_sym + cand_plain if prefer_sym else cand_plain + cand_sym
    return first_existing(cands)


def choose_ref24(data_root_subj: Path) -> Path | None:
    cands = [
        data_root_subj / "MR_rCBF.nii",
        data_root_subj / "MR_rCBF.nii.gz",
        data_root_subj / "MR_ADC.nii",
        data_root_subj / "MR_ADC.nii.gz",
    ]
    return first_existing(cands)


def process_one(
    subj: str,
    data_root: Path,
    seg_root: Path,
    out_root: Path,
    prefer_sym: bool,
    skip_if_done: bool,
) -> str:
    data_subj = data_root / subj
    seg_subj = seg_root / subj

    seg30_path = choose_seg30(seg_subj, prefer_sym=prefer_sym)
    if seg30_path is None:
        print("missing_seg30", subj)
        return "missing"

    ref24_path = choose_ref24(data_subj)
    if ref24_path is None:
        print("missing_ref24", subj)
        return "missing"

    out_dir = ensure_dir(out_root / subj / "seg4")
    out_label = out_dir / "seg4_label_ref24.nii.gz"
    out_gray = out_dir / "seg4_gray_ref24.nii.gz"

    if skip_if_done and file_ok(out_label) and file_ok(out_gray):
        print("skip_done", subj)
        return "skipped"

    seg30_img = nib.load(str(seg30_path))
    ref24_img = nib.load(str(ref24_path))

    seg30 = np.asanyarray(seg30_img.dataobj).astype(np.int32)

    seg4_1mm = seg30_to_seg4(seg30)
    gray_1mm = seg4_to_gray(seg4_1mm)

    to_ref24 = (ref24_img.shape[:3], ref24_img.affine)

    seg4_ref24_img = resample_from_to(
        nib.Nifti1Image(seg4_1mm.astype(np.uint8), affine=seg30_img.affine),
        to_ref24,
        order=0,
    )
    gray_ref24_img = resample_from_to(
        nib.Nifti1Image(gray_1mm.astype(np.uint8), affine=seg30_img.affine),
        to_ref24,
        order=0,
    )

    seg4_ref24 = np.asanyarray(seg4_ref24_img.dataobj).astype(np.uint8)
    gray_ref24 = np.asanyarray(gray_ref24_img.dataobj).astype(np.uint8)

    save_like(ref24_img, seg4_ref24, out_label, dtype=np.uint8)
    save_like(ref24_img, gray_ref24, out_gray, dtype=np.uint8)

    print("saved", subj, "shape", tuple(int(x) for x in seg4_ref24.shape))
    return "processed"


def main() -> None:
    if len(sys.argv) < 4:
        raise RuntimeError(
            "Usage: python3 seg30_to_seg4.py <data_root> <seg_root> <out_root> [target_subject] [prefer_sym] [skip_if_done]"
        )

    data_root = Path(sys.argv[1]).expanduser().resolve()
    seg_root = Path(sys.argv[2]).expanduser().resolve()
    out_root = Path(sys.argv[3]).expanduser().resolve()

    target_subject = sys.argv[4] if len(sys.argv) >= 5 else ""
    prefer_sym = bool(int(sys.argv[5])) if len(sys.argv) >= 6 else True
    skip_if_done = bool(int(sys.argv[6])) if len(sys.argv) >= 7 else True

    if not data_root.exists():
        raise RuntimeError("data_root not found")
    if not seg_root.exists():
        raise RuntimeError("seg_root not found")

    ensure_dir(out_root)

    processed = 0
    skipped = 0
    missing = 0
    found = 0

    for d in iter_subjects(data_root):
        subj = d.name
        if target_subject and subj != target_subject:
            continue
        found += 1
        st = process_one(
            subj=subj,
            data_root=data_root,
            seg_root=seg_root,
            out_root=out_root,
            prefer_sym=prefer_sym,
            skip_if_done=skip_if_done,
        )
        if st == "processed":
            processed += 1
        elif st == "skipped":
            skipped += 1
        else:
            missing += 1

    if target_subject and found == 0:
        raise RuntimeError("target subject not found under data_root")

    print("done")
    print("processed", int(processed))
    print("skipped", int(skipped))
    print("missing", int(missing))


if __name__ == "__main__":
    main()