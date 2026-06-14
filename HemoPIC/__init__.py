from pathlib import Path
import shutil

import numpy as np
import nibabel as nib


### Load a NIfTI file and return image plus numpy array
def load_nifti(path: Path):
    img = nib.load(str(path))
    data = np.asanyarray(img.get_fdata())
    return img, data


### Find a file named stem with suffix nii or nii gz under base
def find_nii(base: Path, stem: str):
    p1 = base / (stem + ".nii")
    if p1.exists():
        return p1
    p2 = base / (stem + ".nii.gz")
    if p2.exists():
        return p2
    return None


### Normalize CTC array shape to X Y Z T
def squeeze_ctc(ctc_raw):
    if int(ctc_raw.ndim) == 5:
        if int(ctc_raw.shape[3]) != 1:
            raise RuntimeError("CTC channel dim is not 1")
        return np.squeeze(ctc_raw, axis=3)
    if int(ctc_raw.ndim) == 4:
        return ctc_raw
    raise RuntimeError("Unexpected CTC ndim")


### Read temporal spacing from header zooms and return a safe dt
def safe_dt_from_header(ctc_img):
    zooms = ctc_img.header.get_zooms()
    if zooms is None or len(zooms) < 4:
        return 1.0
    dt = float(zooms[3])
    if (not np.isfinite(dt)) or dt <= 0.0:
        return 1.0
    return dt


### Remove directory if it exists then recreate it empty
def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


### Save a volume using affine and header copied from ref_img
def save_nifti_like(ref_img, vol, out_path: Path, dtype=np.float32):
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(dtype)
    img = nib.Nifti1Image(vol.astype(dtype), affine=ref_img.affine, header=hdr)
    nib.save(img, str(out_path))


### Save a structured RGB volume using affine and header from ref_img
def save_rgb_nifti_like(ref_img, rgb_struct3, out_path: Path):
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(rgb_struct3.dtype)
    img = nib.Nifti1Image(rgb_struct3, affine=ref_img.affine, header=hdr)
    nib.save(img, str(out_path))