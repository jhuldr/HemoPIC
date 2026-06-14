from pathlib import Path
import sys
import os
import numpy as np
import nibabel as nib

"""
Convert ISLES2017 MR_4DPWI signal into a tracer time series.

Run:
python "/path/to/pwi_to_tracer.py" "/path/to/ISLES2017_Training" 1 48
"""

# Parse a positive float, else fall back to default
def _as_float(x, default_value):
    try:
        v = float(x)
        if np.isfinite(v) and v > 0.0:
            return v
    except Exception:
        pass
    return float(default_value)

# Parse an int, else fall back to default
def _as_int(x, default_value):
    try:
        return int(x)
    except Exception:
        return int(default_value)

# Accept 4D input, or squeeze singleton axes from 5D input into 4D
def _squeeze_to_4d(arr):
    a = np.asanyarray(arr)
    if int(a.ndim) == 4:
        return a
    if int(a.ndim) == 5:
        squeeze_axes = [int(i) for i, s in enumerate(a.shape) if int(s) == 1]
        if int(len(squeeze_axes)) > 0:
            a2 = np.squeeze(a, axis=tuple(squeeze_axes))
            if int(a2.ndim) == 4:
                return a2
    raise RuntimeError("MR_4DPWI array is not 4D after squeezing")

# Estimate baseline S0 and bolus arrival time index from global mean signal
def _estimate_s0_and_bat(signal4, s0_threshold):
    t_len = int(signal4.shape[3])

    sig_avg = np.mean(signal4, axis=(0, 1, 2), dtype=np.float64)
    ttp = int(np.argmin(sig_avg))

    bat = 0
    tiny = float(1.0 / 1000000000000.0)

    while True:
        if int(bat + 1) >= int(t_len):
            t_len_minus_two = int(np.subtract(int(t_len), 2))
            bat = int(max(int(t_len_minus_two), 0))
            break

        s0_avg = float(np.mean(sig_avg[: int(bat + 1)]))
        nxt = float(sig_avg[int(bat + 1)])
        denom = float(max(s0_avg, tiny))
        rel = float(np.abs(np.subtract(s0_avg, nxt)) / denom)

        if float(rel) < float(s0_threshold):
            bat = int(bat + 1)
            continue

        bat_minus_one = int(np.subtract(int(bat), 1))
        bat = int(max(int(bat_minus_one), 0))
        break

    if int(bat) < 1:
        bat = 1

    s0 = np.mean(signal4[..., : int(bat)], axis=3).astype(np.float64)
    return s0, int(bat), int(ttp)

# Convert signal S into tracer proportional to log(S0 / S)
def _tracer_from_pwi(signal4, te_seconds, s0_threshold):
    x = signal4.astype(np.float64, copy=False)
    x = np.where(np.isfinite(x), x, 1.0)
    x = np.maximum(x, 1.0)

    s0, bat, ttp = _estimate_s0_and_bat(x, s0_threshold=float(s0_threshold))
    s0 = np.where(np.isfinite(s0), s0, 1.0)
    s0 = np.maximum(s0, 1.0)

    te = float(te_seconds)
    scale = float(1.0 / te)

    tracer = scale * np.log(np.divide(s0[..., None], x))
    tracer = np.where(np.isfinite(tracer), tracer, 0.0).astype(np.float32)

    info = {"bat_index": int(bat), "ttp_index": int(ttp)}
    return tracer, info

# Support nii and nii.gz
def _find_pwi_path(subject_dir):
    p1 = Path(subject_dir) / "MR_4DPWI.nii"
    if p1.exists():
        return p1
    p2 = Path(subject_dir) / "MR_4DPWI.nii.gz"
    if p2.exists():
        return p2
    return None

# Decide whether input is a file, a subject folder, or a training root folder
def _collect_subject_dirs(input_path):
    p = Path(input_path)
    if p.is_file():
        return [p.parent], True

    pwi = _find_pwi_path(p)
    if pwi is not None:
        return [p], True

    dirs = [d for d in sorted(p.glob("training_*")) if d.is_dir()]
    return dirs, False

# Fast integrity check for an existing output file
def _output_ok(out_path):
    try:
        img = nib.load(str(out_path))
        _ = img.shape
        return True
    except Exception:
        return False

# Write tracer into the subject folder
def _write_tracer_for_subject(subject_dir, te_seconds, dt_seconds, s0_threshold, overwrite):
    subject_dir = Path(subject_dir)
    pwi_path = _find_pwi_path(subject_dir)
    if pwi_path is None:
        return

    out_path = subject_dir / "Tracer_from_MR_4DPWI.nii"

    if out_path.exists() and int(overwrite) == 0:
        if _output_ok(out_path):
            return

    img = nib.load(str(pwi_path))
    pwi = np.asanyarray(img.dataobj)
    pwi4 = _squeeze_to_4d(pwi)

    tracer, _ = _tracer_from_pwi(
        pwi4,
        te_seconds=float(te_seconds),
        s0_threshold=float(s0_threshold),
    )

    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)

    zooms = img.header.get_zooms()
    if zooms is None or int(len(zooms)) < 3:
        sx = 1.0
        sy = 1.0
        sz = 1.0
    else:
        sx = float(zooms[0])
        sy = float(zooms[1])
        sz = float(zooms[2])

    try:
        hdr.set_zooms((sx, sy, sz, float(dt_seconds)))
    except Exception:
        pass

    out_img = nib.Nifti1Image(tracer, affine=img.affine, header=hdr)
    nib.save(out_img, str(out_path))

# Single mode: input is a subject folder or a pwi file
# Root mode: input is a training root folder
def main(argv):
    if int(len(argv)) < 2:
        env_root = os.environ.get("ISLES2017_TRAINING_ROOT", "")
        if env_root == "":
            raise SystemExit("Need an input path or ISLES2017_TRAINING_ROOT")
        input_path = env_root
        rest = []
    else:
        input_path = argv[1]
        rest = argv[2:]

    subject_dirs, single_mode = _collect_subject_dirs(input_path)

    te_seconds = 0.03
    dt_seconds = 1.0
    s0_threshold = 0.05
    overwrite = 0

    if single_mode:
        if int(len(rest)) >= 1:
            te_seconds = _as_float(rest[0], 0.03)
        if int(len(rest)) >= 2:
            dt_seconds = _as_float(rest[1], 1.0)
        if int(len(rest)) >= 3:
            s0_threshold = _as_float(rest[2], 0.05)
        if int(len(rest)) >= 4:
            overwrite = _as_int(rest[3], 0)

        for d in subject_dirs:
            _write_tracer_for_subject(d, te_seconds, dt_seconds, s0_threshold, overwrite)
        return

    start_idx = 1
    end_idx = 48

    if int(len(rest)) >= 1:
        start_idx = _as_int(rest[0], 1)
    if int(len(rest)) >= 2:
        end_idx = _as_int(rest[1], 48)
    if int(len(rest)) >= 3:
        te_seconds = _as_float(rest[2], 0.03)
    if int(len(rest)) >= 4:
        dt_seconds = _as_float(rest[3], 1.0)
    if int(len(rest)) >= 5:
        s0_threshold = _as_float(rest[4], 0.05)
    if int(len(rest)) >= 6:
        overwrite = _as_int(rest[5], 0)

    for d in subject_dirs:
        try:
            idx = int(str(d.name).split("_")[1])
        except Exception:
            continue
        if int(idx) < int(start_idx) or int(idx) > int(end_idx):
            continue
        _write_tracer_for_subject(d, te_seconds, dt_seconds, s0_threshold, overwrite)


if __name__ == "__main__":
    main(sys.argv)