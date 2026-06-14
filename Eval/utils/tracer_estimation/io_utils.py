import csv
import shutil
from pathlib import Path

import numpy as np


### Create directory if missing
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


### Delete directory if it exists then recreate it
def ensure_clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


### Read roi_fit_summary csv into a list of dict rows
def read_roi_csv(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows


### Interpolate a curve to a common time grid
def interp_to_grid(t_src, y_src, t_dst):
    t_src = np.asarray(t_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    t_dst = np.asarray(t_dst, dtype=np.float64)

    ok = np.isfinite(t_src) & np.isfinite(y_src)
    if int(np.sum(ok)) < 2:
        return np.full_like(t_dst, np.nan, dtype=np.float64)

    ts = t_src[ok]
    ys = y_src[ok]
    order = np.argsort(ts)
    ts = ts[order]
    ys = ys[order]

    left = float(ys[0])
    right = float(ys[int(np.subtract(ys.size, 1))])
    out = np.interp(t_dst, ts, ys, left=left, right=right)
    return out.astype(np.float64)


### Mean over axis 0 ignoring NaNs
def nan_mean(a2):
    return np.nanmean(a2, axis=0)


### Percentile over axis 0 ignoring NaNs
def nan_q(a2, q):
    return np.nanpercentile(a2, float(q), axis=0)


def fmt_num(x, nd):
    xf = float(x)
    if not np.isfinite(xf):
        return ""
    return f"{xf:.{int(nd)}f}"
