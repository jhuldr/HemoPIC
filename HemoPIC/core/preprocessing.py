import numpy as np
from scipy.ndimage import gaussian_filter1d


### smooth a 1D curve with a Gaussian filter and baseline removal
def smooth_curve_gaussian(y, sigma_frames, baseline_frames, clip_nonneg):
    y = np.asarray(y, dtype=np.float64)
    if int(y.size) < 5:
        return y.copy()

    bf = int(baseline_frames)
    if bf < 1:
        base = 0.0
    else:
        bf = int(min(bf, int(y.size)))
        base = float(np.mean(y[:bf]))
    
    # remove baseline so smoothing mainly acts on the bolus shaped residual
    y0 = np.subtract(y, base)
    
    # read sigma and fallback sigma to avoid invalid smoothing configuration
    sig = float(sigma_frames)
    if (not np.isfinite(sig)) or sig <= 0.0:
        sig = 1.5
    
     # apply Gaussian smoothing ("nearest" used to avoid edge artifacts)
    ys = gaussian_filter1d(y0, sigma=sig, mode="nearest")

    # add baseline back 
    ys = ys + base
    
    # (concentration should not be negative)
    if bool(clip_nonneg):
        ys = np.maximum(ys, 0.0)

    return ys.astype(np.float64)


### robust median of positive finite values (for constructing priors for inflow and volume)
def robust_median_positive(a, eps):
    a = a[np.isfinite(a) & (a > 0.0)]

    # remove ROI with samples less than 20 for robustness
    if int(a.size) < 20:
        return float(eps)
    
    v = float(np.median(a))
    if v <= 0.0:
        return float(eps)
    return v


### build weights from a normalized AIF like curve
def weight_from_aif(a):
    w = a.astype(np.float64).copy()
    w = np.subtract(w, float(np.min(w)))
    mx = float(np.max(w))
    if mx > 0.0:
        w = w / mx
    w = np.clip(w, 0.0, 1.0)
    return 0.2 + 0.8 * w


### Peak map from tracer concentration after baseline subtraction
def compute_peak_map(ctc, baseline_frames):
    # temporal dimension (the 4-th) from ctc curve (tracer concentration)
    t_len = int(ctc.shape[3])
    bf = int(min(int(baseline_frames), t_len))
    if bf < 1:
        base = np.zeros(ctc.shape[:3], dtype=np.float64)
    else:
        base = np.mean(ctc[..., :bf].astype(np.float64), axis=3)

    d = np.subtract(ctc.astype(np.float64), base[..., None])
    d = np.maximum(d, 0.0)
    peak = np.max(d, axis=3)
    return peak.astype(np.float64)


### Choose a threshold on the peak map using percentiles inside a mask
def choose_peak_threshold(peak, mask):
    v = peak[mask]
    v = v[np.isfinite(v)]
    if int(v.size) < 100:
        return 0.0

    p20 = float(np.percentile(v, 20.0))
    p80 = float(np.percentile(v, 80.0))
    span = float(max(float(np.subtract(p80, p20)), 1.0 / 1e12))
    thr = float(max(p20 + 0.02 * span, 0.0))
    return thr
