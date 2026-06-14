import numpy as np


### Compute a per voxel proxy based on peak divided by time to peak
def voxel_proxy_from_ctc(ctc_vox_t, dt, baseline_frames, eps):
    '''
    (1) ctc_vox_t: voxelwise concentration curves with shape (n, t) (n # of voxels, t # of time points)
    (2) baseline_frames: # of initial frames used to estimate baseline
    (3) eps: small constant used to avoid divide by zero
    (4) dt: Temporal resolution (~ in seconds per frame)
    '''

    x = ctc_vox_t.astype(np.float64)

    # NaN or inf >> 0
    x = np.where(np.isfinite(x), x, 0.0)

    n = int(x.shape[0])
    t_len = int(x.shape[1])
    
    # baseline estimate from first bf frames
    bf = int(min(int(baseline_frames), t_len))
    if bf < 1:
        base = np.zeros((n, 1), dtype=np.float64)
    else:
        base = np.mean(x[:, :bf], axis=1, keepdims=True).astype(np.float64)
    
    # positive excursion above baseline
    d = np.subtract(x, base)
    d = np.maximum(d, 0.0)

    peak = np.max(d, axis=1).astype(np.float64)

    # time to peak as a timing penalty term
    # for a fixed peak amplitude, earlier peak gives larger proxy ( peak / (ttp + const) )
    ttp_idx = np.argmax(d, axis=1).astype(np.float64)
    ttp_sec = ttp_idx * float(dt)

    denom = ttp_sec + float(max(float(dt), float(eps)))
    proxy = np.divide(peak, denom)

    # clean any non finite artifacts
    proxy = np.where(np.isfinite(proxy), proxy, 0.0)
    return proxy.astype(np.float64), ttp_sec.astype(np.float64)


### Build k-means features from tracer concentration curves and spatial coordinates
def compute_features_for_kmeans(ctc4, coords3, dt, baseline_frames):
    '''
    (1) ctc4: voxelwise concentration curves with shape (n, t)
    (2) coords3: spatial coordinates with shape (n, 3) 
    ( (x, y, z) for the same voxel order as ctc4) )
    (3) dt: temporal resolution
    (4) baseline_frames: # of initial frames used to estimate baseline
    '''
    eps = 1.0 / 1e12

    x = ctc4.astype(np.float64)
    n = int(x.shape[0])
    t_len = int(x.shape[1])
    
    # baseline per voxel from first bf frames
    bf = int(min(int(baseline_frames), t_len))
    if bf < 1:
        base = np.zeros((n, 1), dtype=np.float64)
    else:
        base = np.mean(x[:, :bf], axis=1, keepdims=True).astype(np.float64)

    # positive excursion above baseline
    d = np.subtract(x, base)
    d = np.maximum(d, 0.0)
    
    # curve summary features per voxel
    peak = np.max(d, axis=1).astype(np.float64)
    ttp_idx = np.argmax(d, axis=1).astype(np.float64)
    ttp_sec = ttp_idx * float(dt)
    auc = np.sum(d, axis=1).astype(np.float64) * float(dt)
    
    # spatial features
    cx = coords3[:, 0].astype(np.float64)
    cy = coords3[:, 1].astype(np.float64)
    cz = coords3[:, 2].astype(np.float64)

    # rescale by max magnitude 
    cx = cx / float(max(float(np.max(cx)), 1.0))
    cy = cy / float(max(float(np.max(cy)), 1.0))
    cz = cz / float(max(float(np.max(cz)), 1.0))
    
    # feature matrix feats
    feats = np.stack([peak, ttp_sec, auc, cx, cy, cz], axis=1).astype(np.float64)
    feats = np.where(np.isfinite(feats), feats, 0.0)
    
    # robust center and scale per feature dimension
    med = np.median(feats, axis=0, keepdims=True)
    dev = np.abs(np.subtract(feats, med))
    mad = np.median(dev, axis=0, keepdims=True)

    # 1.4826 makes mad comparable to std under Gaussian noise
    scale = mad * 1.4826
    scale = np.maximum(scale, eps)

    z = np.divide(np.subtract(feats, med), scale)
    return z.astype(np.float32)
