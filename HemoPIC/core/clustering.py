import numpy as np
from scipy.ndimage import distance_transform_edt


### k-means with numpy only
def kmeans_numpy(x, k, seed, max_iter):

    '''    
    x: feature matrix with shape (n,d), n is number of points, d is feature dim
    seed: random seed for center initialization
    k: number of clusters
    max_iter: Description
    '''

    rs = np.random.RandomState(int(seed))
    n = int(x.shape[0])

    if k < 2:
        return np.zeros(n, dtype=np.int32), np.mean(x, axis=0, keepdims=True).astype(np.float32)

    idx0 = rs.choice(n, size=int(k), replace=False)
    cent = x[idx0].astype(np.float32).copy()

    labels = np.zeros(n, dtype=np.int32)

    for _ in range(int(max_iter)):
        old = labels.copy()

        best_dist = np.full(n, np.inf, dtype=np.float32)
        best_lab = np.zeros(n, dtype=np.int32)

        for j in range(int(k)):
            diff = np.subtract(x, cent[j])
            dist = np.sum(diff * diff, axis=1).astype(np.float32)
            take = dist < best_dist
            best_dist[take] = dist[take]
            best_lab[take] = int(j)

        labels = best_lab

        for j in range(int(k)):
            sel = labels == int(j)
            m = int(np.sum(sel))
            if m < 1:
                ridx = int(rs.randint(0, n))
                cent[j] = x[ridx]
            else:
                cent[j] = np.mean(x[sel], axis=0).astype(np.float32)

        if int(np.sum(labels != old)) == 0:
            break

    return labels.astype(np.int32), cent


### Merge clusters smaller than min_vox into nearest large cluster
def merge_small_clusters(labels, cent, min_vox):
    k = int(cent.shape[0])
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    large = np.where(counts >= int(min_vox))[0].astype(np.int32)

    if int(large.size) < 1:
        return np.zeros(labels.shape[0], dtype=np.int32), cent[:1].copy()

    lab2 = labels.copy()
    large_set = set([int(x) for x in large.tolist()])

    map_small_to_large = {}
    for j in range(int(k)):
        if int(j) in large_set:
            continue
        c = cent[int(j)].astype(np.float64)
        centL = cent[large].astype(np.float64)
        diff = np.subtract(centL, c[None, :])
        dist = np.sum(diff * diff, axis=1)
        j2 = int(large[int(np.argmin(dist))])
        map_small_to_large[int(j)] = int(j2)

    if len(map_small_to_large) > 0:
        for j, j2 in map_small_to_large.items():
            sel = lab2 == int(j)
            lab2[sel] = int(j2)

    used = np.unique(lab2).astype(np.int32)
    remap = {int(u): int(i) for i, u in enumerate(used.tolist())}

    out = np.zeros_like(lab2)
    for u in used.tolist():
        out[lab2 == int(u)] = int(remap[int(u)])

    cent2 = np.zeros((int(used.size), int(cent.shape[1])), dtype=np.float32)
    for u in used.tolist():
        cent2[int(remap[int(u)])] = cent[int(u)]

    return out.astype(np.int32), cent2


### Fill zeros inside fill_mask by nearest non zero label
def fill_zero_labels_by_nearest(sub3, fill_mask3):
    holes = fill_mask3 & (sub3 == 0)
    if int(np.sum(holes)) < 1:
        return sub3

    ok = fill_mask3 & (sub3 > 0)
    if int(np.sum(ok)) < 1:
        return sub3

    inv = np.logical_not(ok)
    _, inds = distance_transform_edt(inv, return_indices=True)

    nx = inds[0]
    ny = inds[1]
    nz = inds[2]

    sub_filled = sub3.copy()
    hx, hy, hz = np.where(holes)
    sub_filled[hx, hy, hz] = sub3[nx[hx, hy, hz], ny[hx, hy, hz], nz[hx, hy, hz]]
    return sub_filled
