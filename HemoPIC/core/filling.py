import numpy as np
from scipy.ndimage import distance_transform_edt


### Fill invalid voxels inside a mask by nearest valid voxel value
def fill_map_nearest_3d(vol3, inside_mask3, valid_mask3):
    vol = vol3.astype(np.float64).copy()

    inside = inside_mask3.astype(bool)
    valid = valid_mask3.astype(bool) & inside
    holes = inside & (np.logical_not(valid))

    if int(np.sum(holes)) < 1:
        return vol

    if int(np.sum(valid)) < 1:
        return vol

    inv = np.logical_not(valid)
    _, inds = distance_transform_edt(inv, return_indices=True)

    nx = inds[0]
    ny = inds[1]
    nz = inds[2]

    hx, hy, hz = np.where(holes)
    vol[hx, hy, hz] = vol[nx[hx, hy, hz], ny[hx, hy, hz], nz[hx, hy, hz]]
    return vol
