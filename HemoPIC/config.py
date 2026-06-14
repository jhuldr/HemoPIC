from dataclasses import dataclass


@dataclass
class FitConfig:
    ### Curve preprocessing
    smooth_sigma_frames: float = 1.8
    baseline_frames: int = 5
    clip_nonneg: bool = True

    ### ROI fitting
    tau_cap_seconds: float = 300.0
    r_max: float = 5000.0
    q_floor_ratio: float = 0.02
    reg_lambda_logr: float = 0.002

    lambda_in: float = 0.06
    lambda_v: float = 0.04
    lambda_f: float = 0.04

    eps_small: float = 1.0 / 1e6
    p0: float = 1.0

    ### k-means ROI partition
    min_cluster_vox: int = 400
    coord_weight: float = 0.10
    sample_max: int = 80000
    kmeans_max_iter: int = 25

    ### Map construction
    proxy_clip_lo: float = 0.35
    proxy_clip_hi: float = 2.50
    cbv_ratio_clip_lo: float = 0.50
    cbv_ratio_clip_hi: float = 2.00
    tail_frac: float = 0.20

    ### Visualization 
    enable_viz: bool = False
    enable_roi_curves: bool = True
    enable_slice_png: bool = True
    enable_aux_nifti: bool = True