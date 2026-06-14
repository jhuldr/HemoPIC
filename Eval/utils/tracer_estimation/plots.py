import numpy as np
from scipy.ndimage import gaussian_filter1d

from .io_utils import interp_to_grid


### Plot combined cohort curves with no band
def plot_combo(ax, region_curve_data, reg_list, title, plot_tmax, legend_fontsize):
    colors = {
        "GM": "#4E79A7",
        "WM": "#9ECAE1",
        "lesion": "#E15759",
        "c_lesion": "#B07AA1",
        "full_brain": "#F28E2B",
    }
    labels = {
        "GM": "GM",
        "WM": "WM",
        "lesion": "Lesion",
        "c_lesion": "c lesion",
        "full_brain": "Full",
    }

    t_dense = None
    sigma_pts = 2.0

    for reg in reg_list:
        d = region_curve_data.get(reg, None)
        if d is None:
            continue

        t_all = d["t"]
        m = t_all <= float(plot_tmax)
        t2 = t_all[m]

        if t_dense is None:
            if int(t2.size) >= 2:
                dt_base = float(np.median(np.diff(t2)))
                if (not np.isfinite(dt_base)) or dt_base <= 0.0:
                    dt_base = 1.0
            else:
                dt_base = 1.0
            dt_plot = max(dt_base / 4.0, 0.25)
            t_dense = np.arange(0.0, float(plot_tmax) + 0.5 * dt_plot, dt_plot).astype(np.float64)

        obs_c = interp_to_grid(t2, d["obs_mu"][m], t_dense)
        fit_c = interp_to_grid(t2, d["fit_mu"][m], t_dense)

        obs_c = gaussian_filter1d(obs_c.astype(np.float64), sigma=sigma_pts, mode="nearest")
        fit_c = gaussian_filter1d(fit_c.astype(np.float64), sigma=sigma_pts, mode="nearest")

        col = colors.get(reg, "#000000")
        lab = labels.get(reg, reg)

        ax.plot(t_dense, obs_c, color=col, linewidth=2.0, linestyle="solid", label=f"{lab} obs")
        ax.plot(t_dense, fit_c, color=col, linewidth=2.0, linestyle="dashed", label=f"{lab} fit")

    ax.set_title(title, fontsize=16, pad=6)
    ax.set_xlabel("time s", fontsize=13)
    ax.set_ylabel("C", fontsize=13)
    ax.set_xlim(0.0, float(plot_tmax))
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(fontsize=int(legend_fontsize), ncol=2, loc="upper right")
