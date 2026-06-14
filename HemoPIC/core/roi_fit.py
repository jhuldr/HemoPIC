import numpy as np
from scipy.optimize import minimize

from HemoPIC.core.preprocessing import weight_from_aif
from HemoPIC.models.models_windkessel import windkessel_outflow_from_r_tau
from HemoPIC.models.models_tracer import tracer_update


### Fit one ROI tracer curve with Windkessel outflow and a tracer update model
def fit_roi_curve(
    c_obs,
    cart_norm,
    t,
    dt,
    f_prior,
    v_prior,
    cin_prior,
    lambda_f,
    lambda_v,
    lambda_in,
    tau_cap_seconds,
    r_max,
    q_floor_ratio,
    reg_lambda_logr,
    p0,
):
    w = weight_from_aif(cart_norm)

    t_end = float(t[int(np.subtract(t.size, 1))]) if int(t.size) >= 2 else float(dt)
    tau_min = float(max(float(dt), 0.001))
    tau_max = float(min(float(tau_cap_seconds), 0.98 * t_end))
    if not (tau_max > tau_min):
        tau_max = float(tau_min + 20.0 * float(dt))

    eps = 1.0 / 1e12

    f0 = float(f_prior)
    v0 = float(v_prior)
    cin0 = float(cin_prior)

    if (not np.isfinite(f0)) or f0 <= 0.0:
        f0 = 1.0 / 1e6
    if (not np.isfinite(v0)) or v0 <= 0.0:
        v0 = 1.0 / 1e6
    if (not np.isfinite(cin0)) or cin0 <= 0.0:
        cin0 = 1.0 / 1e6

    r0 = 1.0
    log_r0 = float(np.log(r0))

    ### Compute weighted normalized MSE and return simulated series
    def data_mse(r, tau, v, f_in, cin_bar):
        q_out = windkessel_outflow_from_r_tau(t, f_in, r, tau, p0)
        if np.any(np.logical_not(np.isfinite(q_out))):
            return 1e9, None, None

        c_in_t = cin_bar * cart_norm
        c_fit = tracer_update(q_out, c_in_t, v, f_in, dt, q_floor_ratio=q_floor_ratio)

        diff = np.subtract(c_fit, c_obs)
        num = float(np.mean(np.multiply(w, np.multiply(diff, diff))))
        den = float(np.mean(np.multiply(w, np.multiply(c_obs, c_obs))) + eps)
        return float(num / den), q_out, c_fit

    log_f0 = float(np.log(f0))
    log_v0 = float(np.log(v0))
    log_cin0 = float(np.log(cin0))

    ### Loss in log space with quadratic regularization around priors
    def loss(theta):
        log_r = float(theta[0])
        log_tau = float(theta[1])
        log_v = float(theta[2])
        log_f = float(theta[3])
        log_cin = float(theta[4])

        r = float(np.exp(log_r))
        tau = float(np.exp(log_tau))
        v = float(np.exp(log_v))
        f_in = float(np.exp(log_f))
        cin_bar = float(np.exp(log_cin))

        val, _, _ = data_mse(r, tau, v, f_in, cin_bar)

        d_r = float(np.subtract(log_r, log_r0))
        reg_r = float(reg_lambda_logr) * d_r * d_r

        d_f = float(np.subtract(log_f, log_f0))
        d_v = float(np.subtract(log_v, log_v0))
        d_cin = float(np.subtract(log_cin, log_cin0))

        reg = float(lambda_f) * d_f * d_f + float(lambda_v) * d_v * d_v + float(lambda_in) * d_cin * d_cin
        return float(val + reg_r + reg)

    tau0 = float(max(min(0.30 * t_end, tau_max), tau_min))
    theta0 = np.array([log_r0, np.log(tau0), log_v0, log_f0, log_cin0], dtype=np.float64)

    r_lo = 1.0 / 1e10
    r_hi = float(r_max)
    if r_hi <= r_lo:
        r_hi = float(r_lo * 10.0)

    tau_lo = float(tau_min)
    tau_hi = float(tau_max)
    if tau_hi <= tau_lo:
        tau_hi = float(tau_lo + 10.0 * float(dt))

    f_lo = float(max(f0 * 0.20, 1.0 / 1e6))
    f_hi = float(max(f0 * 5.00, f_lo * 1.10))

    v_lo = float(max(v0 * 0.20, 1.0 / 1e6))
    v_hi = float(max(v0 * 5.00, v_lo * 1.10))

    cin_lo = float(max(cin0 * 0.10, 1.0 / 1e6))
    cin_hi = float(max(cin0 * 10.0, cin_lo * 1.10))

    bounds = (
        (np.log(r_lo), np.log(r_hi)),
        (np.log(tau_lo), np.log(tau_hi)),
        (np.log(v_lo), np.log(v_hi)),
        (np.log(f_lo), np.log(f_hi)),
        (np.log(cin_lo), np.log(cin_hi)),
    )

    res = minimize(loss, theta0, method="Powell", bounds=bounds)

    r = float(np.exp(res.x[0]))
    tau = float(np.exp(res.x[1]))
    v = float(np.exp(res.x[2]))
    f_in = float(np.exp(res.x[3]))
    cin_bar = float(np.exp(res.x[4]))

    val, q_out, c_fit = data_mse(r, tau, v, f_in, cin_bar)

    ok = bool(res.success) and (q_out is not None) and np.all(np.isfinite(q_out)) and (c_fit is not None) and np.all(np.isfinite(c_fit))

    return {
        "R": r,
        "tau": tau,
        "C_wk": float(tau / r),
        "V": v,
        "F": f_in,
        "Cin_bar": cin_bar,
        "loss": float(val),
        "success": bool(ok),
        "message": str(res.message),
        "Q_out": q_out.astype(np.float64) if q_out is not None else None,
        "C_fit": c_fit.astype(np.float64) if c_fit is not None else None,
    }
