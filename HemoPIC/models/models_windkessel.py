import numpy as np


### Windkessel outflow given resistence (R) and compilance C (tau = RC) with initial pressure p0
### return closed form solution for outflow q(t) under steady inflow condition
def windkessel_outflow_from_r_tau(t, q_in, r, tau, p0):
    tau = float(tau)
    if (not np.isfinite(tau)) or tau <= 0.0:
        return np.full_like(t, np.nan, dtype=np.float64)

    e = np.exp(np.divide(np.negative(t), tau))
    q0 = float(p0) / float(r)
    return q_in + np.multiply(np.subtract(q0, q_in), e)
