import numpy as np


### Forward simulate tracer concentration with a simple compartment update
def tracer_update(q_out, c_in_t, v_const, q_in, dt, q_floor_ratio):
    t_len = int(len(q_out))
    ct = np.zeros(t_len, dtype=np.float64)

    v = float(v_const) if np.isfinite(v_const) and v_const > 0.0 else (1.0 / 1e6)

    # non negative, finite inflow
    q_in_safe = float(q_in) if np.isfinite(q_in) and q_in >= 0.0 else 0.0

    # prevent (q_in_safe / q_eff) c_in term from blowing up (used in the loop below for updating ct[n+1] from ct[n])
    q_floor = float(max(float(q_floor_ratio) * q_in_safe, 1.0 / 1e12))
    
    # later update ct[n+1] << ct[n] (last valid index is ct[t_len-1])
    t_last = int(np.subtract(t_len, 1))

    for n in range(int(t_last)):
        q = float(q_out[n])
        if (not np.isfinite(q)) or q < 0.0:
            q = 0.0
        
        # exponential decay constant k
        q_eff = q if q > q_floor else q_floor
        k = q_eff / v

        # decay over one time step
        kd = k * float(dt)

        if kd > 60.0:
            a = 0.0
        else:
            a = float(np.exp(np.negative(kd)))

        one_minus_a = float(np.subtract(1.0, a))
        drive = (q_in_safe / q_eff) * float(c_in_t[n]) if q_in_safe > 0.0 else 0.0
        ct[n + 1] = ct[n] * a + drive * one_minus_a

    return ct
