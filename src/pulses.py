# src/pulses/genetic_ansatz.py
import jax
import jax.numpy as jnp
import numpy as np
from rydopt.pulses.softbox_pulse_ansatz_functions import softbox_hann, softbox_blackman

EPS = 1e-6
NUM_BASE_FUNCS = 7   # sin, cos, const, softbox_hann, softbox_blackman, gaussian, lorentzian

# ----------------------------------------------------------------------
# JAX-функция для использования внутри evolve (быстрая, безопасная)
# ----------------------------------------------------------------------
def genetic_sum_ansatz(t, duration, ansatz_params):
    """
    Безопасная версия генетического анзаца.
    """
    t_arr = jnp.asarray(t)
    was_scalar = (t_arr.ndim == 0)
    if was_scalar:
        t_arr = t_arr[None]

    T = duration
    n_params = len(ansatz_params)
    K = n_params // 4
    if n_params % 4 != 0:
        raise ValueError(f"ansatz_params length must be multiple of 4, got {n_params}")

    params_reshaped = ansatz_params.reshape(K, 4)
    types_raw = params_reshaped[:, 0]
    amps = params_reshaped[:, 1]
    mods = params_reshaped[:, 2]
    phases = params_reshaped[:, 3]

    # Замена NaN
    types_raw = jnp.nan_to_num(types_raw, nan=0.0)
    amps = jnp.nan_to_num(amps, nan=0.0)
    mods = jnp.nan_to_num(mods, nan=0.0)
    phases = jnp.nan_to_num(phases, nan=0.0)

    # Преобразование типа [-1,1] -> индекс
    types = jnp.round((types_raw + 1) / 2 * (NUM_BASE_FUNCS - 1)).astype(jnp.int32)
    types = jnp.clip(types, 0, NUM_BASE_FUNCS - 1)

    result = jnp.zeros_like(t_arr)

    for i in range(K):
        typ = types[i]
        amp = amps[i]
        mod = mods[i]
        phase = phases[i]

        # Безопасные значения
        safe_mod_pos = jnp.abs(mod) + EPS
        safe_alpha = jnp.clip(mod, EPS, 1.0 - EPS)

        def f_sin(x):
            freq = mod * (2 * jnp.pi / T)
            return amp * jnp.sin(freq * x + phase)

        def f_cos(x):
            freq = mod * (2 * jnp.pi / T)
            return amp * jnp.cos(freq * x + phase)

        def f_const(x):
            return jnp.full_like(x, amp)

        def f_softbox_hann(x):
            window = softbox_hann(x, T, jnp.array([1.0, safe_alpha]))
            return amp * window

        def f_softbox_blackman(x):
            window = softbox_blackman(x, T, jnp.array([1.0, safe_alpha]))
            return amp * window

        def f_gaussian(x):
            sigma = safe_mod_pos * T
            shift = phase * T
            return amp * jnp.exp(-((x - shift) / sigma) ** 2)

        def f_lorentzian(x):
            gamma = safe_mod_pos * T
            shift = phase * T
            return amp / (1 + ((x - shift) / gamma) ** 2)

        branch_functions = [f_sin, f_cos, f_const, f_softbox_hann, f_softbox_blackman,
                            f_gaussian, f_lorentzian]

        result += jax.lax.switch(typ, branch_functions, t_arr)

    if was_scalar:
        return result[0]
    return result

# ----------------------------------------------------------------------
# Быстрое текстовое представление (без SymPy)
# ----------------------------------------------------------------------
def genes_to_expression_string(ansatz_params, time_sym='t', duration_sym='T', precision=4):
    """
    Преобразует вектор генов в строку с математическим выражением.
    Очень быстро, не использует SymPy.
    """
    n_params = len(ansatz_params)
    K = n_params // 4
    if n_params % 4 != 0:
        raise ValueError(f"ansatz_params length must be multiple of 4, got {n_params}")

    params_reshaped = np.asarray(ansatz_params).reshape(K, 4)
    types_raw = params_reshaped[:, 0]
    amps = params_reshaped[:, 1]
    mods = params_reshaped[:, 2]
    phases = params_reshaped[:, 3]

    def fmt(x):
        return f"{x:.{precision}g}" if abs(x) >= 1e-4 else f"{x:.{precision}e}"

    def type_to_idx(val):
        v = max(-1.0, min(1.0, val))
        return int(round((v + 1) / 2 * (NUM_BASE_FUNCS - 1)))

    terms = []
    for i in range(K):
        typ_idx = type_to_idx(types_raw[i])
        amp = amps[i]
        mod = mods[i]
        phase = phases[i]

        if abs(amp) < 1e-12:
            continue

        amp_str = fmt(amp)

        if typ_idx == 0:      # sin
            freq = mod
            phase_str = f"+{fmt(phase)}" if phase >= 0 else f"{fmt(phase)}"
            term = f"{amp_str}*sin(2π*{fmt(freq)}*{time_sym}/{duration_sym}{phase_str})"
        elif typ_idx == 1:    # cos
            freq = mod
            phase_str = f"+{fmt(phase)}" if phase >= 0 else f"{fmt(phase)}"
            term = f"{amp_str}*cos(2π*{fmt(freq)}*{time_sym}/{duration_sym}{phase_str})"
        elif typ_idx == 2:    # const
            term = amp_str
        elif typ_idx == 3:    # softbox_hann
            alpha = mod
            term = f"{amp_str}*softbox_hann({time_sym}, {duration_sym}, alpha={fmt(alpha)})"
        elif typ_idx == 4:    # softbox_blackman
            alpha = mod
            term = f"{amp_str}*softbox_blackman({time_sym}, {duration_sym}, alpha={fmt(alpha)})"
        elif typ_idx == 5:    # gaussian
            sigma = abs(mod) + EPS
            shift = phase
            term = f"{amp_str}*exp(-(({time_sym} - {fmt(shift)}*{duration_sym})/({fmt(sigma)}*{duration_sym}))^2)"
        elif typ_idx == 6:    # lorentzian
            gamma = abs(mod) + EPS
            shift = phase
            term = f"{amp_str}/(1+(({time_sym} - {fmt(shift)}*{duration_sym})/({fmt(gamma)}*{duration_sym}))^2)"
        else:
            continue

        terms.append(term)

    if not terms:
        return "0"
    return " + ".join(terms)

# ----------------------------------------------------------------------
# Медленная символьная версия (оставлена для желающих, но не рекомендуется)
# ----------------------------------------------------------------------
def genes_to_sympy_expr(ansatz_params, time_sym='t', duration_sym='T'):
    """
    Преобразует вектор генов в SymPy-выражение. МЕДЛЕННО, используйте только для финального вывода.
    """
    import sympy as sp
    T_sym = sp.Symbol(duration_sym, real=True, positive=True)
    t_sym = sp.Symbol(time_sym, real=True)

    params = np.asarray(ansatz_params, dtype=float)
    params = np.nan_to_num(params, nan=0.0)
    n_params = len(params)
    K = n_params // 4
    if n_params % 4 != 0:
        raise ValueError(f"ansatz_params length must be multiple of 4, got {n_params}")

    params_reshaped = params.reshape(K, 4)
    types_raw = params_reshaped[:, 0]
    amps = params_reshaped[:, 1]
    mods = params_reshaped[:, 2]
    phases = params_reshaped[:, 3]

    def type_to_idx(val):
        v = max(-1.0, min(1.0, val))
        return int(round((v + 1) / 2 * (NUM_BASE_FUNCS - 1)))

    total_expr = 0
    for i in range(K):
        typ_idx = type_to_idx(types_raw[i])
        amp = amps[i]
        mod = mods[i]
        phase = phases[i]

        if abs(amp) < 1e-12:
            continue

        if typ_idx == 0:      # sin
            expr = amp * sp.sin(2 * sp.pi * mod * t_sym / T_sym + phase)
        elif typ_idx == 1:    # cos
            expr = amp * sp.cos(2 * sp.pi * mod * t_sym / T_sym + phase)
        elif typ_idx == 2:    # const
            expr = amp
        elif typ_idx == 3:    # softbox_hann
            alpha = float(mod)
            if alpha <= 0:
                expr = amp
            elif alpha >= 1:
                expr = amp * sp.Piecewise((0, t_sym < 0), (1, (t_sym >= 0) & (t_sym <= T_sym)), (0, True))
            else:
                half = alpha * T_sym / 2
                if half <= 0:
                    expr = amp
                else:
                    x_rise = t_sym / half
                    x_fall = (T_sym - t_sym) / half
                    rise = 0.5 - 0.5 * sp.cos(2 * sp.pi * x_rise)
                    fall = 0.5 - 0.5 * sp.cos(2 * sp.pi * x_fall)
                    expr = amp * sp.Piecewise(
                        (rise, t_sym < half),
                        (1, (t_sym >= half) & (t_sym <= T_sym - half)),
                        (fall, t_sym > T_sym - half),
                        (0, True)
                    )
        elif typ_idx == 4:    # softbox_blackman
            alpha = float(mod)
            if alpha <= 0:
                expr = amp
            elif alpha >= 1:
                expr = amp * sp.Piecewise((0, t_sym < 0), (1, (t_sym >= 0) & (t_sym <= T_sym)), (0, True))
            else:
                half = alpha * T_sym / 2
                if half <= 0:
                    expr = amp
                else:
                    x_rise = t_sym / half
                    x_fall = (T_sym - t_sym) / half
                    rise = 0.42 - 0.5*sp.cos(2*sp.pi*x_rise) + 0.08*sp.cos(4*sp.pi*x_rise)
                    fall = 0.42 - 0.5*sp.cos(2*sp.pi*x_fall) + 0.08*sp.cos(4*sp.pi*x_fall)
                    expr = amp * sp.Piecewise(
                        (rise, t_sym < half),
                        (1, (t_sym >= half) & (t_sym <= T_sym - half)),
                        (fall, t_sym > T_sym - half),
                        (0, True)
                    )
        elif typ_idx == 5:    # gaussian
            sigma = abs(mod) * T_sym + EPS * T_sym
            shift = phase * T_sym
            expr = amp * sp.exp(-((t_sym - shift) / sigma) ** 2)
        elif typ_idx == 6:    # lorentzian
            gamma = abs(mod) * T_sym + EPS * T_sym
            shift = phase * T_sym
            expr = amp / (1 + ((t_sym - shift) / gamma) ** 2)
        else:
            expr = 0

        total_expr += expr

    return sp.simplify(total_expr)

def genetic_ansatz_dimension(num_genes: int) -> int:
    """Возвращает длину плоского вектора для заданного количества генов."""
    return num_genes * 4