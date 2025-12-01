# ddsp_tubescream_softplus_noninv
```python
# =============================================================================
# ddsp_vmap_tubescreamer_softplus_noninv.py
#
# Fully differentiable, GDSP-style Tube Screamer non-inverting stage
# ------------------------------------------------------------------
# - Pure JAX (functional, no classes, no dicts)
# - State = tuple, Params = tuple
# - Models a simplified non-inverting op-amp with anti-parallel diodes
#   in the feedback path, Tube Screamer style.
# - Diode pair: smooth "softplus-diode" surrogate (Option B)
# - Static non-inverting stage (no explicit op-amp dynamics), wrapped
#   with simple pre/post 1-pole filters for tone shaping.
#
# Topology (conceptual):
#
#   x --[pre RC]--> v_plus ----> ideal op-amp (+)
#                           |
#                           v
#                        Vout = y
#                           |
#                           +---[Rf]---+
#                           |          |
#                           +-->|--<|--+
#                           |          |
#                          [Rg]
#                           |
#                          GND
#
# Ideal op-amp constraints:
#   V- ≈ V+  (inverting input equals non-inverting input)
#
# Node equation at inverting input:
#
#   (V+ / Rg) = (y - V+)/Rf + i_D(y - V+)
#
# => F(y) = (y - V+)/Rf + i_D(y - V+) - V+/Rg = 0
#
# We solve F(y) = 0 for y via fixed Newton iterations.
#
# Diode pair current (softplus surrogate):
#
#   v_d = y - V+
#   z   = v_d / (nVt)
#   i_D = Is * (softplus(z) - softplus(-z))
#
# with stable softplus:
#
#   softplus(z) = log1p(exp(-|z|)) + max(z, 0)
#
# Fully differentiable, JIT/vmap-friendly.
# =============================================================================

import jax.numpy as jnp
from jax import lax, vmap, jit  # jit not strictly required but allowed

# =============================================================================
# PARAM / STATE LAYOUT
# =============================================================================
#
# params = (
#   fs,              # sample rate
#   drive_raw,       # user drive param (any real)
#   tone_raw,        # user tone param (any real)
#   smooth,          # smoothing factor in [0,1] (currently unused, placeholder)
#   Is,              # diode saturation current
#   inv_nVt,         # 1 / (n * Vt)
#   diode_v_clip,    # clamp for diode exponent argument
#   pre_fc_hz,       # pre-EQ 1-pole LP cutoff
#   tone_min_hz,     # min cutoff for post tone LP
#   tone_max_hz,     # max cutoff for post tone LP
#   eps,             # small epsilon for Newton stability
#   newton_iters,    # fixed number of Newton iterations
# )
#
# state = (
#   env,             # envelope / smoothing state (placeholder)
#   diode_state,     # placeholder (not used; kept for future WDF integration)
#   opamp_state,     # (pre_lp_state, post_lp_state)
# )
#
# opamp_state = (
#   pre_lp_y,        # last output of pre 1-pole LP
#   post_lp_y,       # last output of post 1-pole LP
# )
# =============================================================================


# -----------------------------------------------------------------------------
# INIT
# -----------------------------------------------------------------------------

def ddsp_vmap_tubescreamer_noninv_init(
    fs,
    drive=0.0,
    tone=0.0,
    smooth=0.0,
    Is=1e-12,
    n=1.8,
    Vt=0.02585,
    diode_v_clip=8.0,
    pre_fc_hz=720.0,
    tone_min_hz=500.0,
    tone_max_hz=5000.0,
    eps=1e-8,
    newton_iters=3,
    state_shape=(),
):
    """
    Initialize params and state for the Tube Screamer non-inverting block.

    Args:
      fs           : sample rate (Hz)
      drive        : drive param (any real); mapped via sigmoid -> gain ratio
      tone         : tone param (any real); mapped -> LP cutoff between tone_min/max
      smooth       : 0..1 smoothing factor (placeholder; not used yet)
      Is, n, Vt    : diode model parameters
      diode_v_clip : limit for diode exponent argument
      pre_fc_hz    : pre-EQ LP cutoff (Hz)
      tone_min_hz  : min tone LP cutoff (Hz)
      tone_max_hz  : max tone LP cutoff (Hz)
      eps          : epsilon for Newton denominator regularization
      newton_iters : fixed number of Newton iterations
      state_shape  : broadcast shape for states (e.g. (), (B,), ...)

    Returns:
      params, state
    """
    fs = jnp.float32(fs)
    drive_raw = jnp.float32(drive)
    tone_raw = jnp.float32(tone)
    smooth = jnp.float32(smooth)
    Is = jnp.float32(Is)
    Vt = jnp.float32(Vt)
    n = jnp.float32(n)
    diode_v_clip = jnp.float32(diode_v_clip)
    pre_fc_hz = jnp.float32(pre_fc_hz)
    tone_min_hz = jnp.float32(tone_min_hz)
    tone_max_hz = jnp.float32(tone_max_hz)
    eps = jnp.float32(eps)

    inv_nVt = jnp.float32(1.0) / (n * Vt)

    params = (
        fs,
        drive_raw,
        tone_raw,
        smooth,
        Is,
        inv_nVt,
        diode_v_clip,
        pre_fc_hz,
        tone_min_hz,
        tone_max_hz,
        eps,
        jnp.int32(newton_iters),
    )

    zeros = jnp.zeros(state_shape, dtype=jnp.float32)
    env = zeros               # placeholder env
    diode_state = zeros       # placeholder diode_state (not used here)
    pre_lp_y = zeros          # pre-filter LP state
    post_lp_y = zeros         # post-filter LP state
    opamp_state = (pre_lp_y, post_lp_y)

    state = (env, diode_state, opamp_state)
    return params, state


def ddsp_vmap_tubescreamer_noninv_update_state(state, params):
    """
    Placeholder state updater (e.g., could be used for slow envelope or
    parameter smoothing). Currently an identity.
    """
    return state


# -----------------------------------------------------------------------------
# HELPER: Softplus and sigmoid (JAX-only, no jax.nn)
# -----------------------------------------------------------------------------

def _softplus(x):
    """
    Numerically stable softplus:
        softplus(x) = log(1 + exp(x))
    Implemented with:
        softplus(x) = log1p(exp(-|x|)) + max(x, 0)
    """
    abs_x = jnp.abs(x)
    return jnp.log1p(jnp.exp(-abs_x)) + jnp.maximum(x, 0.0)


def _sigmoid(x):
    # 1 / (1 + exp(-x))
    return 1.0 / (1.0 + jnp.exp(-x))


# -----------------------------------------------------------------------------
# PARAM MAPPINGS
# -----------------------------------------------------------------------------

def _compute_drive_gain(drive_raw):
    """
    Map drive_raw (any real) -> feedback ratio Rf/Rg in [1, 10]:

      gain_ratio = 1 + 9 * sigmoid(drive_raw)

    This keeps the feedback gain moderate and stable.
    """
    s = _sigmoid(drive_raw)
    gain_ratio = 1.0 + 9.0 * s
    return gain_ratio  # ~1..10


def _compute_tone_alpha(tone_raw, fs, tone_min_hz, tone_max_hz):
    """
    Map tone_raw (any real) -> 1-pole LP smoothing factor alpha in (0,1):

      tone_norm = sigmoid(tone_raw) in (0,1)
      fc = tone_min_hz * exp(log(tone_max/tone_min) * tone_norm)
      alpha = exp(-2*pi*fc / fs)

    Used for post-LP tone filter; alpha near 1 => darker sound.
    """
    tone_norm = _sigmoid(tone_raw)
    tone_norm = jnp.clip(tone_norm, jnp.float32(1e-3), jnp.float32(1.0 - 1e-3))

    log_ratio = jnp.log(tone_max_hz / tone_min_hz)
    fc = tone_min_hz * jnp.exp(log_ratio * tone_norm)

    alpha = jnp.exp(-2.0 * jnp.pi * fc / fs)
    return alpha


def _compute_pre_lp_alpha(pre_fc_hz, fs):
    """
    Pre 1-pole LP coefficient:

      alpha = exp(-2*pi*fc/fs)
    """
    alpha = jnp.exp(-2.0 * jnp.pi * pre_fc_hz / fs)
    return alpha


# -----------------------------------------------------------------------------
# DIODE PAIR SOFTPLUS MODEL
# -----------------------------------------------------------------------------

def _diode_pair_current(vd, Is, inv_nVt, v_clip):
    """
    Symmetric anti-parallel diode pair current using softplus surrogate:

      z = clamp(vd * inv_nVt, -v_clip, v_clip)
      i_D = Is * ( softplus(z) - softplus(-z) )

    This is odd in vd, smooth, and approximates diode conduction with a
    soft knee.
    """
    z = vd * inv_nVt
    z = jnp.clip(z, -v_clip, v_clip)
    i_pos = _softplus(z)
    i_neg = _softplus(-z)
    i_D = Is * (i_pos - i_neg)
    return i_D


def _diode_pair_current_deriv(vd, Is, inv_nVt, v_clip):
    """
    Derivative di_D / dvd.

    Using:
      z = vd * inv_nVt
      i_D(z) = Is(softplus(z) - softplus(-z))

      d softplus(z)/dz = sigmoid(z)
      d softplus(-z)/dz = -sigmoid(-z)

    => di_D/dz = Is (sigmoid(z) + sigmoid(-z))
       di_D/dvd = di_D/dz * dz/dvd = Is*(sigmoid(z) + sigmoid(-z))*inv_nVt
    """
    z = vd * inv_nVt
    z = jnp.clip(z, -v_clip, v_clip)

    sig_z = _sigmoid(z)
    sig_neg_z = _sigmoid(-z)

    di_dz = Is * (sig_z + sig_neg_z)
    di_dvd = di_dz * inv_nVt
    return di_dvd


# -----------------------------------------------------------------------------
# NEWTON SOLVE FOR y (OP-AMP OUTPUT)
# -----------------------------------------------------------------------------

def _newton_solve_opamp_output(v_plus, state, params):
    """
    Solve for op-amp output y given non-inverting input v_plus.

    Equation at inverting input (ideal op-amp, V- = V+):

      (v_plus / Rg) = (y - v_plus)/Rf + i_D(y - v_plus)

    Let Rg = 1, Rf = gain_ratio (derived from drive).

    => F(y) = (y - v_plus)/Rf + i_D(y - v_plus) - v_plus/Rg = 0.

    We use fixed Newton iterations:

      y_{n+1} = y_n - F(y_n)/F'(y_n)

    with:
      vd = y - v_plus
      F(y) = (vd)/Rf + i_D(vd) - v_plus
      dF/dy = 1/Rf + di_D/dvd
    """
    (
        fs,
        drive_raw,
        tone_raw,
        smooth,
        Is,
        inv_nVt,
        v_clip,
        pre_fc_hz,
        tone_min_hz,
        tone_max_hz,
        eps,
        newton_iters,
    ) = params

    gain_ratio = _compute_drive_gain(drive_raw)
    Rf = gain_ratio
    Rg = jnp.float32(1.0)

    # Initial guess: linear non-inverting gain
    y_init = v_plus * (1.0 + Rf / Rg)

    def body_fun(i, y_curr):
        vd = y_curr - v_plus
        iD = _diode_pair_current(vd, Is, inv_nVt, v_clip)
        diD_dvd = _diode_pair_current_deriv(vd, Is, inv_nVt, v_clip)

        F = vd / Rf + iD - v_plus  # (y - v_plus)/Rf + iD - v_plus
        dF = (1.0 / Rf) + diD_dvd

        dF_safe = jnp.where(jnp.abs(dF) < eps,
                            jnp.sign(dF) * eps,
                            dF)

        y_next = y_curr - F / dF_safe
        return y_next

    y_final = lax.fori_loop(0, newton_iters, body_fun, y_init)
    return y_final


# -----------------------------------------------------------------------------
# TICK
# -----------------------------------------------------------------------------

def ddsp_vmap_tubescreamer_noninv_tick(x_t, state, params):
    """
    Single-sample tick for the non-inverting Tube Screamer-style stage.

    Args:
      x_t   : scalar input (float32)
      state : (env, diode_state, opamp_state)
      params: params tuple

    Returns:
      y_t, new_state
    """
    (env, diode_state, opamp_state) = state
    (pre_lp_y, post_lp_y) = opamp_state

    (
        fs,
        drive_raw,
        tone_raw,
        smooth,
        Is,
        inv_nVt,
        v_clip,
        pre_fc_hz,
        tone_min_hz,
        tone_max_hz,
        eps,
        newton_iters,
    ) = params

    x_t = jnp.float32(x_t)

    # --- Pre LP (input shaping) ----------------------------------------------
    pre_alpha = _compute_pre_lp_alpha(pre_fc_hz, fs)
    v_plus = (1.0 - pre_alpha) * x_t + pre_alpha * pre_lp_y
    pre_lp_y_new = v_plus

    # --- Non-inverting op-amp + diode feedback --------------------------------
    y_opamp = _newton_solve_opamp_output(v_plus, state, params)

    # --- Post tone LP ---------------------------------------------------------
    tone_alpha = _compute_tone_alpha(tone_raw, fs, tone_min_hz, tone_max_hz)
    y_post = (1.0 - tone_alpha) * y_opamp + tone_alpha * post_lp_y
    post_lp_y_new = y_post

    # env and diode_state currently pass-through (placeholders)
    env_new = env
    diode_state_new = diode_state
    opamp_state_new = (pre_lp_y_new, post_lp_y_new)
    new_state = (env_new, diode_state_new, opamp_state_new)

    return y_post, new_state


# -----------------------------------------------------------------------------
# PROCESS (TIME SEQUENCE)
# -----------------------------------------------------------------------------

def ddsp_vmap_tubescreamer_noninv_process(x, state, params):
    """
    Process a 1-D input signal x with lax.scan.

    Args:
      x     : (T,) float32
      state : state tuple
      params: params tuple

    Returns:
      y: (T,) output
      final_state
    """
    def body_fn(carry, x_t):
        s = carry
        y_t, s_new = ddsp_vmap_tubescreamer_noninv_tick(x_t, s, params)
        return s_new, y_t

    final_state, y = lax.scan(body_fn, state, x)
    return y, final_state


# -----------------------------------------------------------------------------
# VMAP (BATCH)
# -----------------------------------------------------------------------------

def ddsp_vmap_tubescreamer_noninv_vmap(x_batch, state_batch, params_batch):
    """
    Batch wrapper over process() using vmap.

    Args:
      x_batch      : (B, T)
      state_batch  : state tuple, each leaf has leading dim B
      params_batch : params tuple, each leaf has leading dim B or is broadcastable

    Returns:
      y_batch: (B, T)
      state_batch_out
    """
    def process_one(x, s, p):
        return ddsp_vmap_tubescreamer_noninv_process(x, s, p)

    yb, sb = vmap(process_one)(x_batch, state_batch, params_batch)
    return yb, sb


# -----------------------------------------------------------------------------
# __main__ : smoke test + plotting + audio demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import soundfile as sf
    import sounddevice as sd

    print("=== Smoke Test: ddsp_vmap_tubescreamer_noninv ===")

    fs = 48000.0

    params, state = ddsp_vmap_tubescreamer_noninv_init(
        fs=fs,
        drive=0.0,
        tone=0.0,
        smooth=0.0,
        Is=1e-12,
        n=1.8,
        Vt=0.02585,
        diode_v_clip=8.0,
        pre_fc_hz=720.0,
        tone_min_hz=500.0,
        tone_max_hz=5000.0,
        eps=1e-8,
        newton_iters=3,
        state_shape=(),
    )

    # Small test signal (ramp)
    T = 512
    x = jnp.linspace(-0.5, 0.5, T).astype(jnp.float32)

    y, state_out = ddsp_vmap_tubescreamer_noninv_process(x, state, params)

    print("Output stats:")
    print("  min:", float(y.min()))
    print("  max:", float(y.max()))
    print("  mean:", float(y.mean()))

    # -------------------------------------------------------------------------
    # Static waveshaping view (input vs output)
    # -------------------------------------------------------------------------

    x_grid = jnp.linspace(-1.0, 1.0, 1024).astype(jnp.float32)
    # Reinit state to zeros
    _, state0 = ddsp_vmap_tubescreamer_noninv_init(
        fs=fs,
        drive=1.0,
        tone=0.0,
        smooth=0.0,
        Is=1e-12,
        n=1.8,
        Vt=0.02585,
        diode_v_clip=8.0,
        pre_fc_hz=720.0,
        tone_min_hz=500.0,
        tone_max_hz=5000.0,
        eps=1e-8,
        newton_iters=3,
        state_shape=(),
    )
    y_grid, _ = ddsp_vmap_tubescreamer_noninv_process(x_grid, state0, params)

    plt.figure(figsize=(5, 5))
    plt.plot(np.array(x_grid), np.array(y_grid))
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("Static-ish waveshaping curve (drive=0)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Example: frequency response (approx small-signal)
    # -------------------------------------------------------------------------

    print("=== Approx small-signal response ===")
    N = 4096
    impulse = jnp.zeros((N,), dtype=jnp.float32).at[0].set(1e-4)
    _, state_lin = ddsp_vmap_tubescreamer_noninv_init(
        fs=fs,
        drive=-3.0,  # very low drive -> quasi-linear
        tone=0.0,
        smooth=0.0,
        Is=1e-12,
        n=1.8,
        Vt=0.02585,
        diode_v_clip=8.0,
        pre_fc_hz=720.0,
        tone_min_hz=500.0,
        tone_max_hz=5000.0,
        eps=1e-8,
        newton_iters=3,
        state_shape=(),
    )
    y_imp, _ = ddsp_vmap_tubescreamer_noninv_process(impulse, state_lin, params)

    Y = np.fft.rfft(np.array(y_imp))
    freq = np.fft.rfftfreq(N, 1.0 / fs)
    mag = 20.0 * np.log10(np.abs(Y) + 1e-12)

    plt.figure(figsize=(8, 4))
    plt.plot(freq, mag)
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("ddsp_vmap_tubescreamer_noninv – Approx small-signal response")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Audio demo (input.wav)
    # -------------------------------------------------------------------------

    try:
        print("=== Audio Demo: input.wav ===")
        x_audio, fs_in = sf.read("input.wav", dtype="float32")
        if fs_in != fs:
            print(f"Warning: expected fs={fs}, got fs_in={fs_in}. Using fs_in.")

        if x_audio.ndim > 1:
            x_audio = x_audio[:, 0]

        x_jax = jnp.array(x_audio, dtype=jnp.float32)

        # Reinit state for audio
        params_audio, state_audio = ddsp_vmap_tubescreamer_noninv_init(
            fs=fs,
            drive=2.0,   # higher drive
            tone=0.5,
            smooth=0.0,
            Is=1e-12,
            n=1.8,
            Vt=0.02585,
            diode_v_clip=8.0,
            pre_fc_hz=720.0,
            tone_min_hz=500.0,
            tone_max_hz=5000.0,
            eps=1e-8,
            newton_iters=4,
            state_shape=(),
        )

        y_audio, _ = ddsp_vmap_tubescreamer_noninv_process(
            x_jax, state_audio, params_audio
        )

        # Normalize for safety
        y_audio = y_audio / (jnp.max(jnp.abs(y_audio)) + 1e-6)
        y_np = np.array(y_audio)

        print("Playing dry...")
        sd.play(x_audio, fs_in)
        sd.wait()

        print("Playing wet...")
        sd.play(y_np, fs_in)
        sd.wait()

    except Exception as e:
        print("Audio demo skipped:", e)

```
