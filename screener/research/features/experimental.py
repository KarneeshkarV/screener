"""Categories 7 and 8: advanced trend estimation and spectral features.

Everything here is research-grade and deliberately not wired into any screen.
The point of the category is to test whether a heavier estimator beats a cheap
one such as efficiency ratio or regression R-squared, and the honest prior is
that most will not.

Each of these has a textbook form that is **not causal**, so each is
reformulated here. That reformulation is the whole reason this module is long,
and it is called out per feature:

- **Kalman**: the forward filtering pass only. No RTS smoother, because
  smoothing revises past states using later observations.
- **Savitzky-Golay**: the textbook filter is centered. Here the polynomial is
  fitted on a trailing window and evaluated at that window's right edge, which
  turns it into a one-sided FIR filter.
- **Wavelet**: the standard undecimated transform uses symmetric filters. Here
  the cascade uses trailing means only, so every scale looks strictly backward.
- **Spectral**: the FFT is taken on a trailing window, never on the whole
  series, and nothing is reconstructed. Fourier reconstruction over the full
  sample is exactly the look-ahead the brief bans.
- **L1 trend filter**: the textbook problem is a whole-sample optimization. Here
  it is solved on a trailing window and only the endpoint is kept.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from screener.research.features.base import FeatureCtx, feature, safe_ratio

# ── Kalman local linear trend ────────────────────────────────────────


def _kalman_local_linear(
    y: np.ndarray, *, obs_var: np.ndarray, ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward Kalman pass over a local-linear-trend model.

    State is ``[level, slope]`` with transition ``[[1, 1], [0, 1]]`` and a
    level-only observation. Returns filtered level, filtered slope and the
    posterior standard deviation of the level.

    Filtering only: state ``t`` is conditioned on observations ``<= t``. A
    smoother would condition on the whole sample and is why this is hand-rolled
    rather than taken from a library default.

    ``obs_var`` is per-bar and must itself be trailing. A single scalar fitted
    over the whole series would be look-ahead: it would let the noise scale at
    bar 10 depend on the volatility of bar 900.
    """
    n = y.shape[0]
    level = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    sigma = np.full(n, np.nan)

    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    process = np.zeros((2, 2))
    state = np.zeros(2)
    covariance = np.eye(2) * 1e6  # diffuse prior: the first observations set it
    started = False

    for i in range(n):
        observation = y[i]
        r = obs_var[i]
        if not np.isfinite(r) or r <= 0.0:
            # Noise scale not yet estimable from trailing data: emit nothing
            # rather than fall back to a constant borrowed from the future.
            continue
        process[1, 1] = r * ratio
        if not np.isfinite(observation):
            # Missing bar: propagate, do not update. The filter coasts on its
            # own dynamics rather than inventing an observation.
            if started:
                state = transition @ state
                covariance = transition @ covariance @ transition.T + process
                level[i], slope[i] = state[0], state[1]
                sigma[i] = np.sqrt(max(covariance[0, 0], 0.0))
            continue
        if not started:
            state = np.array([observation, 0.0])
            covariance = np.array([[r, 0.0], [0.0, r * ratio * 10.0]])
            started = True
        else:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
        innovation = observation - state[0]
        innovation_var = covariance[0, 0] + r
        gain = covariance[:, 0] / innovation_var
        state = state + gain * innovation
        covariance = covariance - np.outer(gain, covariance[0, :])
        level[i], slope[i] = state[0], state[1]
        sigma[i] = np.sqrt(max(covariance[0, 0], 0.0))
    return level, slope, sigma


def _kalman_frame(ctx: FeatureCtx, ratio: float, obs_window: int) -> pd.DataFrame:
    """Run the filter on log price with observation noise set from trailing vol.

    The noise scale is the name's own trailing return variance, lagged one bar,
    so one ``ratio`` works across the universe without per-stock tuning and
    without any bar informing its own noise estimate.
    """
    log_close = ctx.log_close
    obs_std = (
        log_close.diff()
        .rolling(obs_window, min_periods=obs_window)
        .std(ddof=1)
        .shift(1)
    )
    level, slope, sigma = _kalman_local_linear(
        log_close.to_numpy(dtype=float),
        obs_var=(obs_std.to_numpy(dtype=float) ** 2),
        ratio=ratio,
    )
    return pd.DataFrame(
        {"level": level, "slope": slope, "sigma": sigma}, index=ctx.bars.index
    )


@feature(
    "kalman_slope",
    category="experimental",
    doc="Filtered daily drift from a causal local-linear-trend Kalman filter.",
    params={"ratio": 1e-3, "obs_window": 60},
    grid=({"ratio": 1e-4, "obs_window": 60}, {"ratio": 1e-2, "obs_window": 60}),
    min_lookback=120,
)
def kalman_slope(ctx: FeatureCtx, *, ratio: float, obs_window: int) -> pd.Series:
    return _kalman_frame(ctx, ratio, obs_window)["slope"]


@feature(
    "kalman_slope_snr",
    category="experimental",
    doc="Kalman slope divided by the filter's own level uncertainty.",
    params={"ratio": 1e-3, "obs_window": 60},
    grid=({"ratio": 1e-4, "obs_window": 60}, {"ratio": 1e-2, "obs_window": 60}),
    min_lookback=120,
)
def kalman_slope_snr(ctx: FeatureCtx, *, ratio: float, obs_window: int) -> pd.Series:
    # The filter's own answer to "how much do I trust this trend", which is the
    # thing a plain slope cannot tell you.
    frame = _kalman_frame(ctx, ratio, obs_window)
    return safe_ratio(frame["slope"], frame["sigma"])


@feature(
    "kalman_uncertainty",
    category="experimental",
    doc="Posterior standard deviation of the Kalman level estimate.",
    params={"ratio": 1e-3, "obs_window": 60},
    grid=({"ratio": 1e-4, "obs_window": 60}, {"ratio": 1e-2, "obs_window": 60}),
    min_lookback=120,
    higher_is_stronger=False,
)
def kalman_uncertainty(ctx: FeatureCtx, *, ratio: float, obs_window: int) -> pd.Series:
    return _kalman_frame(ctx, ratio, obs_window)["sigma"]


# ── causal Savitzky-Golay (trailing local polynomial) ────────────────


def _endpoint_polyfit_weights(window: int, degree: int, derivative: int) -> np.ndarray:
    """FIR weights that read a derivative at the right edge of a trailing window.

    Fits ``degree``-order polynomial in ``t = -(window-1)..0`` by least squares
    and returns the row of the pseudo-inverse that evaluates the requested
    derivative at ``t = 0``. Because the design matrix is fixed, this collapses
    to a constant one-sided filter: no per-bar solve, and no centering.
    """
    if derivative > degree:
        raise ValueError("derivative order exceeds polynomial degree")
    t = np.arange(-(window - 1), 1, dtype=float)
    design = np.vander(t, degree + 1, increasing=True)
    pinv = np.linalg.pinv(design)
    # Coefficient `derivative` times factorial gives the derivative at t = 0.
    weights: np.ndarray = pinv[derivative] * float(math.factorial(derivative))
    return weights


def _savgol_causal(
    series: pd.Series, window: int, degree: int, derivative: int
) -> pd.Series:
    weights = _endpoint_polyfit_weights(window, degree, derivative)
    return series.rolling(window, min_periods=window).apply(
        lambda values: float(np.dot(weights, values)), raw=True
    )


@feature(
    "savgol_slope",
    category="experimental",
    doc="First derivative of a trailing local polynomial fit to log price.",
    params={"window": 61, "degree": 2},
    grid=(
        {"window": 21, "degree": 2},
        {"window": 121, "degree": 2},
        {"window": 61, "degree": 3},
    ),
    min_lookback=130,
)
def savgol_slope(ctx: FeatureCtx, *, window: int, degree: int) -> pd.Series:
    return _savgol_causal(ctx.log_close, window, degree, 1)


@feature(
    "savgol_acceleration",
    category="experimental",
    doc="Second derivative of a trailing local polynomial fit to log price.",
    params={"window": 61, "degree": 2},
    grid=({"window": 21, "degree": 2}, {"window": 121, "degree": 3}),
    min_lookback=130,
)
def savgol_acceleration(ctx: FeatureCtx, *, window: int, degree: int) -> pd.Series:
    return _savgol_causal(ctx.log_close, window, degree, 2)


# ── causal Haar wavelet cascade ──────────────────────────────────────


def _causal_haar_details(
    series: pd.Series, levels: int
) -> tuple[list[pd.Series], pd.Series]:
    """Undecimated Haar-style cascade built from trailing means only.

    At each level the smooth is a trailing mean over ``2**level`` bars and the
    detail is what that smoothing removed. The standard a-trous transform uses
    symmetric filters and would read future bars; using a trailing mean keeps
    the multiresolution structure and stays causal.
    """
    details: list[pd.Series] = []
    smooth = series
    for level in range(levels):
        width = 2 ** (level + 1)
        next_smooth = smooth.rolling(width, min_periods=width).mean()
        details.append(smooth - next_smooth)
        smooth = next_smooth
    return details, smooth


@feature(
    "wavelet_lf_ratio",
    category="experimental",
    doc="Share of causal-wavelet energy in the coarse band. High = smooth trend.",
    params={"levels": 5, "window": 120},
    grid=({"levels": 4, "window": 120}, {"levels": 6, "window": 250}),
    min_lookback=400,
)
def wavelet_lf_ratio(ctx: FeatureCtx, *, levels: int, window: int) -> pd.Series:
    # Energy is measured on the detail bands of log price. A clean trend puts
    # its variance in the coarse band; chop puts it in the fine ones.
    details, smooth = _causal_haar_details(ctx.log_close, levels)
    fine = sum(
        (detail**2).rolling(window, min_periods=window).mean() for detail in details[:2]
    )
    coarse = sum(
        (detail**2).rolling(window, min_periods=window).mean() for detail in details[2:]
    )
    trend_energy = (smooth.diff() ** 2).rolling(window, min_periods=window).mean()
    low = coarse + trend_energy
    return safe_ratio(low, low + fine)


@feature(
    "wavelet_hf_energy",
    category="experimental",
    doc="Fine-scale causal-wavelet energy, normalized by total. High = noisy.",
    params={"levels": 5, "window": 120},
    grid=({"levels": 4, "window": 120}, {"levels": 6, "window": 250}),
    min_lookback=400,
    higher_is_stronger=False,
)
def wavelet_hf_energy(ctx: FeatureCtx, *, levels: int, window: int) -> pd.Series:
    return 1.0 - wavelet_lf_ratio(ctx, levels=levels, window=window)


# ── causal rolling spectral features ─────────────────────────────────


def _rolling_spectrum(series: pd.Series, window: int, reducer) -> pd.Series:
    """Apply ``reducer`` to the power spectrum of each trailing window.

    A Hann taper is applied inside the window and the mean is removed, so the
    result describes the shape of the recent oscillation rather than its level.
    Nothing is reconstructed and no FFT ever sees the whole series.
    """
    taper = np.hanning(window)

    def evaluate(values: np.ndarray) -> float:
        centered = values - values.mean()
        spectrum = np.abs(np.fft.rfft(centered * taper)) ** 2
        spectrum = spectrum[1:]  # drop DC, it carries no oscillation information
        total = spectrum.sum()
        if not np.isfinite(total) or total <= 0.0:
            return float("nan")
        return float(reducer(spectrum / total))

    return series.rolling(window, min_periods=window).apply(evaluate, raw=True)


@feature(
    "spectral_lf_ratio",
    category="experimental",
    doc="Share of rolling spectral power below a low-frequency cutoff.",
    params={"window": 128, "cutoff": 0.25},
    grid=(
        {"window": 64, "cutoff": 0.25},
        {"window": 256, "cutoff": 0.25},
        {"window": 128, "cutoff": 0.125},
    ),
    min_lookback=280,
)
def spectral_lf_ratio(ctx: FeatureCtx, *, window: int, cutoff: float) -> pd.Series:
    # The direct spectral analogue of "is this a trend or is this chop", and the
    # head-to-head the brief actually asks for against efficiency ratio.
    def reducer(power: np.ndarray) -> float:
        edge = max(1, int(round(cutoff * power.shape[0])))
        return float(power[:edge].sum())

    return _rolling_spectrum(ctx.log_returns, window, reducer)


@feature(
    "spectral_entropy",
    category="experimental",
    doc="Normalized Shannon entropy of the rolling power spectrum, in [0, 1].",
    params={"window": 128},
    grid=({"window": 64}, {"window": 256}),
    min_lookback=280,
    higher_is_stronger=False,
)
def spectral_entropy(ctx: FeatureCtx, *, window: int) -> pd.Series:
    # 1.0 is white noise with power spread everywhere; low values mean the move
    # is concentrated at a few frequencies.
    def reducer(power: np.ndarray) -> float:
        positive = power[power > 0.0]
        entropy = -float((positive * np.log(positive)).sum())
        return entropy / float(np.log(power.shape[0]))

    return _rolling_spectrum(ctx.log_returns, window, reducer)


@feature(
    "dominant_freq_stability",
    category="experimental",
    doc="Stability of the dominant spectral frequency across recent windows.",
    params={"window": 128, "history": 60},
    grid=({"window": 64, "history": 60}, {"window": 128, "history": 120}),
    min_lookback=340,
)
def dominant_freq_stability(ctx: FeatureCtx, *, window: int, history: int) -> pd.Series:
    # A regime with one persistent cycle keeps its argmax bin; a regime without
    # structure wanders. Reported so that higher means more stable.
    dominant = _rolling_spectrum(
        ctx.log_returns, window, lambda power: float(np.argmax(power)) / power.shape[0]
    )
    wander = dominant.rolling(history, min_periods=history).std(ddof=1)
    return 1.0 / (1.0 + wander)


# ── causal L1 trend filtering ────────────────────────────────────────


def _l1_operator(n: int) -> sp.csc_matrix:
    diagonals = [np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)]
    return sp.diags(diagonals, offsets=[0, 1, 2], shape=(n - 2, n), format="csc")


def _l1_trend_endpoint(
    values: np.ndarray, lam: float, iterations: int, factor, operator
) -> float:
    """Slope at the right edge of an L1-trend fit over one trailing window.

    Solved with ADMM on the second-difference operator. Only the endpoint slope
    is returned; refitting per bar on a trailing window is what keeps the
    textbook whole-sample formulation from becoming look-ahead.
    """
    n = values.shape[0]
    z = np.zeros(n - 2)
    u = np.zeros(n - 2)
    rho = 1.0
    x = values.copy()
    for _ in range(iterations):
        x = factor(values + rho * (operator.T @ (z - u)))
        dx = operator @ x
        target = dx + u
        z = np.sign(target) * np.maximum(np.abs(target) - lam / rho, 0.0)
        u = u + dx - z
    return float(x[-1] - x[-2])


@feature(
    "l1_trend_slope",
    category="experimental",
    doc="Endpoint slope of a trailing L1 trend filter: piecewise-linear drift.",
    params={"window": 120, "lam": 5.0, "stride": 5, "iterations": 60},
    grid=(
        {"window": 60, "lam": 5.0, "stride": 5, "iterations": 60},
        {"window": 120, "lam": 20.0, "stride": 5, "iterations": 60},
        {"window": 250, "lam": 5.0, "stride": 5, "iterations": 60},
    ),
    min_lookback=280,
)
def l1_trend_slope(
    ctx: FeatureCtx, *, window: int, lam: float, stride: int, iterations: int
) -> pd.Series:
    # L1 on the second difference gives a piecewise-linear trend: it holds a
    # slope flat and then breaks it, rather than bending continuously.
    #
    # `stride` recomputes the fit only on bars whose calendar date ordinal is a
    # multiple of it, carrying the last solved value forward in between. That is
    # the difference between minutes and hours over a universe of several
    # hundred names.
    #
    # The trigger is the date, not the bar's position in the array, on purpose:
    # a position-anchored stride would make the value at a given date depend on
    # where the history happened to start, so the same bar would take different
    # values in a 1-year and a 5-year run.
    log_close = ctx.log_close
    values = log_close.to_numpy(dtype=float)
    n = values.shape[0]
    out = np.full(n, np.nan)
    if n < window:
        return pd.Series(out, index=ctx.bars.index)

    operator = _l1_operator(window)
    system = (sp.identity(window, format="csc") + operator.T @ operator).tocsc()
    factor = spla.factorized(system)

    index = ctx.bars.index
    ordinals = np.array([pd.Timestamp(ts).toordinal() for ts in index], dtype=np.int64)
    last = np.nan
    for i in range(window - 1, n):
        if i == window - 1 or ordinals[i] % stride == 0:
            chunk = values[i - window + 1 : i + 1]
            if np.isfinite(chunk).all():
                last = _l1_trend_endpoint(chunk, lam, iterations, factor, operator)
            else:
                last = np.nan
        out[i] = last
    return pd.Series(out, index=ctx.bars.index)
