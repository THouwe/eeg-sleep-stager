"""Per-epoch signal features (band powers, Hjorth parameters, spectral entropy).

Pure functions over a 1-D numpy signal; called inside Spark workers and
unit-tested with synthetic signals. Nothing here does I/O or touches Spark, so
the leaf logic stays trivially testable.

Frequency-domain features use a Welch periodogram. The default segment length
gives roughly 0.4 Hz resolution at 100 Hz, which resolves the delta/theta/alpha
edges while averaging enough segments to keep the estimate stable.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

# Analysis band: powers are expressed relative to total power in this range.
_ANALYSIS_FMIN = 0.5
_ANALYSIS_FMAX = 30.0


def _welch_psd(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD estimate. Segment length adapts to short signals."""
    sig = np.asarray(signal, dtype=float)
    nperseg = min(sig.size, max(256, 2 * sample_rate))
    freqs, psd = welch(sig, fs=sample_rate, nperseg=nperseg)
    return freqs, psd


def band_powers(
    signal: np.ndarray,
    sample_rate: int,
    bands: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Relative power in each named frequency band.

    Each band's power is the PSD integrated over ``[low, high)`` divided by the
    total power in the analysis band (0.5-30 Hz). Bands may overlap (sigma sits
    inside beta's neighbourhood), so the relative values are not required to sum
    to 1.

    Returns a dict with the same keys as ``bands``.
    """
    freqs, psd = _welch_psd(signal, sample_rate)
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0

    total_mask = (freqs >= _ANALYSIS_FMIN) & (freqs <= _ANALYSIS_FMAX)
    total = float(psd[total_mask].sum() * df)

    out: dict[str, float] = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        power = float(psd[mask].sum() * df)
        out[name] = power / total if total > 0.0 else 0.0
    return out


def spectral_entropy(signal: np.ndarray, sample_rate: int) -> float:
    """Normalized Shannon entropy of the power spectrum, in [0, 1].

    The PSD (excluding DC) is treated as a probability distribution over
    frequency; its Shannon entropy is normalized by ``log(n_bins)`` so a flat
    spectrum (white noise) approaches 1 and a single-tone spectrum approaches 0.
    """
    freqs, psd = _welch_psd(signal, sample_rate)
    psd = psd[1:]  # drop the DC bin
    total = float(psd.sum())
    if total <= 0.0:
        return 0.0
    p = psd / total
    p = p[p > 0.0]
    if p.size <= 1:
        return 0.0
    entropy = float(-np.sum(p * np.log(p)))
    return entropy / np.log(p.size)


def hjorth(signal: np.ndarray) -> tuple[float, float, float]:
    """Hjorth parameters (activity, mobility, complexity).

    - activity:   variance of the signal (signal power).
    - mobility:   sqrt(var(dx)/var(x)); a proxy for mean frequency.
    - complexity: mobility(dx)/mobility(x); how much the shape departs from a
                  pure sine (=1 for a sine wave).

    Degenerate (constant) inputs yield zeros instead of NaNs/inf.
    """
    x = np.asarray(signal, dtype=float)
    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x = float(np.var(x))
    var_dx = float(np.var(dx))
    var_ddx = float(np.var(ddx))

    activity = var_x
    mobility = np.sqrt(var_dx / var_x) if var_x > 0.0 else 0.0
    mobility_dx = np.sqrt(var_ddx / var_dx) if var_dx > 0.0 else 0.0
    complexity = mobility_dx / mobility if mobility > 0.0 else 0.0

    return float(activity), float(mobility), float(complexity)
