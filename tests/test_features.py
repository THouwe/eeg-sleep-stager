"""Feature-function tests (features.py) on synthetic signals with known properties."""

from __future__ import annotations

import numpy as np
import pytest

from eeg_sleep_stager.config import BandsConfig
from eeg_sleep_stager.features import band_powers, hjorth, spectral_entropy

FS = 100                       # Hz
N = FS * 30                    # one 30 s epoch = 3000 samples
BANDS = BandsConfig().as_dict()


def _sine(freq: float, n: int = N, fs: int = FS, amp: float = 1.0) -> np.ndarray:
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def _white_noise(n: int = N, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


# --- band_powers ----------------------------------------------------------

def test_band_powers_alpha_dominates_on_10hz_sine():
    powers = band_powers(_sine(10.0), FS, BANDS)
    assert set(powers) == set(BANDS)
    # 10 Hz sits in the alpha band (8-13 Hz) -> alpha should dominate.
    assert max(powers, key=powers.get) == "alpha"
    assert powers["alpha"] > 0.5
    for v in powers.values():
        assert 0.0 <= v <= 1.0


def test_band_powers_delta_dominates_on_1hz_sine():
    powers = band_powers(_sine(1.0), FS, BANDS)
    assert max(powers, key=powers.get) == "delta"


def test_band_powers_constant_signal_is_all_zero():
    powers = band_powers(np.ones(N), FS, BANDS)
    assert all(v == 0.0 for v in powers.values())


# --- spectral_entropy -----------------------------------------------------

def test_spectral_entropy_noise_higher_than_sine():
    ent_noise = spectral_entropy(_white_noise(), FS)
    ent_sine = spectral_entropy(_sine(10.0), FS)
    assert 0.0 <= ent_sine < ent_noise <= 1.0
    # A near-flat spectrum should be close to the maximum.
    assert ent_noise > 0.8
    # A single tone concentrates power -> low entropy.
    assert ent_sine < 0.3


# --- hjorth ---------------------------------------------------------------

def test_hjorth_activity_matches_variance_on_white_noise():
    x = _white_noise(seed=1)
    activity, mobility, complexity = hjorth(x)
    assert activity == pytest.approx(np.var(x))
    assert mobility > 0.0
    assert complexity > 0.0
    assert np.isfinite([activity, mobility, complexity]).all()


def test_hjorth_mobility_increases_with_frequency():
    # Mobility is a proxy for mean frequency: a faster sine -> higher mobility.
    _, mob_slow, _ = hjorth(_sine(2.0))
    _, mob_fast, _ = hjorth(_sine(20.0))
    assert mob_fast > mob_slow


def test_hjorth_complexity_near_one_for_pure_sine():
    # Complexity is ~1 for a single sinusoid (shape closest to a pure tone).
    _, _, complexity = hjorth(_sine(10.0))
    assert complexity == pytest.approx(1.0, abs=0.05)


def test_hjorth_constant_signal_is_zero():
    assert hjorth(np.ones(N)) == (0.0, 0.0, 0.0)
