"""Benchmark instances. Part of the ruler: not in the agent's allowed scope.

The TSP instances are fixed. The denoise signals are redrawn per run from a
run seed the eval claims at import and keeps in a closure: a solver that could
name the seed could regenerate the clean signal and score zero, so the seed is
not reachable by accident (captured before any solver imports, its environment
variable scrubbed, no module attribute, one-shot claim). This is not a sandbox;
it makes every leak deliberate rather than easy.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Callable

import numpy as np

TSP_SEED = 20260805
TSP_N_INSTANCES = 8
TSP_N_CITIES = 200

DENOISE_N = 4096
DENOISE_NOISE_STD = 0.3
DENOISE_T_SPAN = 8.0 * np.pi  # observation window, in the signal's time units
DENOISE_N_SIGNALS = 64  # signals per official eval
DENOISE_DRAWS_PER_SIGNAL = 2  # noise draws per signal
DENOISE_COMPONENTS = (2, 4)  # inclusive range of sinusoid counts
DENOISE_FREQ_RANGE = (0.3, 6.0)  # log-uniform, continuous
DENOISE_AMP_RANGE = (0.2, 1.0)
DENOISE_FREQ_BIN = 2.0 * np.pi / DENOISE_T_SPAN
DENOISE_MIN_SEP_BINS = 1.0
DENOISE_MAX_REDRAWS = 256

# Sample-size knob for fast tests, captured once at import so a solver's
# runtime environment writes cannot shrink an official score.
_ENV_N_SIGNALS = os.environ.get("QUICKSTART_DENOISE_SIGNALS")

_SEED_ENV_VARS = {"denoise": "QUICKSTART_DENOISE_SEED"}


def tsp_instances() -> list[np.ndarray]:
    """Fixed set of (n_cities, 2) coordinate arrays in the unit square."""
    rng = np.random.default_rng(TSP_SEED)
    return [rng.random((TSP_N_CITIES, 2)) for _ in range(TSP_N_INSTANCES)]


def _compute_run_seed(override: str | None, sha: str | None) -> int:
    """Override wins (local reproduction); else the commit under test (CI
    determinism); else cryptographic randomness, never the date, which is
    predictable on the day a change is authored. The eval prints the seed it
    used, so a random run is still reproducible afterwards."""
    if override:
        return int(override)
    if sha:
        return int(sha[:12], 16)
    return secrets.randbits(31)


def _capture_run_seeds() -> Callable[[str], int]:
    sha = os.environ.get("GITHUB_SHA")
    unclaimed: dict[str, int | None] = {
        name: _compute_run_seed(os.environ.get(var), sha) for name, var in _SEED_ENV_VARS.items()
    }
    for var in (*_SEED_ENV_VARS.values(), "GITHUB_SHA"):
        os.environ.pop(var, None)

    def claim(name: str) -> int:
        """Return the run seed for `name` and forget it; a second claim raises."""
        if name not in unclaimed:
            raise KeyError(f"no run seed named {name!r}")
        seed = unclaimed[name]
        unclaimed[name] = None
        if seed is None:
            raise RuntimeError(
                f"the {name} run seed was already claimed in this process. The eval "
                "claims each seed once, before any solver is imported; a second claim "
                "means something else took it first. Refusing to score."
            )
        return seed

    return claim


claim_run_seed = _capture_run_seeds()


def _denoise_components(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frequencies, amplitudes, phases) of one clean signal. Frequencies are
    continuous, so there is no harmonic comb to fish for, and at least one
    Fourier bin apart, so every instance is identifiable on this window."""
    rng = np.random.default_rng(seed)
    k_lo, k_hi = DENOISE_COMPONENTS
    k = int(rng.integers(k_lo, k_hi + 1))
    f_lo, f_hi = DENOISE_FREQ_RANGE
    min_sep = DENOISE_MIN_SEP_BINS * DENOISE_FREQ_BIN
    for _ in range(DENOISE_MAX_REDRAWS):
        freqs = np.sort(np.exp(rng.uniform(np.log(f_lo), np.log(f_hi), size=k)))
        if k < 2 or float(np.min(np.diff(freqs))) >= min_sep:
            break
    else:
        raise RuntimeError(f"no separated frequency draw for denoise seed {seed}")
    amps = rng.uniform(*DENOISE_AMP_RANGE, size=k)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=k)
    return freqs, amps, phases


def denoise_signal(seed: int) -> np.ndarray:
    """One clean signal: a sum of 2 to 4 sinusoids. Deterministic in `seed`."""
    freqs, amps, phases = _denoise_components(seed)
    t = np.linspace(0.0, DENOISE_T_SPAN, DENOISE_N)
    return np.sum(amps * np.sin(freqs * t[:, None] + phases), axis=1)


def denoise_signal_seeds(run_seed: int) -> list[int]:
    """The signal seeds one run scores, drawn from its run seed."""
    n_signals = int(_ENV_N_SIGNALS) if _ENV_N_SIGNALS else DENOISE_N_SIGNALS
    rng = np.random.default_rng(run_seed)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n_signals)]


def denoise_signal_draws(signal: np.ndarray, seed: int) -> list[np.ndarray]:
    """Independent noisy observations of one clean signal."""
    return [
        signal + np.random.default_rng([seed, i]).normal(0.0, DENOISE_NOISE_STD, signal.shape)
        for i in range(DENOISE_DRAWS_PER_SIGNAL)
    ]
