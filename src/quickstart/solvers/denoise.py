"""Signal denoiser. Baseline: centered moving average, window 9.

Improvement ladder: better windows, Savitzky-Golay, Wiener/wavelet filtering,
anything that lowers held-out MSE. `denoise` must return an array of the same
shape; it sees only the noisy signal.
"""

from __future__ import annotations

import numpy as np


def denoise(noisy: np.ndarray) -> np.ndarray:
    """Return an estimate of the clean signal."""
    window = 9
    kernel = np.ones(window) / window
    padded = np.pad(noisy, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")
