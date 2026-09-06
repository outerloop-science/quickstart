"""Frozen eval harness. Not in the agent's allowed scope.

Usage: `python -m quickstart.eval [--env tsp|denoise|all] [--json]`
Prints one JSON object per benchmark: {benchmark, metric, value, direction}.
Each benchmark has a wall-time budget; exceeding it is an error, not a score.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import json
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from quickstart import instances


def _capture_seeds() -> Callable[[str], int]:
    """Claim every run seed while this module imports, strictly before any
    solver module is imported (solver imports are deferred into `_solver`)."""
    seeds = {name: instances.claim_run_seed(name) for name in instances._SEED_ENV_VARS}
    return lambda name: seeds[name]


_run_seed = _capture_seeds()


@functools.cache
def _solver(name: str) -> Any:
    """Import one solver module on first use, after the seeds are claimed."""
    return importlib.import_module(f"quickstart.solvers.{name}")


def _strip_seed(result: dict[str, Any]) -> dict[str, Any]:
    """The solver-facing entry points do not hand the seed back; `main` prints
    it, so the emitted JSON is still reproducible."""
    return {k: v for k, v in result.items() if k != "run_seed"}


TIME_BUDGETS_S = {"tsp": 30.0, "denoise": 60.0}


def tour_length(coords: np.ndarray, tour: list[int]) -> float:
    ordered = coords[np.array(tour)]
    diffs = ordered - np.roll(ordered, -1, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))


def eval_tsp() -> dict[str, Any]:
    lengths = []
    for coords in instances.tsp_instances():
        tour = _solver("tsp").solve(coords)
        if sorted(tour) != list(range(coords.shape[0])):
            raise ValueError("invalid tour: not a permutation of all cities")
        lengths.append(tour_length(coords, tour))
    return {
        "benchmark": "tsp",
        "metric": "mean_tour_length",
        "value": float(np.mean(lengths)),
        "direction": "min",
    }


DENOISE_REFERENCE_WIDTH = 9


def reference_denoise(noisy: np.ndarray) -> np.ndarray:
    """The frozen reference this benchmark starts from: a centered 9-sample
    moving average. Solver-independent, so results/baseline.json can be
    reproduced from the repo alone with the row's recorded run seed."""
    width = DENOISE_REFERENCE_WIDTH
    padded = np.pad(noisy, width // 2, mode="reflect")
    return np.convolve(padded, np.full(width, 1.0 / width), mode="valid")


def score_denoise(estimator: Callable[[np.ndarray], np.ndarray], run_seed: int) -> dict[str, Any]:
    """Mean MSE of `estimator` over this run's signals and noise draws. `stderr`
    is the standard error over signals: the run-to-run noise floor for
    comparing two runs scored under different seeds."""
    per_signal = []
    for signal_seed in instances.denoise_signal_seeds(run_seed):
        clean = instances.denoise_signal(signal_seed)
        mses = []
        for noisy in instances.denoise_signal_draws(clean, signal_seed):
            estimate = estimator(noisy)
            if estimate.shape != clean.shape:
                raise ValueError("denoise output shape mismatch")
            if not np.all(np.isfinite(estimate)):
                raise ValueError("denoise output must be finite")
            mses.append(float(np.mean((estimate - clean) ** 2)))
        per_signal.append(float(np.mean(mses)))
    n = len(per_signal)
    stderr = float(np.std(per_signal, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "benchmark": "denoise",
        "metric": "mean_mse",
        "value": float(np.mean(per_signal)),
        "direction": "min",
        "stderr": stderr,
        "run_seed": run_seed,
        "n_signals": n,
        "n_draws_per_signal": instances.DENOISE_DRAWS_PER_SIGNAL,
    }


def eval_denoise() -> dict[str, Any]:
    return _strip_seed(score_denoise(_solver("denoise").denoise, _run_seed("denoise")))


EVALS = {"tsp": eval_tsp, "denoise": eval_denoise}
SEEDED_ENVS = tuple(instances._SEED_ENV_VARS)


def run_eval(name: str) -> dict[str, Any]:
    """Run one benchmark under its wall-time budget."""
    start = time.perf_counter()
    result = EVALS[name]()
    elapsed = time.perf_counter() - start
    if elapsed > TIME_BUDGETS_S[name]:
        raise ValueError(f"{name}: time budget exceeded ({elapsed:.1f}s > {TIME_BUDGETS_S[name]}s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="all", choices=[*EVALS, "all"])
    parser.add_argument("--json", action="store_true", help="machine-readable output only")
    args = parser.parse_args()
    names = list(EVALS) if args.env == "all" else [args.env]
    for name in names:
        result = run_eval(name)
        if name in SEEDED_ENVS:
            result["run_seed"] = _run_seed(name)  # printed here and nowhere else
        print(json.dumps(result))


if __name__ == "__main__":
    main()
