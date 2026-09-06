import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quickstart import eval as qs_eval
from quickstart import instances
from quickstart.eval import eval_tsp, reference_denoise, run_eval, score_denoise, tour_length
from quickstart.solvers import tsp as tsp_solver


def solver_module(name: str) -> Any:
    return qs_eval._solver(name)


def timed(name: str, score) -> dict[str, Any]:
    start = time.perf_counter()
    result = score()
    assert time.perf_counter() - start <= qs_eval.TIME_BUDGETS_S[name]
    return result


# -- tsp ----------------------------------------------------------------------


def test_tsp_tours_are_valid_permutations() -> None:
    for coords in instances.tsp_instances():
        tour = tsp_solver.solve(coords)
        assert sorted(tour) == list(range(coords.shape[0]))


def test_tsp_eval_within_budget() -> None:
    result = timed("tsp", eval_tsp)
    assert result["direction"] == "min"
    assert 0 < result["value"] < 1000


def test_tour_length_square() -> None:
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert abs(tour_length(coords, [0, 1, 2, 3]) - 4.0) < 1e-12


# -- denoise ------------------------------------------------------------------


def test_denoise_beats_identity(monkeypatch) -> None:
    monkeypatch.setattr(instances, "_ENV_N_SIGNALS", "4")
    raw = []
    for seed in instances.denoise_signal_seeds(1234):
        clean = instances.denoise_signal(seed)
        raw += [
            float(np.mean((noisy - clean) ** 2))
            for noisy in instances.denoise_signal_draws(clean, seed)
        ]
    assert score_denoise(solver_module("denoise").denoise, 1234)["value"] < float(np.mean(raw))


def test_denoise_signals_are_drawn_not_frozen() -> None:
    a, b = instances.denoise_signal(1), instances.denoise_signal(2)
    assert a.shape == (instances.DENOISE_N,) and not np.allclose(a, b)
    assert np.array_equal(a, instances.denoise_signal(1))
    assert instances.denoise_signal_seeds(77) == instances.denoise_signal_seeds(77)
    assert instances.denoise_signal_seeds(77) != instances.denoise_signal_seeds(78)


def test_denoise_frequencies_are_continuous_and_separable() -> None:
    seen: set[float] = set()
    min_sep = instances.DENOISE_MIN_SEP_BINS * instances.DENOISE_FREQ_BIN
    lo, hi = instances.DENOISE_FREQ_RANGE
    for seed in range(200):
        freqs, amps, phases = instances._denoise_components(seed)
        assert instances.DENOISE_COMPONENTS[0] <= freqs.shape[0] <= instances.DENOISE_COMPONENTS[1]
        assert amps.shape == phases.shape == freqs.shape
        assert np.all((freqs >= lo) & (freqs <= hi))
        assert float(np.min(np.diff(freqs))) >= min_sep
        assert not seen & set(freqs.tolist())
        seen |= set(freqs.tolist())


def test_denoise_draws_are_independent_and_reproducible() -> None:
    signal = instances.denoise_signal(11)
    draws = instances.denoise_signal_draws(signal, 11)
    assert len(draws) == instances.DENOISE_DRAWS_PER_SIGNAL
    assert not np.allclose(draws[0], draws[1])
    assert np.array_equal(draws[0], instances.denoise_signal_draws(signal, 11)[0])
    assert abs(float(np.std(draws[0] - signal)) - instances.DENOISE_NOISE_STD) < 0.02


def test_denoise_official_sample_within_budget(monkeypatch) -> None:
    monkeypatch.setattr(instances, "_ENV_N_SIGNALS", None)
    result = timed("denoise", lambda: score_denoise(solver_module("denoise").denoise, 20260809))
    assert result["n_signals"] == instances.DENOISE_N_SIGNALS
    assert result["run_seed"] == 20260809
    assert 0.0 < result["stderr"] < 0.25 * result["value"]


def test_denoise_baseline_row_is_reproducible(monkeypatch) -> None:
    """results/baseline.json is the solver-independent reference at a recorded
    seed, so the ladder's bottom rung can be re-derived from the repo alone."""
    path = Path(__file__).resolve().parents[1] / "results" / "baseline.json"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    row = next(r for r in rows if r["benchmark"] == "denoise")
    monkeypatch.setattr(instances, "_ENV_N_SIGNALS", None)
    scored = score_denoise(reference_denoise, int(row["run_seed"]))
    assert abs(scored["value"] - row["value"]) < 1e-12


# -- the seed discipline -------------------------------------------------------


def test_run_seed_env_vars_are_scrubbed_before_solvers_load() -> None:
    for var in (*instances._SEED_ENV_VARS.values(), "GITHUB_SHA"):
        assert var not in os.environ


def test_run_seeds_are_claimed_exactly_once() -> None:
    for name in instances._SEED_ENV_VARS:
        with pytest.raises(RuntimeError, match="already claimed"):
            instances.claim_run_seed(name)
    with pytest.raises(KeyError):
        instances.claim_run_seed("nonesuch")


def test_run_seed_sources() -> None:
    assert instances._compute_run_seed("4242", "0123456789abcdef") == 4242
    assert instances._compute_run_seed(None, "0123456789abcdef") == int("0123456789ab", 16)
    a, b = instances._compute_run_seed(None, None), instances._compute_run_seed(None, None)
    assert a != b and 0 <= a < 2**31


def test_eval_entry_points_do_not_hand_back_the_run_seed(monkeypatch, capsys) -> None:
    monkeypatch.setattr(instances, "_ENV_N_SIGNALS", "2")
    assert "run_seed" not in qs_eval.eval_denoise()
    monkeypatch.setattr("sys.argv", ["quickstart.eval", "--env", "denoise", "--json"])
    qs_eval.main()
    assert isinstance(json.loads(capsys.readouterr().out.strip())["run_seed"], int)


def test_all_evals_report_schema(monkeypatch) -> None:
    monkeypatch.setattr(instances, "_ENV_N_SIGNALS", "2")
    for name in qs_eval.EVALS:
        result = run_eval(name)
        assert {"benchmark", "metric", "value", "direction"} <= set(result)
        assert result["benchmark"] == name
