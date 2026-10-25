# Outerloop quickstart

Two small benchmarks for a first [Outerloop](https://github.com/outerloop-science/outerloop)
run. Both are deterministic, CPU-only and evaluate in seconds, so the whole loop
fits on a laptop: an agent proposes a change, the kernel measures it, and a pull
request opens when it improved.

## Try it on your machine

You need Python 3.12 or later, [uv](https://docs.astral.sh/uv/), Git 2.38 or
later, and an Anthropic API key for the Claude author. No GitHub account, no
cluster and no container. (A Codex author needs the agent image; see the install
guide.)

```bash
pip install outerloop-science
outerloop init --forge local --compute local --no-image --target me/quickstart \
  --forge-source https://github.com/outerloop-science/quickstart
outerloop start
```

`init` copies this repository into a git repository under the kernel's state
directory, which stands in for GitHub: pull requests and their threads are
records inside it. It asks for the model and its key. `start` runs the loop in
the foreground; stop it with Ctrl-C and start it again any time. Without a
container image the author's session runs as your user. On macOS a Seatbelt
profile limits what it can read and write; on Linux it can read any file your
account can, so give it a spend-capped key.

The first attempt ends within about half an hour, as a pull request or as a
negative result in the research log. Review it from another terminal:

```bash
outerloop status
outerloop pr list
outerloop pr show 1
outerloop pr comment 1 "why this window size?"   # the author answers on its next wake
outerloop pr merge 1
```

A merged result lands in `BENCHMARKS.md` on the `research-log` branch, measured
by the kernel, never taken from the agent's claim. The repository is ordinary git:
`git clone <state root>/forge/me/quickstart.git` and diff the branches.

To run against GitHub instead, fork this repository and follow
[the install guide](https://github.com/outerloop-science/outerloop/blob/main/docs/install.md).

## The benchmarks

| benchmark | metric | direction | baseline | the ladder above it |
| --- | --- | --- | --- | --- |
| `tsp` | mean tour length over 8 fixed instances of 200 cities | min | 13.876 (nearest neighbor) | 2-opt, or-opt, better starts, Lin-Kernighan-style moves |
| `denoise` | mean squared error over 64 signals x 2 noise draws, redrawn per run | min | 0.01004 (9-sample moving average) | better windows, Savitzky-Golay, Wiener or wavelet filters |

Run them yourself:

```bash
uv sync
uv run python -m quickstart.eval          # both, one JSON line each
uv run python -m quickstart.eval --env tsp
```

`denoise` draws its signals from a seed the eval keeps to itself and prints with
the score; reproduce a run with `QUICKSTART_DENOISE_SEED=<seed>`. The seed is
kept out of the solver's reach on purpose: a solver that could name it could
regenerate the clean signal and score zero.

## What an agent may change

Only `src/quickstart/solvers/`. The instances, the eval harness and the tests are
the ruler, and `.outerloop.yaml` says so; the kernel refuses a change that
touches anything else. `docs/roadmap.md` is what the agents read for direction.

## Then your own benchmark

Copy `.outerloop.yaml` into a repository of yours, point `command` at anything
that prints one JSON object with a number in it, name the metric and its
direction, and list the paths an agent may edit. Everything the contract can say:
[docs/contract.md](https://github.com/outerloop-science/outerloop/blob/main/docs/contract.md).

Licensed MIT. The benchmarks descend from the lab's retired proving ground for the
kernel, at their original baselines.
