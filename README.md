# Outerloop quickstart

Two small benchmarks for a first [Outerloop](https://github.com/outerloop-science/outerloop)
run. Both are deterministic, CPU-only, and evaluate in seconds, so the whole loop,
an agent proposing a change, the kernel measuring it, a pull request opening when
it improved, fits in one sitting on a laptop.

## Try it

You need an API key for the model that will write the code (Claude or Codex), and
a GitHub account that can create an App in some organization or on your user.

1. Fork this repository. The agents open pull requests against your fork.
2. Install the kernel and run the wizard. It asks where the loop runs, which
   repository, which model and its key, and mints the GitHub bot for you:

   ```bash
   pip install outerloop-science
   outerloop init
   ```

3. Start the loop. Without a cluster it runs in the foreground:

   ```bash
   outerloop start
   ```

The first attempt ends within about half an hour, as a pull request on your fork
or as a negative result in the research log the kernel writes to the
`research-log` branch. `BENCHMARKS.md` records every measured improvement.

Step by step, with other model backends and a Slurm cluster:
[docs/install.md](https://github.com/outerloop-science/outerloop/blob/main/docs/install.md).

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
