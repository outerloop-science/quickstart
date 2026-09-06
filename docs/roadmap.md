# Roadmap

What the agents read for direction. Human-written; no agent role edits it.

- **tsp**: the baseline is nearest neighbor from city 0. Local search (2-opt,
  or-opt) is the first rung; the eight instances are fixed and small, so the
  tour length is exact and every improvement is real.
- **denoise**: the baseline is a 9-sample moving average. Signals are sums of
  a few sinusoids with continuous frequencies, redrawn per run, so a method
  must work on the class, not on one instance. Better windows, Savitzky-Golay,
  and Wiener or wavelet filters are the ladder.
