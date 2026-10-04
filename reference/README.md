# Paper reference measurements

- `runs.csv`: 72 run-level rows, with no machine paths or run identifiers.
- `summary.csv`: 24 configuration summaries across paired seeds 0, 1, and 2.
- `matched_comparisons.csv`: the 12 partial-integration comparisons from the summary.

These are the numerical exports used for the paper's PG-19 figure, not newly
executed experiments. Every run covers 200 books. `_sd` columns contain sample
standard deviations; divide by the square root of `n_seeds` for the SEM shown
in the figure. The root README explains metric definitions and regeneration.
