# Task Vector Descent

Code for **Task Vector Descent: Learning from Non-IID Batches**. Currently includes
PG-19 continual pretraining; remaining experiments coming soon.

## Setup

Python 3.12 and four CUDA GPUs with BF16 support. Models and data download automatically.
Install a CUDA-compatible PyTorch build, then:

```sh
pip install -e '.[plot]'
export TVD_DATA_ROOT="$PWD/data"
export TVD_RESULTS_ROOT="$PWD/results"
export OMP_NUM_THREADS=4
```

## Train

```sh
torchrun --standalone --nproc_per_node=4 \
  -m experiments.pg19.scripts.train \
  --config experiments/pg19/config/paper/lr4e-5_lambda0p4_seed0.json
```

All 72 paper configurations:

```sh
python -m experiments.pg19.scripts.sweep --run
```

## Plot

From the paper's reference data:

```sh
python -m experiments.pg19.scripts.collect \
  --runs-csv reference/runs.csv --output results/reference
python -m experiments.pg19.scripts.plot \
  --summary results/reference/summary.csv --output results/reference/pg19.png
```

For your own runs, replace `--runs-csv reference/runs.csv` with
`--results-root "$TVD_RESULTS_ROOT/pg19"`. Collection requires the complete 72-run grid.
