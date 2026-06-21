# CLAUDE.md — rl-control-template

# Purpose
Template for all future RL research projects. Must stay minimal and clean, demonstrating:
1. Experiment definition and management
2. Data collection and analysis
3. Parallel experiment execution locally and on Slurm clusters

# Experiment structure
- Each experiment lives in `experiments/<study>/<env>/`. Each subfolder variation has one or more JSON files, each defining a hyperparameter combination.
- Each JSON defines two hyper dicts: `vmapParameters` (swept and vmapped; e.g. LR, GAMMA) and `staticParameters` (fixed per file; e.g. BUFFER_SIZE, BATCH_SIZE, NETWORK_PRESET). `ExperimentModel` aliases `vmapParameters → metaParameters` internally for PyExpUtils compatibility.

# Running experiments (`main.py`)
- Accepts multiple indices via `-i` and runs them all in a single `jax.jit(jax.vmap(train_fn))` call. All indices in one call must share the same `staticParameters`.
- **Dispatch encoding:** `seed = idx // n_params`, `param_id = idx % n_params`, decoded by PyExpUtils.
- **Storage ID:** assigned by `assign_storage_ids` (`src/utils/results.py`). Equals `idx` unless that ID already exists in the DB with different params (e.g. after a JSON sweep change), in which case it gets `max_existing_id + 1`. This preserves historical runs when `n_params` changes.

# Results
- Stored in `results/` as SQLite databases (`results.db`), one per JSON file.
- Written via `ml-instrumentation`; read via `src/utils/results.py`.
- `Result.load()` filters the loaded DataFrame to only rows matching the current JSON's permutations — stale rows from prior sweep configs are excluded.
- Plotting scripts live alongside the JSON configs in `experiments/`.

# Local parallel execution (`scripts/local.py`)
- Detects missing indices per JSON and dispatches subprocesses.
- CLI args: `--runs`, `-e` (glob patterns), `--cpus` (default: all cores), `--cores-per-task` (default: 1).
- `num_parallel = cpus // cores_per_task` JSON files run simultaneously; each subprocess capped at `cores-per-task` threads via `OMP_NUM_THREADS`.

# Next goal
- Slurm integration: `scripts/slurm.py` for running vmapped sweeps on a cluster.

# Tests
- Small tests: `tests/small/`, run with `pytest tests/small/ -v` from the project root.
- CI: `.github/workflows/test-rl-control-template.yml` — runs on PRs touching this project.

# ml_instrumentation reader API
Key functions from `ml_instrumentation.reader` (`.venv/lib/python3.13/site-packages/ml_instrumentation/reader.py`):
- `load_all_results(db_path, metrics=None, ids=None) -> pl.DataFrame | None` — single query loading all results. Left-joins `_metadata_` table. Columns: `id`, `frame`, one per metric, plus all hyperparameter columns.
- `get_run_ids(db_path, params: dict) -> list[int]` — returns run IDs from `_metadata_` matching all keys in `params` exactly.
