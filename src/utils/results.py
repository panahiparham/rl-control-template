from collections.abc import Callable, Iterable, Sequence
import importlib
import sqlite3
from pathlib import Path
from PyExpUtils.models.ExperimentDescription import ExperimentDescription, loadExperiment
from PyExpUtils.results.tools import getHeader, getParamsAsDict
from PyExpUtils.results.indices import listIndices
from ml_instrumentation.reader import load_all_results, get_run_ids  # get_run_ids used in detect_missing_indices

import polars as pl

class Result[Exp: ExperimentDescription]:
    def __init__(self, exp_path: str | Path, exp: Exp, metrics: Sequence[str] | None = None):
        self.exp_path = str(exp_path)
        self.exp = exp
        self.metrics = metrics

    def load(self):
        db_path = self.exp.buildSaveContext(0).resolve('results.db')

        if not Path(db_path).exists():
            return None

        df = load_all_results(db_path, self.metrics)
        if df is None or df.is_empty():
            return df

        header = getHeader(self.exp)
        filter_expr = pl.lit(False)
        for param_id in range(self.exp.numPermutations()):
            params = getParamsAsDict(self.exp, param_id, header=header)
            perm_cond = pl.lit(True)
            for k, v in params.items():
                if k in df.columns:
                    perm_cond = perm_cond & (pl.col(k) == v)
            filter_expr = filter_expr | perm_cond

        return df.filter(filter_expr)

    @property
    def filename(self):
        return self.exp_path.split('/')[-1].removesuffix('.json')


class ResultCollection[Exp: ExperimentDescription]:
    def __init__(self, path: str | Path | None = None, metrics: Sequence[str] | None = None, Model: type[Exp] = ExperimentDescription):
        self.metrics = metrics
        self.Model = Model

        if path is None:
            main_file = importlib.import_module('__main__').__file__
            assert main_file is not None
            path = Path(main_file).parent

        self.path = Path(path)

        project = Path.cwd()
        paths = self.path.glob('**/*.json')
        paths = map(lambda p: p.relative_to(project), paths)
        paths = map(str, paths)
        self.paths = list(paths)


    def _result(self, path: str):
        exp = loadExperiment(path, self.Model)
        return Result[Exp](path, exp, self.metrics)


    def get_hyperparameter_columns(self):
        hypers = set[str]()

        for path in self.paths:
            exp = loadExperiment(path, self.Model)
            hypers |= set(getHeader(exp))

        return sorted(hypers)


    def groupby_directory(self, level: int):
        uniques = set(
            p.split('/')[level] for p in self.paths
        )

        for group in uniques:
            group_paths = [p for p in self.paths if p.split('/')[level] == group]
            results = map(self._result, group_paths)
            yield group, list(results)


    def __iter__(self):
        return map(self._result, self.paths)


def _get_max_id(db_path: str | Path) -> int:
    db_path = Path(db_path)
    if not db_path.exists():
        return -1
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    max_id = -1
    for table in tables:
        try:
            row = cur.execute(f'SELECT MAX(id) FROM "{table}"').fetchone()
            if row and row[0] is not None:
                max_id = max(max_id, int(row[0]))
        except Exception:
            pass
    con.close()
    return max_id


def assign_storage_ids(
    db_path: str | Path,
    indices: list[int],
    params_list: list[dict],
) -> dict[int, int]:
    db_path = Path(db_path)

    existing_meta: dict[int, list[dict]] = {}
    if db_path.exists():
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        tables = set(r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall())
        if '_metadata_' in tables:
            rows = cur.execute('SELECT * FROM _metadata_').fetchall()
            col_names = [d[0] for d in cur.description]
            for row in rows:
                row_dict = dict(zip(col_names, row))
                rid = row_dict['id']
                existing_meta.setdefault(rid, []).append(row_dict)
        con.close()

    # Determine which indices collide (same id in DB but different params)
    collisions = []
    for idx, params in zip(indices, params_list):
        existing_rows = existing_meta.get(idx, [])
        col = existing_rows and not any(
            all(r.get(k) == v for k, v in params.items())
            for r in existing_rows
        )
        collisions.append(col)

    # Reserved: all existing DB ids + formula ids used by non-collision indices
    reserved = set(existing_meta.keys()) | {idx for idx, col in zip(indices, collisions) if not col}

    storage_ids: dict[int, int] = {}
    next_id = _get_max_id(db_path) + 1
    for idx, col in zip(indices, collisions):
        if not col:
            storage_ids[idx] = idx
        else:
            while next_id in reserved:
                next_id += 1
            storage_ids[idx] = next_id
            reserved.add(next_id)
            next_id += 1

    return storage_ids


def detect_missing_indices(exp: ExperimentDescription, runs: int, base: str = './'):
    context = exp.buildSaveContext(0, base=base)
    header = getHeader(exp)
    path = context.resolve('results.db')

    if not context.exists('results.db'):
        yield from listIndices(exp, runs)
        return

    n_params = exp.numPermutations()
    for param_id in range(n_params):
        params_dict = getParamsAsDict(exp, param_id, header=header)

        # Query database to get the actual seeds present for this parameter config
        seeds_in_db = set()
        query_succeeded = False
        try:
            conn = sqlite3.connect(path)
            cursor = conn.cursor()

            # Query the _metadata_ table for seeds matching this parameter config
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='_metadata_'")
            if cursor.fetchone():
                query_succeeded = True

                # Build WHERE clause for parameter filtering
                where_parts = []
                where_values = []
                for key, value in params_dict.items():
                    where_parts.append(f"{key} = ?")
                    where_values.append(value)

                if where_parts:
                    where_clause = " AND ".join(where_parts)
                    query = f"SELECT DISTINCT seed FROM _metadata_ WHERE {where_clause}"
                    cursor.execute(query, where_values)

                    for (seed,) in cursor.fetchall():
                        try:
                            seeds_in_db.add(int(seed))
                        except (ValueError, TypeError):
                            pass

            conn.close()
        except Exception:
            # If query fails, query_succeeded stays False
            pass

        # Use metadata-based detection if query succeeded
        if query_succeeded:
            # Check which seeds are missing based on actual metadata
            for seed in range(runs):
                if seed not in seeds_in_db:
                    idx = seed * n_params + param_id
                    yield idx
        else:
            # Fallback: use old idx formula logic only if metadata query failed
            run_ids = set(get_run_ids(path, params_dict))
            for seed in range(runs):
                run_id = seed * n_params + param_id
                if run_id not in run_ids:
                    yield run_id


def gather_missing_indices(experiment_paths: Iterable[str], runs: int, loader: Callable[[str], ExperimentDescription] = loadExperiment, base: str = './'):
    path_to_indices: dict[str, list[int]] = {}

    for path in experiment_paths:
        exp = loader(path)
        indices = detect_missing_indices(exp, runs, base=base)
        indices = sorted(indices)
        path_to_indices[path] = indices

        size = exp.numPermutations() * runs
        print(path, f'{len(indices)} / {size}')

    return path_to_indices
