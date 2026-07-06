import sqlite3
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal
from ml_instrumentation.reader import load_all_results

from utils.results import load_all_results_fast


def _make_db(path: Path) -> None:
    """Two metric tables + metadata, with runs missing from one metric to
    exercise the full-join path."""
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute('CREATE TABLE "return" (frame INTEGER, id INTEGER, measurement REAL)')
    cur.execute('CREATE TABLE "steps" (frame INTEGER, id INTEGER, measurement REAL)')
    for run_id in range(3):
        for frame in (0, 100, 200):
            cur.execute('INSERT INTO "return" VALUES (?, ?, ?)', (frame, run_id, float(run_id + frame)))
            if run_id < 2:  # run 2 has no steps rows
                cur.execute('INSERT INTO "steps" VALUES (?, ?, ?)', (frame, run_id, float(frame)))
    cur.execute('CREATE TABLE _metadata_ ("LR", "seed", "id")')
    for run_id in range(3):
        cur.execute('INSERT INTO _metadata_ VALUES (?, ?, ?)', (0.001 * (run_id + 1), run_id, run_id))
    con.commit()
    con.close()


def _normalized(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(sorted(df.columns)).sort(['id', 'frame'])


class TestLoadAllResultsFast:
    def test_matches_upstream_single_metric(self, tmp_path):
        db = tmp_path / 'results.db'
        _make_db(db)

        expected = load_all_results(db, ['return'])
        actual = load_all_results_fast(db, ['return'])
        assert_frame_equal(_normalized(actual), _normalized(expected))

    def test_matches_upstream_all_metrics(self, tmp_path):
        db = tmp_path / 'results.db'
        _make_db(db)

        expected = load_all_results(db)
        actual = load_all_results_fast(db)
        assert_frame_equal(_normalized(actual), _normalized(expected))
