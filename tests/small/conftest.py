import sqlite3
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / 'fixtures'
FIXTURE_EXP = FIXTURES_DIR / 'test-exp' / 'TestEnv' / 'dqn.json'


@pytest.fixture
def project_env(tmp_path, monkeypatch):
    """Minimal project root with config.json in a temp dir; chdir there."""
    (tmp_path / 'config.json').write_text('{"save_path": "results/{name}/{desc}"}')
    monkeypatch.chdir(tmp_path)
    return tmp_path


def make_metadata_db(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cols = list(rows[0].keys())
    quoted = ', '.join(f'"{c}"' for c in cols)
    placeholders = ', '.join(['?'] * len(cols))
    cur.execute(f'CREATE TABLE _metadata_ ({quoted})')
    for row in rows:
        cur.execute(f'INSERT INTO _metadata_ ({quoted}) VALUES ({placeholders})',
                    [row[c] for c in cols])
    con.commit()
    con.close()


def make_results_db(path: Path, metric_rows: list[tuple], metadata_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute('CREATE TABLE "return" (frame INTEGER, id INTEGER, measurement REAL)')
    cur.executemany('INSERT INTO "return" VALUES (?, ?, ?)', metric_rows)
    if metadata_rows:
        cols = list(metadata_rows[0].keys())
        quoted = ', '.join(f'"{c}"' for c in cols)
        placeholders = ', '.join(['?'] * len(cols))
        cur.execute(f'CREATE TABLE _metadata_ ({quoted})')
        for row in metadata_rows:
            cur.execute(f'INSERT INTO _metadata_ ({quoted}) VALUES ({placeholders})',
                        [row[c] for c in cols])
    con.commit()
    con.close()
