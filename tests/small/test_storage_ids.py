import sqlite3
import pytest
from pathlib import Path
from utils.results import _get_max_id, assign_storage_ids
from conftest import make_metadata_db


class TestGetMaxId:
    def test_nonexistent_db(self, tmp_path):
        assert _get_max_id(tmp_path / 'nope.db') == -1

    def test_empty_db_no_tables(self, tmp_path):
        db = tmp_path / 'empty.db'
        sqlite3.connect(db).close()
        assert _get_max_id(db) == -1

    def test_returns_max_across_tables(self, tmp_path):
        db = tmp_path / 'test.db'
        con = sqlite3.connect(db)
        con.execute('CREATE TABLE a (id INTEGER, x INTEGER)')
        con.execute('CREATE TABLE b (id INTEGER, y INTEGER)')
        con.executemany('INSERT INTO a VALUES (?,?)', [(0, 1), (3, 2)])
        con.executemany('INSERT INTO b VALUES (?,?)', [(1, 1), (7, 2)])
        con.commit()
        con.close()
        assert _get_max_id(db) == 7

    def test_single_row(self, tmp_path):
        db = tmp_path / 'test.db'
        con = sqlite3.connect(db)
        con.execute('CREATE TABLE t (id INTEGER)')
        con.execute('INSERT INTO t VALUES (42)')
        con.commit()
        con.close()
        assert _get_max_id(db) == 42


class TestAssignStorageIds:
    def test_no_db_returns_formula_ids(self, tmp_path):
        db = tmp_path / 'nope.db'
        indices = [0, 1, 2]
        params = [{'LR': 0.001}, {'LR': 0.01}, {'LR': 0.1}]
        result = assign_storage_ids(db, indices, params)
        assert result == {0: 0, 1: 1, 2: 2}

    def test_same_params_no_collision(self, tmp_path):
        db = tmp_path / 'test.db'
        make_metadata_db(db, [
            {'id': 0, 'seed': 0, 'LR': 0.001, 'TN': 1},
            {'id': 1, 'seed': 0, 'LR': 0.01,  'TN': 1},
        ])
        result = assign_storage_ids(db, [0, 1], [{'LR': 0.001, 'TN': 1}, {'LR': 0.01, 'TN': 1}])
        assert result == {0: 0, 1: 1}

    def test_different_params_triggers_collision(self, tmp_path):
        db = tmp_path / 'test.db'
        make_metadata_db(db, [{'id': 0, 'seed': 0, 'LR': 0.001}])
        # idx=0 in DB has LR=0.001; new run wants LR=0.01 → collision
        result = assign_storage_ids(db, [0], [{'LR': 0.01}])
        assert result[0] == 1  # max_id(0) + 1

    def test_auto_increment_does_not_step_on_non_collision_formula_id(self, tmp_path):
        db = tmp_path / 'test.db'
        # n_params=5, 3 seeds. IDs 0-14. TN values for param_ids 0-4: 1,8,64,512,4096.
        rows = []
        for seed in range(3):
            for pid, tn in enumerate([1, 8, 64, 512, 4096]):
                rows.append({'id': seed * 5 + pid, 'seed': seed, 'TN': tn})
        make_metadata_db(db, rows)

        # Extend: idx=2 (TN=64 in DB), idx=9 (TN=512 in DB), idx=16 (not in DB)
        indices = [2, 9, 16]
        new_params = [{'TN': 32}, {'TN': 32}, {'TN': 32}]
        result = assign_storage_ids(db, indices, new_params)

        assert result[16] == 16                      # no collision → formula id
        assert result[2] != 2                        # collision with TN=64
        assert result[9] != 9                        # collision with TN=512
        assert result[2] != result[9]                # distinct new IDs
        assert 16 not in {result[2], result[9]}      # auto-increment skips reserved 16
        assert len(set(result.values())) == 3        # all unique

    def test_multiple_metadata_rows_one_matching_is_not_collision(self, tmp_path):
        """Duplicate metadata rows: if one row matches current params, it's not a collision."""
        db = tmp_path / 'test.db'
        make_metadata_db(db, [
            {'id': 2, 'seed': 0, 'TN': 64},
            {'id': 2, 'seed': 0, 'TN': 32},  # stale duplicate
        ])
        # Running TN=64 again → one row matches → no collision, reuse id=2
        result = assign_storage_ids(db, [2], [{'TN': 64}])
        assert result[2] == 2

    def test_all_collisions_get_unique_ids(self, tmp_path):
        db = tmp_path / 'test.db'
        make_metadata_db(db, [
            {'id': 0, 'seed': 0, 'LR': 0.001},
            {'id': 1, 'seed': 0, 'LR': 0.01},
        ])
        # Both indices collide (different LR values)
        result = assign_storage_ids(db, [0, 1], [{'LR': 0.1}, {'LR': 0.2}])
        assert result[0] != 0
        assert result[1] != 1
        assert result[0] != result[1]
        assert set(result.values()).isdisjoint({0, 1})
